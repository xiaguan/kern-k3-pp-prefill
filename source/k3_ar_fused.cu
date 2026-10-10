// A DCP step's two all-reduces with the work on either side of them, on the
// Lamport one-shot of peer_allreduce_bf16.cu (TensorRT-LLM's allreduce_fusion
// protocol): every rank pushes its bf16 partial into slot `rank` of every
// peer's stage, a 16-byte word holding a bf16 -0.0 has not arrived, every
// rank sums the slots in rank order in f32 and rounds once. The stage
// rotation and the re-poisoning share peer_allreduce_bf16.cu's `lamport` and
// `state` (the calls of a step may mix the three kernels).
//
// The exchange runs over a row's RV sixteen-byte vectors at a time. Every
// block pushes (and re-poisons) a grid-stride share of all rows' vectors;
// block b < B then waits for row b and runs the epilogue on it, a row per
// 1024 threads with the reduction trees of the kernels it replaces.
//
//   kern_k3_ar_attnres_rms(const bf16* x, u8* lamport, const u64* lamport_peers, i32* state,
//       i32* err, int rank, const bf16* prefix, const bf16* blocks, const f32* sw,
//       const bf16* gamma, bf16* prefix2, bf16* normed, int nb, int snapshot, int B,
//       i64 stage_bytes, i64 timeout_ns)
//     o_proj's partials x [B, H] summed, then k3_residual.cu's K1b (-DLAND_BF16) on the sum.
//
//   kern_k3_ar_finalize_rms(const bf16* fc2, const i32* exp2perm, const f32* wts,
//       const bf16* shared, u8* lamport, const u64* lamport_peers, i32* state, i32* err,
//       int rank, const bf16* gamma, bf16* latent_norm, bf16* shared_sum, int B,
//       i64 stage_bytes, i64 timeout_ns)
//     A row's routed latent partial (k3_moe_prefill.cu's kern_k3_moe_finalize over this
//     rank's experts, bf16) and its shared expert's down partial `shared` [B, H], summed;
//     latent_norm = k3_land.cu's kern_k3_rms(latent sum, gamma), shared_sum the shared sum.
//
//   grid (G >= B, 1, 1), every block resident at once   block (1024, 1, 1)
//   `err` is sticky: 1 + the rank whose data did not show within `timeout_ns`.

#include <cuda_bf16.h>
#include <cstdint>

#ifndef NRANKS
#define NRANKS 8
#endif
#define KH 7168
#define KVEC (KH / 8)
#define KLAT 3584
#define KLATV (KLAT / 8)
#define KNB_MAX 8
#define KEPS 1e-5f
#define KTHREADS 1024
#define TOPK 16

typedef __nv_bfloat16 bf16_t;
typedef __nv_bfloat162 bf162_t;

// ---------------------------------------------------------------- protocol
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ bool poisoned(uint32_t w) { return (w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u; }
__device__ __forceinline__ bool arrived(uint4 v) {
  return !(poisoned(v.x) || poisoned(v.y) || poisoned(v.z) || poisoned(v.w));
}
__device__ __forceinline__ uint32_t unpoison(uint32_t w) {
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
}
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t round_bf16(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t pack2(float a, float b) { return round_bf16(a) | (round_bf16(b) << 16); }

struct Lamport {
  uint4* slot[NRANKS];  // this rank's slot in every peer's stage
  const uint4* mine;    // this rank's stage
  uint4* clear_buf;
  long long clear_count, tot;
  int flag;
};

__device__ __forceinline__ Lamport lamport_open(uint8_t* lamport, const unsigned long long* peers, int* state,
                                                int rank, long long tot, long long stage_bytes) {
  Lamport l;
  l.flag = state[2];
  l.clear_count = *reinterpret_cast<long long*>(state + 4);
  l.tot = tot;
#pragma unroll
  for (int q = 0; q < NRANKS; ++q)
    l.slot[q] = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[q]) + (l.flag % 3) * stage_bytes) +
                (long long)rank * tot;
  l.mine = reinterpret_cast<const uint4*>(lamport + (l.flag % 3) * stage_bytes);
  l.clear_buf = reinterpret_cast<uint4*>(lamport + ((l.flag + 2) % 3) * stage_bytes);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(state, 1);
  return l;
}

__device__ __forceinline__ void lamport_push(const Lamport& l, long long i, uint4 v) {
  v = make_uint4(unpoison(v.x), unpoison(v.y), unpoison(v.z), unpoison(v.w));
#pragma unroll
  for (int q = 0; q < NRANKS; ++q) l.slot[q][i] = v;
}

__device__ __forceinline__ void lamport_clear(const Lamport& l) {
  const uint4 poison = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < l.clear_count;
       i += (long long)gridDim.x * blockDim.x)
    l.clear_buf[i] = poison;
}

// The rank-order f32 sum of vector i over the slots, rounded once.
__device__ __forceinline__ uint4 lamport_sum(const Lamport& l, long long i, long long timeout_ns, int& fail) {
  uint4 vals[NRANKS];
  unsigned long long t0 = 0;
  while (true) {
    int missing = 0;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      vals[r] = ld_volatile(l.mine + (long long)r * l.tot + i);
      if (missing == 0 && !arrived(vals[r])) missing = 1 + r;
    }
    if (missing == 0) break;
    const unsigned long long now = gtimer();
    if (t0 == 0) {
      t0 = now;
    } else if ((long long)(now - t0) > timeout_ns) {
      fail = missing;
      break;
    }
  }
  float acc[8];
  acc[0] = lo(vals[0].x), acc[1] = hi(vals[0].x), acc[2] = lo(vals[0].y), acc[3] = hi(vals[0].y);
  acc[4] = lo(vals[0].z), acc[5] = hi(vals[0].z), acc[6] = lo(vals[0].w), acc[7] = hi(vals[0].w);
#pragma unroll
  for (int r = 1; r < NRANKS; ++r) {
    acc[0] += lo(vals[r].x), acc[1] += hi(vals[r].x), acc[2] += lo(vals[r].y), acc[3] += hi(vals[r].y);
    acc[4] += lo(vals[r].z), acc[5] += hi(vals[r].z), acc[6] += lo(vals[r].w), acc[7] += hi(vals[r].w);
  }
  return make_uint4(pack2(acc[0], acc[1]), pack2(acc[2], acc[3]), pack2(acc[4], acc[5]), pack2(acc[6], acc[7]));
}

__device__ __forceinline__ void lamport_close(const Lamport& l, int* state, int* err, int fail) {
  if (fail) atomicMax(err, fail);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (*reinterpret_cast<volatile int*>(state) != (int)gridDim.x) {
    }
    state[2] = (l.flag + 1) % 3;
    *reinterpret_cast<long long*>(state + 4) = (long long)NRANKS * l.tot;
    state[0] = 0;
  }
}

// ---------------------------------------------------------------- row math (k3_residual.cu)
struct V8 {
  unsigned w[4];
};
__device__ __forceinline__ V8 ldv(const void* p) {
  const uint4 t = *(const uint4*)p;
  V8 v;
  v.w[0] = t.x, v.w[1] = t.y, v.w[2] = t.z, v.w[3] = t.w;
  return v;
}
__device__ __forceinline__ V8 asv(uint4 t) {
  V8 v;
  v.w[0] = t.x, v.w[1] = t.y, v.w[2] = t.z, v.w[3] = t.w;
  return v;
}
__device__ __forceinline__ void stv(void* p, const V8& v) { *(uint4*)p = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]); }
__device__ __forceinline__ bf162_t as_bf162(unsigned u) {
  return __halves2bfloat162(__ushort_as_bfloat16((unsigned short)(u & 0xffffu)),
                            __ushort_as_bfloat16((unsigned short)(u >> 16)));
}
__device__ __forceinline__ unsigned from_bf162(bf162_t h) {
  return (unsigned)(unsigned short)__bfloat16_as_ushort(__low2bfloat16(h)) |
         ((unsigned)(unsigned short)__bfloat16_as_ushort(__high2bfloat16(h)) << 16);
}
__device__ __forceinline__ float2 bf2f(unsigned u) { return __bfloat1622float2(as_bf162(u)); }
__device__ __forceinline__ unsigned f2bf(float2 f) { return from_bf162(__float22bfloat162_rn(f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// k3_residual.cu's attnres_rms_row, verbatim.
__device__ __forceinline__ void attnres_rms_row(const bf16_t* __restrict__ blk_row, const V8 pv,
                                                const float* __restrict__ sw, const bf16_t* __restrict__ gamma,
                                                bf16_t* __restrict__ normed_row, int nb, int t) {
  const int lane = t & 31, warp = t >> 5;
  const bool act = (t < KVEC);

  __shared__ float s_red[(KNB_MAX + 1) * 2 * 32];
  __shared__ float s_val[(KNB_MAX + 1) * 2];
  __shared__ float s_red2[32];

  const int ncand = nb + 1;
  V8 mixed;

  if (nb == 0) {
    mixed = pv;
  } else {
    float swv[8];
    if (act) {
      const float4* sp4 = (const float4*)(sw + t * 8);
      const float4 a = sp4[0], b = sp4[1];
      swv[0] = a.x; swv[1] = a.y; swv[2] = a.z; swv[3] = a.w;
      swv[4] = b.x; swv[5] = b.y; swv[6] = b.z; swv[7] = b.w;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) swv[j] = 0.f;
    }
#pragma unroll
    for (int c0 = 0; c0 < KNB_MAX; c0 += 2) {
      V8 xx[2];
#pragma unroll
      for (int g = 0; g < 2; ++g)
        if (act && c0 + g < nb) xx[g] = ldv(blk_row + (size_t)(c0 + g) * KH + t * 8);
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const int c = c0 + g;
        if (c >= nb) continue;
        float sq = 0.f, dp = 0.f;
        if (act) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 f = bf2f(xx[g].w[j]);
            sq += f.x * f.x; sq += f.y * f.y;
            dp += f.x * swv[2 * j]; dp += f.y * swv[2 * j + 1];
          }
        }
        sq = warp_sum(sq);
        dp = warp_sum(dp);
        if (lane == 0) { s_red[(c * 2) * 32 + warp] = sq; s_red[(c * 2 + 1) * 32 + warp] = dp; }
      }
    }
    {
      float sq = 0.f, dp = 0.f;
      if (act) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = bf2f(pv.w[j]);
          sq += f.x * f.x; sq += f.y * f.y;
          dp += f.x * swv[2 * j]; dp += f.y * swv[2 * j + 1];
        }
      }
      sq = warp_sum(sq);
      dp = warp_sum(dp);
      if (lane == 0) { s_red[(nb * 2) * 32 + warp] = sq; s_red[(nb * 2 + 1) * 32 + warp] = dp; }
    }
    __syncthreads();
    if (warp < 2 * ncand) {
      const float v = warp_sum(s_red[warp * 32 + lane]);
      if (lane == 0) s_val[warp] = v;
    }
    __syncthreads();
    float sc = -3.0e38f;
    if (lane < ncand) {
      const float sqv = s_val[2 * lane], dpv = s_val[2 * lane + 1];
      sc = dpv * rsqrtf(sqv * (1.0f / (float)KH) + KEPS);
    }
    float mx = sc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float ex = (lane < ncand) ? __expf(sc - mx) : 0.f;
    float den = ex;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    const float pmine = ex / den;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    if (act) {
#pragma unroll
      for (int c = 0; c < KNB_MAX; ++c) {
        if (c >= nb) continue;
        const V8 x = ldv(blk_row + (size_t)c * KH + t * 8);
        const float p = __shfl_sync(0xffffffffu, pmine, c);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = bf2f(x.w[j]);
          acc[2 * j] += p * f.x;
          acc[2 * j + 1] += p * f.y;
        }
      }
      const float p = __shfl_sync(0xffffffffu, pmine, nb);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf2f(pv.w[j]);
        acc[2 * j] += p * f.x;
        acc[2 * j + 1] += p * f.y;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) mixed.w[j] = f2bf(make_float2(acc[2 * j], acc[2 * j + 1]));
  }

  float sq = 0.f;
  if (act) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = bf2f(mixed.w[j]);
      sq += f.x * f.x; sq += f.y * f.y;
    }
  }
  sq = warp_sum(sq);
  if (lane == 0) s_red2[warp] = sq;
  __syncthreads();
  const float tot = warp_sum(s_red2[lane]);
  const float r = rsqrtf(tot * (1.0f / (float)KH) + KEPS);
  if (act) {
    const V8 g = ldv(gamma + t * 8);
    V8 o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = bf2f(mixed.w[j]);
      f.x *= r; f.y *= r;
      o.w[j] = from_bf162(__hmul2(__float22bfloat162_rn(f), as_bf162(g.w[j])));
    }
    stv(normed_row + t * 8, o);
  }
}

// ---------------------------------------------------------------- attention all-reduce + K1b
extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_ar_attnres_rms(
    const uint4* __restrict__ x, uint8_t* lamport, const unsigned long long* lamport_peers, int* state, int* err,
    int rank, const bf16_t* __restrict__ prefix, const bf16_t* __restrict__ blocks, const float* __restrict__ sw,
    const bf16_t* __restrict__ gamma, bf16_t* __restrict__ prefix2, bf16_t* __restrict__ normed, int nb, int snapshot,
    int B, long long stage_bytes, long long timeout_ns) {
  const int t = threadIdx.x, b = blockIdx.x;
  const long long tot = (long long)B * KVEC;
  const Lamport l = lamport_open(lamport, lamport_peers, state, rank, tot, stage_bytes);
  for (long long i = (long long)b * KTHREADS + t; i < tot; i += (long long)gridDim.x * KTHREADS) lamport_push(l, i, x[i]);
  lamport_clear(l);

  int fail = 0;
  if (b < B) {
    const bool act = t < KVEC;
    V8 pv;
    pv.w[0] = 0u, pv.w[1] = 0u, pv.w[2] = 0u, pv.w[3] = 0u;
    if (act) {
      const size_t off = (size_t)b * KH + t * 8;
      const V8 lp = asv(lamport_sum(l, (long long)b * KVEC + t, timeout_ns, fail));
      if (snapshot) {
        pv = lp;
      } else {
        const V8 pr = ldv(prefix + off);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 rf = bf2f(pr.w[j]), lf = bf2f(lp.w[j]);
          pv.w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
        }
      }
      stv(prefix2 + off, pv);
    }
    attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, sw, gamma, normed + (size_t)b * KH, nb, t);
  }
  lamport_close(l, state, err, fail);
}

// ---------------------------------------------------------------- MoE finalize + all-reduce + latent rms
// A row's exchange: the latent's KLATV vectors, then the shared expert's KVEC.
#define ROWV (KLATV + KVEC)

// kern_k3_moe_finalize's column vector h of row t
__device__ __forceinline__ uint4 finalize(const bf16_t* __restrict__ fc2, const int* __restrict__ exp2perm,
                                          const float* __restrict__ wts, int t, int h) {
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const int p = exp2perm[t * TOPK + k];
    if (p < 0) continue;
    const float w = wts[t * TOPK + k];
    const uint4 u = reinterpret_cast<const uint4*>(fc2 + (long long)p * KLAT)[h];
    const __nv_bfloat162* v = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(v[j]);
      acc[2 * j] += w * f.x;
      acc[2 * j + 1] += w * f.y;
    }
  }
  uint4 o;
  __nv_bfloat162* ov = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int j = 0; j < 4; ++j) ov[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
  return o;
}

extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_ar_finalize_rms(
    const bf16_t* __restrict__ fc2, const int* __restrict__ exp2perm, const float* __restrict__ wts,
    const uint4* __restrict__ shared, uint8_t* lamport, const unsigned long long* lamport_peers, int* state, int* err,
    int rank, const bf16_t* __restrict__ gamma, bf16_t* __restrict__ latent_norm, uint4* __restrict__ shared_sum, int B,
    long long stage_bytes, long long timeout_ns) {
  __shared__ float sm[33];
  const int t = threadIdx.x, b = blockIdx.x;
  const long long tot = (long long)B * ROWV;
  const Lamport l = lamport_open(lamport, lamport_peers, state, rank, tot, stage_bytes);
  for (long long i = (long long)b * KTHREADS + t; i < tot; i += (long long)gridDim.x * KTHREADS) {
    const int row = (int)(i / ROWV), c = (int)(i % ROWV);
    lamport_push(l, i, c < KLATV ? finalize(fc2, exp2perm, wts, row, c) : shared[(long long)row * KVEC + c - KLATV]);
  }
  lamport_clear(l);

  int fail = 0;
  if (b < B) {
    const long long base = (long long)b * ROWV;
    if (t < KVEC) shared_sum[(long long)b * KVEC + t] = lamport_sum(l, base + KLATV + t, timeout_ns, fail);
    // kern_k3_rms at h = KLAT: vector t of the row, its squares serially, a warp
    // butterfly, then the 32 warp slots (zero past the row) by another
    uint4 x = make_uint4(0u, 0u, 0u, 0u), g;
    float sum = 0.f;
    if (t < KLATV) {
      x = lamport_sum(l, base + t, timeout_ns, fail);
      g = reinterpret_cast<const uint4*>(gamma)[t];
      const V8 v = asv(x);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf2f(v.w[j]);
        sum += f.x * f.x;
        sum += f.y * f.y;
      }
    }
    sum = warp_sum(sum);
    const int lane = t & 31, warp = t >> 5;
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (warp == 0) {
      float v = sm[lane];
      v = warp_sum(v);
      if (lane == 0) sm[32] = v;
    }
    __syncthreads();
    const float rs = rsqrtf(sm[32] / (float)KLAT + 1e-5f);
    if (t < KLATV) {
      const V8 v = asv(x), gv = asv(g);
      V8 o;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf2f(v.w[j]);
        o.w[j] = from_bf162(__hmul2(__floats2bfloat162_rn(f.x * rs, f.y * rs), as_bf162(gv.w[j])));
      }
      stv(latent_norm + (long long)b * KLAT + t * 8, o);
    }
  }
  lamport_close(l, state, err, fail);
}
