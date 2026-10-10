// The front of a prefill chunk's MoE in one launch: the router top-k, the
// latent in mxfp8, the routing tables of TRT-LLM gen's batched GEMMs and the
// shared expert's activation, where they were six launches (router_topk,
// moe_quant, FlashInfer's init / histogram / offsets, situ).
//
//   kern_k3g_moe_route(S, bias, rs, x, ldx, gu, ldgu, q, sf, idx, wts, act, sync, cta_batch, cta_limit,
//                      num_non_exiting, total_padded, route_map, exp2perm, tile, B)
//     S f32 [B, EXPERTS] router logits; bias f32 [EXPERTS]; rs bf16 [1];
//     x bf16 [B, ldx] the latent; gu bf16 [B, ldgu] the shared expert's gate | up;
//     q u8 [B, LATENT], sf u8 [B, LATENT / 32]: k3_moe_prefill.cu's kern_k3_moe_quant;
//     idx i32 / wts f32 [B, 16]: k3_prefill_glue.cu's router top-k, bit for bit;
//     act bf16 [B, SHARED]: the shared expert's activation, k3_prefill_glue.cu's kern_k3g_situ;
//     sync u32 [2 + 2 * TABLE]: the grid barrier (arrivals, generation) and two routing tables
//       (Table below), the generation's parity picking this call's; zeroed by
//       kern_k3g_moe_route_init before a stage's first call, every call of a stage's B rows
//       leaves the barrier and the next call's table zero;
//     the tables FlashInfer's routing writes for a rank holding every expert, `tile` rows a CTA.
//   grid (64, 1, 1) up to 256 rows, then (304, 1, 1)   block (512, 1, 1): every block resident
//   (two an SM) for the grid barrier
//
//   kern_k3g_moe_route_init(sync)
//   grid (any, 1, 1)   block (1024, 1, 1)
//
// Phase 1: a warp a row's top-k (a grid-stride over rows when B passes 16 *
// gridDim), its picks marked in this call's table (each expert's count, its
// count per 256-row chunk, its rows' bitmap: fire-and-forget atomics); then
// every warp a share of the rows' 256-column pieces of the latent to quantise
// and of the shared activation; the next call's table zeroed. A grid barrier: every
// block is resident, the generation counts the calls. Phase 2: every block
// scans the counts into the experts' CTA offsets and writes its share of the
// CTA tables; a pick's place among its expert's rows is the picks in the
// chunks and the words before its row's, so every expert's rows are in row
// order (the batched GEMM gathers them faster so than in an atomic's order)
// and the tables are the same bytes every run.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef EXPERTS
#define EXPERTS 896
#endif
#define TOPK 16
#define LATENT 3584
#define SHARED 6144
#define WARPS 16
#define THREADS (WARPS * 32)
#define TK_PER (EXPERTS / 32)
static_assert(EXPERTS % 128 == 0 && EXPERTS <= 2 * THREADS, "float4s of experts a lane, two experts a thread in the scan");

typedef __nv_bfloat16 bf16_t;
typedef unsigned int u32;

__device__ __forceinline__ u32 ord_f32(float f) {
  const u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ u32 ld_acquire(const u32* p) {
  u32 v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release(u32* p, u32 v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// 1 / d rounded as `1.0f / d` and __frcp_rn round it, without their slow path's branch:
// equal to both for every d in [2^-126, 2^126) (checked over all of them); the callers
// take it only when no lane of the warp holds a d outside.
__device__ __forceinline__ float rcp_fast(float d) {
  float q;
  asm("{\n .reg .f32 r, e;\n rcp.approx.ftz.f32 r, %1;\n fma.rn.f32 e, %1, r, 0fBF800000;\n neg.f32 e, e;\n"
      " fma.rn.f32 %0, r, e, r;\n}"
      : "=f"(q)
      : "f"(d));
  return q;
}
__device__ __forceinline__ bool rcp_ok(float d) { return d >= 0x1p-126f && d < 0x1p126f; }

// a / 448.f without its slow path's branch: equal for every bf16 magnitude but infinity.
__device__ __forceinline__ float div448(float a) {
  float q;
  asm("{\n .reg .f32 r, t, q0, m;\n mov.f32 r, 0f3B124925;\n fma.rn.f32 t, r, 0fC3E00000, 0f3F800000;\n"
      " fma.rn.f32 r, t, r, r;\n fma.rn.f32 q0, %1, r, 0f00000000;\n fma.rn.f32 m, q0, 0fC3E00000, %1;\n"
      " fma.rn.f32 %0, r, m, q0;\n}"
      : "=f"(q)
      : "f"(a));
  return a == __int_as_float(0x7f800000) ? a : q;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// kern_k3g_router_topk's picks: the 16 largest keys sigmoid(S) + bias, ties to
// the smaller expert, lane r < 16 ending with the r-th and its sigmoid. Lane l
// holds experts l * TK_PER + j. Every pick is at least the 16th largest of the
// lanes' maxima (16 lanes hold a key that large), so the keys that clear it
// are the candidates (16 or more; past CAND, the original 16 rounds), ranked by
// key then expert into their places.
#define CAND 64
#ifndef QBATCH
#define QBATCH 8  // quant pieces a warp loads at once
#endif
#ifndef SBATCH
#define SBATCH 4  // situ pieces a warp loads at once
#endif
__device__ __forceinline__ void topk_row(const float* __restrict__ row, const float* __restrict__ bias, int lane,
                                         u32* __restrict__ ck, int* __restrict__ ce, int& pick_e, float& pick_w) {
  u32 o[TK_PER];
  bool ok = true;
#pragma unroll
  for (int j = 0; j < TK_PER; j += 4) {
    const float4 sv = *reinterpret_cast<const float4*>(row + lane * TK_PER + j);
    const float s4[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float d = 1.0f + expf(-s4[i]);
      ok &= rcp_ok(d);
      o[j + i] = __float_as_uint(d);
    }
  }
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int j = 0; j < TK_PER; ++j) o[j] = __float_as_uint(rcp_fast(__uint_as_float(o[j])));
  } else {
#pragma unroll
    for (int j = 0; j < TK_PER; ++j) o[j] = __float_as_uint(1.0f / __uint_as_float(o[j]));
  }
#pragma unroll
  for (int j = 0; j < TK_PER; j += 4) {
    const float4 bv = *reinterpret_cast<const float4*>(bias + lane * TK_PER + j);
    const float b4[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) o[j + i] = ord_f32(__uint_as_float(o[j + i]) + b4[i]);
  }
  u32 v = o[0];
#pragma unroll
  for (int j = 1; j < TK_PER; ++j) v = max(v, o[j]);
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const u32 p = __shfl_xor_sync(0xffffffffu, v, j);
      v = (((lane & j) == 0) == ((lane & k) == 0)) ? min(v, p) : max(v, p);
    }
  const u32 tau = __shfl_sync(0xffffffffu, v, 16);
  int c = 0;
#pragma unroll
  for (int j = 0; j < TK_PER; ++j) c += o[j] >= tau;
  int inc = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc, d);
    if (lane >= d) inc += y;
  }
  const int n = __shfl_sync(0xffffffffu, inc, 31);
  pick_e = 0;
  if (n <= CAND) {
    int at = inc - c;
#pragma unroll
    for (int j = 0; j < TK_PER; ++j)
      if (o[j] >= tau) {
        ck[at] = o[j];
        ce[at] = lane * TK_PER + j;
        ++at;
      }
    __syncwarp();
    int rk[CAND / 32], ex[CAND / 32];
#pragma unroll
    for (int s = 0; s < CAND / 32; ++s) {
      const int i = lane + 32 * s;
      rk[s] = TOPK;
      if (i < n) {
        const u32 ki = ck[i];
        const int ei = ce[i];
        int r = 0;
        for (int j = 0; j < n; ++j) {
          const u32 kj = ck[j];
          r += (kj > ki) | ((kj == ki) & (ce[j] < ei));
        }
        rk[s] = r;
        ex[s] = ei;
      }
    }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < CAND / 32; ++s)
      if (rk[s] < TOPK) ce[rk[s]] = ex[s];
    __syncwarp();
    if (lane < TOPK) pick_e = ce[lane];
    __syncwarp();
  } else {
    for (int r = 0; r < TOPK; ++r) {
      u32 m = o[0];
#pragma unroll
      for (int j = 1; j < TK_PER; ++j) m = max(m, o[j]);
      m = __reduce_max_sync(0xffffffffu, m);
      u32 my = 0xffffffffu;
#pragma unroll
      for (int j = TK_PER - 1; j >= 0; --j)
        if (o[j] == m) my = (u32)(lane * TK_PER + j);
      const u32 ew = __reduce_min_sync(0xffffffffu, my);
      if (my == ew)
#pragma unroll
        for (int j = 0; j < TK_PER; ++j)
          if (lane * TK_PER + j == (int)ew) o[j] = 0u;
      if (lane == r) pick_e = (int)ew;
    }
  }
  pick_w = lane < TOPK ? sigmoid(row[pick_e]) : 0.f;
}

__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// k3_prefill_glue.cu's kern_k3g_situ: act = situ(gate, up) on 8 columns; `fast` when every
// lane's sigmoid denominators are in rcp_fast's range.
__device__ __forceinline__ float situ(float g, float u, bool fast) {
  const float a = 4.0f * tanh_approx(g * 0.25f);
  const float d = 1.0f + __expf(-g);
  const float s = fast ? rcp_fast(d) : __frcp_rn(d);
  const float c = 25.0f * tanh_approx(u * 0.04f);
  return (a * s) * c;
}

__device__ __forceinline__ void situ8(const uint4 g, const uint4 u, bf16_t* __restrict__ act) {
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&g);
  const __nv_bfloat162* upp = reinterpret_cast<const __nv_bfloat162*>(&u);
  uint4 o;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
  bool ok = true;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const float2 gf = __bfloat1622float2(gp[l]);
    ok &= rcp_ok(1.0f + __expf(-gf.x)) & rcp_ok(1.0f + __expf(-gf.y));
  }
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const float2 gf = __bfloat1622float2(gp[l]), uf = __bfloat1622float2(upp[l]);
      op[l] = __float22bfloat162_rn(make_float2(situ(gf.x, uf.x, true), situ(gf.y, uf.y, true)));
    }
  } else {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const float2 gf = __bfloat1622float2(gp[l]), uf = __bfloat1622float2(upp[l]);
      op[l] = __float22bfloat162_rn(make_float2(situ(gf.x, uf.x, false), situ(gf.y, uf.y, false)));
    }
  }
  *reinterpret_cast<uint4*>(act) = o;
}

// kern_k3_moe_quant on 8 elements, four lanes a 32-element group.
__device__ __forceinline__ void quant8(const uint4 u, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int lane) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
  float2 v[4];
  float amax = 0.f;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    v[l] = __bfloat1622float2(p[l]);
    amax = fmaxf(amax, fmaxf(fabsf(v[l].x), fabsf(v[l].y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  int e = amax > 0.f ? (int)ceilf(log2f(div448(amax))) : 0;
  e = min(max(e, -127), 127);
  const float s = exp2f((float)-e);
  if ((lane & 3) == 0) *sf = (uint8_t)(e + 127);
  uint2 w;
  unsigned short* h = reinterpret_cast<unsigned short*>(&w);
#pragma unroll
  for (int l = 0; l < 4; ++l)
    h[l] = __nv_cvt_float2_to_fp8x2(make_float2(v[l].x * s, v[l].y * s), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint2*>(q) = w;
}

// The routing state, one table per parity of the generation: every expert's
// pick count, its picks per 256-row chunk, and its rows as a bitmap.
#define MAX_ROWS 8192
#define CHUNK 256
#define CHUNKS (MAX_ROWS / CHUNK)
#define WORDS (MAX_ROWS / 32)
#define TABLE (EXPERTS * (1 + CHUNKS + WORDS))
static_assert(CHUNKS <= 32 && CHUNK / 32 <= 32, "a chunk prefix is one warp sum");
struct Table {
  int* tot;   // [EXPERTS]
  int* chunk;  // [EXPERTS][CHUNKS]
  u32* bits;   // [EXPERTS][WORDS]
};
__device__ __forceinline__ Table table(u32* sync, u32 parity) {
  u32* t = sync + 2 + parity * TABLE;
  return {reinterpret_cast<int*>(t), reinterpret_cast<int*>(t + EXPERTS), t + EXPERTS * (1 + CHUNKS)};
}

extern "C" __global__ void __launch_bounds__(THREADS, 2) kern_k3g_moe_route(
    const float* __restrict__ S, const float* __restrict__ bias, const bf16_t* __restrict__ rs,
    const bf16_t* __restrict__ x, int ldx, const bf16_t* __restrict__ gu, int ldgu, uint8_t* __restrict__ q,
    uint8_t* __restrict__ sf, int* __restrict__ idx, float* __restrict__ wts, bf16_t* __restrict__ act,
    u32* __restrict__ sync, int* __restrict__ cta_batch, int* __restrict__ cta_limit,
    int* __restrict__ num_non_exiting, int* __restrict__ total_padded, int* __restrict__ route_map,
    int* __restrict__ exp2perm, int tile, int B) {
  __shared__ u32 gen_s;
  __shared__ int off[EXPERTS], cnt[EXPERTS];
  __shared__ int wsum[WARPS];
  __shared__ u32 ck[WARPS][CAND];
  __shared__ int ce[WARPS][CAND];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int nthreads = gridDim.x * THREADS, gt = blockIdx.x * THREADS + t;
  const int rstride = gridDim.x * WARPS, r0 = blockIdx.x * WARPS + warp;

  if (t == 0) gen_s = ld_acquire(sync + 1);
  __syncthreads();
  const u32 gen = gen_s;
  const Table cur = table(sync, gen & 1), next = table(sync, ~gen & 1);
  // the next call's table: what a call of these B rows marks
  {
    const int nc = (B + CHUNK - 1) / CHUNK, nw = (B + 31) / 32;
    for (int i = gt; i < EXPERTS * (1 + nc + nw); i += nthreads) {
      const int e = i % EXPERTS, j = i / EXPERTS;
      if (j == 0)
        next.tot[e] = 0;
      else if (j <= nc)
        next.chunk[e * CHUNKS + j - 1] = 0;
      else
        next.bits[e * WORDS + j - 1 - nc] = 0;
    }
  }

  const float rsv = __bfloat162float(rs[0]);
  for (int b = r0; b < B; b += rstride) {
    int pick_e;
    float pick_w;
    topk_row(S + (long long)b * EXPERTS, bias, lane, ck[warp], ce[warp], pick_e, pick_w);
    float den = pick_w;
#pragma unroll
    for (int o = TOPK / 2; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    if (lane < TOPK) {
      wts[b * TOPK + lane] = (pick_w / (den + 1e-20f)) * rsv;
      idx[b * TOPK + lane] = pick_e;
      atomicAdd(cur.tot + pick_e, 1);
      atomicAdd(cur.chunk + pick_e * CHUNKS + b / CHUNK, 1);
      atomicOr(cur.bits + pick_e * WORDS + b / 32, 1u << (b & 31));
    }
  }
  // the rows' latent and shared activation, 256 columns a warp a pass over every warp, the first
  // pieces to the warps that had no top-k row (on a short chunk, the top-k and these overlap)
  const int nq = B * (LATENT / 256), ns = B * (SHARED / 256);
  const int i0 = (r0 + rstride - min(B, rstride)) % rstride;
  // a batch of pieces' loads in flight before any of the warp-wide votes and shuffles that would hold
  // the next piece's loads back
  for (int i = i0; i < nq; i += QBATCH * rstride) {
    uint4 u[QBATCH];
#pragma unroll
    for (int k = 0; k < QBATCH; ++k) {
      const int ik = min(i + k * rstride, nq - 1);
      const int b = ik / (LATENT / 256), c = ik % (LATENT / 256) * 256 + lane * 8;
      u[k] = *reinterpret_cast<const uint4*>(x + (long long)b * ldx + c);
    }
#pragma unroll
    for (int k = 0; k < QBATCH; ++k) {
      const int ik = i + k * rstride;
      if (ik >= nq) break;
      const int b = ik / (LATENT / 256), c = ik % (LATENT / 256) * 256 + lane * 8;
      quant8(u[k], q + (long long)b * LATENT + c, sf + (long long)b * (LATENT / 32) + c / 32, lane);
    }
  }
  for (int i = (i0 + rstride - nq % rstride) % rstride; i < ns; i += SBATCH * rstride) {
    uint4 g[SBATCH], u[SBATCH];
#pragma unroll
    for (int k = 0; k < SBATCH; ++k) {
      const int ik = min(i + k * rstride, ns - 1);
      const int b = ik / (SHARED / 256), c = ik % (SHARED / 256) * 256 + lane * 8;
      const bf16_t* gate = gu + (long long)b * ldgu + c;
      g[k] = *reinterpret_cast<const uint4*>(gate);
      u[k] = *reinterpret_cast<const uint4*>(gate + SHARED);
    }
#pragma unroll
    for (int k = 0; k < SBATCH; ++k) {
      const int ik = i + k * rstride;
      if (ik >= ns) break;
      const int b = ik / (SHARED / 256), c = ik % (SHARED / 256) * 256 + lane * 8;
      situ8(g[k], u[k], act + (long long)b * SHARED + c);
    }
  }

  // grid barrier
  __syncthreads();
  if (t == 0) {
    __threadfence();
    if (atomicAdd(sync, 1u) == gridDim.x - 1) {
      sync[0] = 0;
      st_release(sync + 1, gen + 1);
    } else {
      while (ld_acquire(sync + 1) == gen) __nanosleep(64);
    }
    __threadfence();
  }
  __syncthreads();

  // the experts' CTA counts, exclusive-scanned: thread t holds experts 2t, 2t + 1
  int c0 = 0, c1 = 0;
  if (2 * t < EXPERTS) {
    cnt[2 * t] = __ldcg(cur.tot + 2 * t);
    cnt[2 * t + 1] = __ldcg(cur.tot + 2 * t + 1);
    c0 = (cnt[2 * t] + tile - 1) / tile;
    c1 = (cnt[2 * t + 1] + tile - 1) / tile;
  }
  int incl = c0 + c1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  int base = 0, total = 0;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) {
    const int s = wsum[w];
    base += w < warp ? s : 0;
    total += s;
  }
  if (2 * t < EXPERTS) {
    off[2 * t] = base + incl - c0 - c1;
    off[2 * t + 1] = base + incl - c1;
  }
  __syncthreads();
  if (gt == 0) {
    num_non_exiting[0] = total;
    total_padded[0] = total * tile;
  }
  for (int e = gt; e < EXPERTS; e += nthreads) {
    const int n = (cnt[e] + tile - 1) / tile, o = off[e];
    for (int c = 0; c < n; ++c) {
      cta_batch[o + c] = e;
      cta_limit[o + c] = min((o + c + 1) * tile, o * tile + cnt[e]);
    }
  }
  // row b's place among its expert's rows: the picks of the chunks before b's, then of b's chunk's words
  // before b's, then of b's word below b
  for (int b = r0; b < B; b += rstride) {
    const int mine = lane < TOPK ? idx[b * TOPK + lane] : 0;
    const int bc = b / CHUNK, bw = b / 32;
    int place = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int e = __shfl_sync(0xffffffffu, mine, k);
      int v = lane < bc ? __ldcg(cur.chunk + e * CHUNKS + lane) : 0;
      const int w = bc * (CHUNK / 32) + lane;
      if (w <= bw) {
        const u32 word = __ldcg(cur.bits + e * WORDS + w);
        v += __popc(w < bw ? word : word & ((1u << (b & 31)) - 1));
      }
      const int r = __reduce_add_sync(0xffffffffu, v);
      if (lane == k) place = r;
    }
    if (lane < TOPK) {
      const int p = off[mine] * tile + place;
      exp2perm[b * TOPK + lane] = p;
      route_map[p] = b;
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3g_moe_route_init(u32* __restrict__ sync) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2 + 2 * TABLE; i += gridDim.x * blockDim.x) sync[i] = 0;
}
