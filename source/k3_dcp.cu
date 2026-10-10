// K3 MLA decode under decode context parallelism (DCP): a sequence's KV is
// dealt over the NRANKS members of the tp group position by position
// (position p on member p % NRANKS), every member runs the MLA decode over its
// own positions for every row and all HEADS heads, and these three kernels
// turn the members' partials into this member's HL = HEADS / NRANKS heads of
// the whole attention. Between pack and combine sits one all-to-all of equal
// chunks (NCCL's, `count` = chunk elements per peer), so a chunk's layout is
// the whole contract:
//
//   chunk(R) = R * HL * (LAT + 2) bf16 elements, chunk q of a buffer at
//   q * chunk(R): o [R][HL][LAT] bf16, then lse [R][HL] f32.
//   Head h of a partial travels to member h / HL as its local head h % HL.
//
// lse is the natural-log log-sum-exp the FlashInfer DSL decode returns
// (its user-facing lse, already divided by log2 e), over the member's
// positions only.
//
//   extern "C" __global__ void kern_k3_dcp_fixup(
//       bf16* o_lat,          // [R, HEADS, LAT]   in place
//       float* lse,           // [R, HEADS]        in place
//       const int* seq_lens,  // [R]  this member's positions of the row
//       int R);
//   grid (R, 1, 1)   block 256
//   A row with seq_lens[b] == 0 (the member holds none of its positions yet,
//   or a padding row) gets o = 0 and lse = -inf; other rows are untouched.
//   The decode leaves lse = +inf and an undefined o there.
//
//   extern "C" __global__ void kern_k3_dcp_pack(
//       const bf16* o_lat,    // [R, HEADS, LAT]
//       const float* lse,     // [R, HEADS]
//       bf16* send,           // [NRANKS, chunk(R)]
//       int R);
//   grid (R, HEADS / 4, 1)   block 256: four heads per block, one 16-byte
//   vector of o per thread.
//
//   extern "C" __global__ void kern_k3_dcp_combine(
//       const bf16* recv,     // [NRANKS, chunk(R)]  chunk q from member q
//       bf16* o,              // [R, HL, LAT]
//       int R);
//   grid (R, HL / 4, 1)   block 256: four local heads per block, one 16-byte
//   vector of o per thread.
//   Per (b, j), over the NRANKS partials q (all f32):
//     l_q = lse_q, with NaN and +inf read as -inf
//     m   = max_q l_q;  if m == -inf the row has no position anywhere: o = 0
//     w_q = exp(l_q - m)
//     o[b, j] = bf16(sum_q w_q * f32(o_q) / sum_q w_q), a term with w_q == 0
//               never reading o_q, so whatever an empty partial left there
//               (NaN included) cannot reach the sum.
//   Summation order is q = 0 .. NRANKS - 1, so the result is the same bytes on
//   every run.
//
// kern_k3_dcp_exchange is the DSL decode's split reduction and the four
// steps above in one launch: fixup, pack, the all-to-all and combine, with
// the group's peer-mapped Lamport stages for the wire instead of NCCL
// (TensorRT-LLM's one-shot protocol, as peer_allreduce_bf16.cu runs it).
//
//   extern "C" __global__ void kern_k3_dcp_exchange(
//       const float* acc_o,                    // [R][M_TILE][split_max][LAT]  the splits' normalized o
//       const float* acc_lse,                  // [R][M_TILE][split_max]       their log2-sum-exp
//       const int* seq_lens, const int* bsk,   // [R]  positions held, splits asked for
//       int split_max,
//       bf16* o,                               // [R, HL, LAT]
//       uint8_t* lamport,                      // 3 stages of `stage_bytes`
//       const unsigned long long* peers,       // [NRANKS] every member's `lamport`
//       int* state, int* err,                  // [8] zeroed carry; sticky 1 + the late member
//       int rank, int R, long long stage_bytes, long long timeout_ns);
//   grid: any grid whose blocks are all resident at once   block 256
//
//   The splits merge as the DSL's reduction kernel does: row b ran
//   S = ceil(t / ceil(t / bsk[b])) splits of its t = ceil(seq_lens[b] / 128)
//   tiles; g = m + log2(sum_s exp2(l_s - m)) over them (m their max, 0 if
//   -inf), o = bf16(sum_s acc_o[s] * exp2(l_s - g)) and lse = g / log2(e),
//   ex2 / lg2 approximate.
//
//   A stage holds NRANKS slots, slot q member q's partials of this member's
//   heads: R * HL records of REC 16-byte vectors, a head's o (LANES vectors,
//   -0.0 sent as +0.0: a bf16 -0.0 in a 16-byte vector means "not arrived")
//   and then its lse, one byte per 32-bit word (low half 0x01XX, never the
//   poison). The empty partial (seq_lens[b] == 0, or an lse of NaN / +inf)
//   travels as o = 0, lse = -inf, so the receiver merges exactly as combine
//   does. Every block sends, re-poisons its share of the stage the previous
//   call used, then merges; the stages rotate through state[2] as the
//   all-reduce's do.
//
//   nvcc -cubin -arch=sm_103a source/k3_dcp.cu
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

#ifndef HEADS
#define HEADS 96
#endif
#ifndef NRANKS
#define NRANKS 8
#endif
#define LAT 512
#define HL (HEADS / NRANKS)
#define VEC 8                 // bf16 per 16-byte vector
#define LANES (LAT / VEC)     // 64 threads cover one head's row
#define HPB (256 / LANES)     // heads per block

static_assert(HEADS % NRANKS == 0, "every member takes the same number of heads");
static_assert(HEADS % HPB == 0 && HL % HPB == 0, "a block's four heads never straddle two members");

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ long long chunk(int R) { return (long long)R * HL * (LAT + 2); }

__device__ __forceinline__ float* chunk_lse(bf16* buf, int q, int R) {
  return reinterpret_cast<float*>(buf + q * chunk(R) + (long long)R * HL * LAT);
}

extern "C" __global__ void __launch_bounds__(256) kern_k3_dcp_fixup(bf16* __restrict__ o_lat, float* __restrict__ lse,
                                                                   const int* __restrict__ seq_lens, int R) {
  const int b = blockIdx.x;
  if (seq_lens[b] != 0) return;
  uint4* o = reinterpret_cast<uint4*>(o_lat + (long long)b * HEADS * LAT);
  for (int i = threadIdx.x; i < HEADS * LANES; i += blockDim.x) o[i] = make_uint4(0, 0, 0, 0);
  for (int h = threadIdx.x; h < HEADS; h += blockDim.x) lse[(long long)b * HEADS + h] = -CUDART_INF_F;
}

extern "C" __global__ void __launch_bounds__(256) kern_k3_dcp_pack(const bf16* __restrict__ o_lat,
                                                                  const float* __restrict__ lse,
                                                                  bf16* __restrict__ send, int R) {
  const int b = blockIdx.x, h = blockIdx.y * HPB + threadIdx.x / LANES, v = threadIdx.x % LANES;
  const int q = h / HL, j = h % HL;
  const uint4 x = reinterpret_cast<const uint4*>(o_lat + ((long long)b * HEADS + h) * LAT)[v];
  reinterpret_cast<uint4*>(send + q * chunk(R) + ((long long)b * HL + j) * LAT)[v] = x;
  if (v == 0) chunk_lse(send, q, R)[(long long)b * HL + j] = lse[(long long)b * HEADS + h];
}

extern "C" __global__ void __launch_bounds__(256) kern_k3_dcp_combine(const bf16* __restrict__ recv,
                                                                     bf16* __restrict__ o, int R) {
  const int b = blockIdx.x, j = blockIdx.y * HPB + threadIdx.x / LANES, v = threadIdx.x % LANES;
  float l[NRANKS];
  float m = -CUDART_INF_F;
#pragma unroll
  for (int q = 0; q < NRANKS; ++q) {
    const float x = chunk_lse(const_cast<bf16*>(recv), q, R)[(long long)b * HL + j];
    l[q] = (x != x || x == CUDART_INF_F) ? -CUDART_INF_F : x;
    m = fmaxf(m, l[q]);
  }
  float acc[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float den = 0.f;
  if (m != -CUDART_INF_F) {
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      const float w = expf(l[q] - m);
      if (w == 0.f) continue;
      den += w;
      const uint4 x = reinterpret_cast<const uint4*>(recv + q * chunk(R) + ((long long)b * HL + j) * LAT)[v];
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        const float2 f = __bfloat1622float2(x2[k]);
        acc[2 * k] = fmaf(w, f.x, acc[2 * k]);
        acc[2 * k + 1] = fmaf(w, f.y, acc[2 * k + 1]);
      }
    }
  }
  const float r = den > 0.f ? 1.0f / den : 0.f;
  uint4 out;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int k = 0; k < VEC / 2; ++k) o2[k] = __floats2bfloat162_rn(acc[2 * k] * r, acc[2 * k + 1] * r);
  reinterpret_cast<uint4*>(o + ((long long)b * HL + j) * LAT)[v] = out;
}

#define REC (LANES + 1)
#define M_TILE 128
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float lg2(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
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

__device__ __forceinline__ uint4 lse_word(float l) {
  const uint32_t u = __float_as_uint(l);
  return make_uint4((u & 0xffu) | 0x100u, ((u >> 8) & 0xffu) | 0x100u, ((u >> 16) & 0xffu) | 0x100u, (u >> 24) | 0x100u);
}

__device__ __forceinline__ float word_lse(uint4 v) {
  return __uint_as_float((v.x & 0xffu) | (v.y & 0xffu) << 8 | (v.z & 0xffu) << 16 | (v.w & 0xffu) << 24);
}

// Spins until the vector at p has arrived; `fail` takes 1 + `from` past the deadline.
__device__ __forceinline__ uint4 await(const uint4* p, int from, long long timeout_ns, int& fail) {
  unsigned long long t0 = 0;
  while (true) {
    const uint4 v = ld_volatile(p);
    if (arrived(v) || fail) return v;
    const unsigned long long now = gtimer();
    if (t0 == 0) {
      t0 = now;
    } else if ((long long)(now - t0) > timeout_ns) {
      fail = 1 + from;
      return v;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 1) kern_k3_dcp_exchange(
    const float* __restrict__ acc_o, const float* __restrict__ acc_lse, const int* __restrict__ seq_lens,
    const int* __restrict__ bsk, int split_max, bf16* __restrict__ o, uint8_t* lamport, const unsigned long long* __restrict__ peers, int* state, int* err,
    int rank, int R, long long stage_bytes, long long timeout_ns) {
  const int flag = state[2];
  long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
  const long long clear_count = *clear_ptr;
  const long long slot = (long long)R * HL * REC;
  const long long stage = (flag % 3) * stage_bytes;
  const uint4* mine = reinterpret_cast<const uint4*>(lamport + stage);
  uint4* clear_buf = reinterpret_cast<uint4*>(lamport + ((flag + 2) % 3) * stage_bytes);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(state, 1);

  const int lane = threadIdx.x % LANES, sub = threadIdx.x / LANES;
  for (int it = blockIdx.x; it < R * (HEADS / HPB); it += gridDim.x) {
    const int b = it / (HEADS / HPB), h = it % (HEADS / HPB) * HPB + sub;
    const int n = seq_lens[b];
    float l = -CUDART_INF_F;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (n != 0) {
      const int tiles = (n + M_TILE - 1) / M_TILE, per = (tiles + bsk[b] - 1) / bsk[b], splits = (tiles + per - 1) / per;
      const long long rec = (long long)b * M_TILE + h;
      const float* ls = acc_lse + rec * split_max;
      float m = -CUDART_INF_F;
      for (int i = 0; i < splits; ++i) m = fmaxf(m, ls[i]);
      if (m == -CUDART_INF_F) m = 0.f;
      float sum = 0.f;
      for (int i = 0; i < splits; ++i) sum += ex2(ls[i] - m);
      l = sum != 0.f ? m + lg2(sum) : CUDART_INF_F;
      float a[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const float4* src = reinterpret_cast<const float4*>(acc_o + rec * split_max * LAT) + 2 * lane;
      for (int i = 0; i < splits; ++i, src += LAT / 4) {
        const float w = ex2(ls[i] - l);
        const float4 x = src[0], y = src[1];
        a[0] = fmaf(x.x, w, a[0]), a[1] = fmaf(x.y, w, a[1]), a[2] = fmaf(x.z, w, a[2]), a[3] = fmaf(x.w, w, a[3]);
        a[4] = fmaf(y.x, w, a[4]), a[5] = fmaf(y.y, w, a[5]), a[6] = fmaf(y.z, w, a[6]), a[7] = fmaf(y.w, w, a[7]);
      }
      __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) v2[k] = __floats2bfloat162_rn(a[2 * k], a[2 * k + 1]);
      v = make_uint4(unpoison(v.x), unpoison(v.y), unpoison(v.z), unpoison(v.w));
      l *= 1.0f / LOG2E;
    }
    uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[h / HL]) + stage) + rank * slot +
                 ((long long)b * HL + h % HL) * REC;
    dst[lane] = v;
    if (lane == 0) dst[LANES] = lse_word(l != l || l == CUDART_INF_F ? -CUDART_INF_F : l);
  }
  const uint4 poison = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < clear_count; i += (long long)gridDim.x * blockDim.x)
    clear_buf[i] = poison;

  int fail = 0;
  for (int it = blockIdx.x; it < R * (HL / HPB); it += gridDim.x) {
    const int b = it / (HL / HPB), j = it % (HL / HPB) * HPB + sub;
    const uint4* rec = mine + ((long long)b * HL + j) * REC;
    float l[NRANKS];
    float m = -CUDART_INF_F;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      l[q] = word_lse(await(rec + q * slot + LANES, q, timeout_ns, fail));
      m = fmaxf(m, l[q]);
    }
    float acc[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float den = 0.f;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      const uint4 x = await(rec + q * slot + lane, q, timeout_ns, fail);
      const float w = m == -CUDART_INF_F ? 0.f : expf(l[q] - m);
      if (w == 0.f) continue;
      den += w;
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        const float2 f = __bfloat1622float2(x2[k]);
        acc[2 * k] = fmaf(w, f.x, acc[2 * k]);
        acc[2 * k + 1] = fmaf(w, f.y, acc[2 * k + 1]);
      }
    }
    const float r = den > 0.f ? 1.0f / den : 0.f;
    uint4 out;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) o2[k] = __floats2bfloat162_rn(acc[2 * k] * r, acc[2 * k + 1] * r);
    reinterpret_cast<uint4*>(o + ((long long)b * HL + j) * LAT)[lane] = out;
  }
  if (fail) atomicMax(err, fail);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (*reinterpret_cast<volatile int*>(state) != (int)gridDim.x) {
    }
    state[2] = (flag + 1) % 3;
    *clear_ptr = NRANKS * slot;
    state[0] = 0;
  }
}
