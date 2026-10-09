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
//   nvcc -cubin -arch=sm_103a source/k3_dcp.cu
#include <cuda_bf16.h>
#include <math_constants.h>

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
