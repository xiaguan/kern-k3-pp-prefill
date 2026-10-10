// K3 MLA decode, absorb (kern's tools/kernels-src/k3_mla_absorb.cu; GROUPS
// may be set at build): the per-head query lands as bf16 and is folded
// through W_UK into the latent space, so attention runs against the cached
// latent rows directly (docs/k3-kernel-abi.md K5a).
//
//   extern "C" __global__ void kern_k3_mla_absorb(
//       const f32*  q_partial,   // [B, HEADS*192]  f32 partial, nope 128 | rope 64
//       const bf16* w_kv_b,      // [HEADS*256, 512]  W_UK = rows h*256+0..128
//       bf16*       q_abs,       // [B, HEADS, 576]  latent 512 | rope 64
//       int B);
//
//   grid  (ceil(B/(8*GROUPS)), HEADS, 8)   block (128, 1, 1)   smem 0 dynamic
//
// Block (bt, h, cs) produces columns cs*64 .. +64 of q_abs for rows
// 8*GROUPS*bt .. +8*GROUPS of head h, eight rows at a time. A decode batch of
// tens of rows wants GROUPS=1: the groups run one after another, so at 48
// rows the default makes 2 blocks per head that walk 4 row groups each.
// W_UK is 12 MB the whole batch shares, so the kernel is a bandwidth
// problem: 768 blocks at B=1, each thread owning
// 8 d-rows x 8 columns with all eight 16-byte loads in flight, the whole
// matrix is requested at once and stays in registers across the row groups;
// the 16 d-slices of a block meet in shared memory. Staging loads are
// unconditional (rows past B re-read the last row and are zeroed) so they
// too are all in flight at once.
//
//   q_h     = bf16(q_partial[b, h*192 .. +192])
//   q_abs   = [ bf16(sum_d q_h[d]*W_UK_h[d,j]) for j<512 | q_h[128..192] ]   f32 acc
//
//   nvcc -cubin -arch=sm_103a -O3 [-DGROUPS=1] source/k3_mla_absorb.cu
#include <cuda_bf16.h>
#define HEADS 96
#define NOPE 128
#define ROPE 64
#define LAT 512
#define ROW 576
#define QW 192
#define RB 8          // rows per group
#ifndef GROUPS
#define GROUPS 4      // row groups per block
#endif
#define CS 64         // columns per block
#define KS 16         // d-slices per block (8 rows of W each)
#define OCT (CS / 8)  // 8-column octets per slice

// -DQBF16: the query arrives landed (bf16 [B, HEADS*192], a prefill's q_b GEMM output): the same qh.
// The prefill launches it for B <= 512 rows with the FMHA's tables (k3_fmha_plan.cuh, ns words a
// row): past short_max rows there is no absorbed attention, and the blocks instead copy a split
// sequence's q rows (Lq from row q0) to the FMHA pieces' rows B + Lq - 1 - i (reversed) and B + Lq + i,
// block (bt, h, cs) rows bt * 32 + cs * 4 .. +4 of head h.
#ifdef QBF16
typedef __nv_bfloat16 q_t;
__device__ __forceinline__ float qf(q_t x) { return __bfloat162float(x); }
#else
typedef float q_t;
__device__ __forceinline__ float qf(q_t x) { return x; }
#endif

extern "C" __global__ void __launch_bounds__(128) kern_k3_mla_absorb(const q_t* __restrict__ q_partial,
                                                                    const __nv_bfloat16* __restrict__ w_kv_b,
                                                                    __nv_bfloat16* __restrict__ q_abs, int B
#ifdef QBF16
                                                                    , const int* __restrict__ lens, int ns, int short_max
#endif
) {
#ifdef QBF16
  if (B > short_max) {
    const int* __restrict__ rec = lens + 3 * ns;
    const int lq = rec[3], i = blockIdx.x * 32 + blockIdx.z * 4 + threadIdx.x / (QW / 8), e = threadIdx.x % (QW / 8);
    if (threadIdx.x >= 4 * (QW / 8) || i >= lq) return;
    q_t* __restrict__ q = const_cast<q_t*>(q_partial);
    const size_t col = (size_t)blockIdx.y * QW + e * 8;
    const uint4 v = *reinterpret_cast<const uint4*>(q + (size_t)(rec[2] + i) * (HEADS * QW) + col);
    *reinterpret_cast<uint4*>(q + (size_t)(B + lq - 1 - i) * (HEADS * QW) + col) = v;
    *reinterpret_cast<uint4*>(q + (size_t)(B + lq + i) * (HEADS * QW) + col) = v;
    return;
  }
#endif
  __shared__ __nv_bfloat16 qh[RB][QW];
  __shared__ float red[KS][RB][CS];
  const int h = blockIdx.y, c0 = blockIdx.z * CS, t = threadIdx.x;
  const int ks = t / OCT, oc = t - ks * OCT;
  const int d0 = ks * (NOPE / KS), c = c0 + oc * 8;
  const __nv_bfloat16* __restrict__ w = w_kv_b + (size_t)h * 256 * LAT + (size_t)d0 * LAT + c;
  uint4 wv[NOPE / KS];
#pragma unroll
  for (int d = 0; d < NOPE / KS; ++d) wv[d] = *reinterpret_cast<const uint4*>(w + (size_t)d * LAT);
  for (int g = 0; g < GROUPS; ++g) {
    const int b0 = (blockIdx.x * GROUPS + g) * RB;
    const int nr = min(RB, B - b0);
    if (nr <= 0) break;
    // RB*QW / 128 = 12 elements per thread, all loads in flight before any store
    float qv[RB * QW / 128];
#pragma unroll
    for (int k = 0; k < RB * QW / 128; ++k) {
      const int i = t + k * 128, r = i / QW, e = i - r * QW;
      qv[k] = qf(q_partial[(size_t)min(b0 + r, B - 1) * (HEADS * QW) + (size_t)h * QW + e]);
    }
    if (g) __syncthreads();  // the previous group is done reading qh / red
#pragma unroll
    for (int k = 0; k < RB * QW / 128; ++k) {
      const int i = t + k * 128, r = i / QW, e = i - r * QW;
      qh[r][e] = __float2bfloat16_rn(r < nr ? qv[k] : 0.f);
    }
    __syncthreads();
    float acc[RB][8];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[r][j] = 0.f;
#pragma unroll
    for (int d = 0; d < NOPE / KS; ++d) {
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[d]);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(w2[j]);
        wf[2 * j] = f.x;
        wf[2 * j + 1] = f.y;
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const float q = __bfloat162float(qh[r][d0 + d]);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][j] += q * wf[j];
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j) red[ks][r][oc * 8 + j] = acc[r][j];
    __syncthreads();
    // (row, column pair) per thread: RB*CS/2 = 256 pairs over 128 threads
    for (int i = t; i < RB * (CS / 2); i += 128) {
      const int r = i / (CS / 2), j2 = i - r * (CS / 2);
      if (r >= nr) continue;
      float x = 0.f, y = 0.f;
#pragma unroll
      for (int k = 0; k < KS; ++k) {
        x += red[k][r][2 * j2];
        y += red[k][r][2 * j2 + 1];
      }
      __nv_bfloat16* out = q_abs + ((size_t)(b0 + r) * HEADS + h) * ROW + c0;
      reinterpret_cast<__nv_bfloat162*>(out)[j2] = __floats2bfloat162_rn(x, y);
    }
    if (blockIdx.z == 0) {
      for (int i = t; i < nr * ROPE; i += 128) {
        const int r = i / ROPE, k = i - r * ROPE;
        q_abs[((size_t)(b0 + r) * HEADS + h) * ROW + LAT + k] = qh[r][NOPE + k];
      }
    }
  }
}
