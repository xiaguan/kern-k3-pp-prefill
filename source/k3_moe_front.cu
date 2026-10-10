// A DCP decode step's MoE front in one launch: everything between the front
// GEMM (router | lat_down | this rank's shared gate | up, one f32 row of LDS
// columns) and the batched GEMMs.
//
//   kern_k3_moe_front(const f32* S, const f32* bias, const bf16* rs, i32* idx, f32* wts,
//       u8* q, u8* sf, bf16* act, i32* done, i32* cta_batch, i32* cta_limit,
//       i32* num_non_exiting, i32* total_padded, i32* route_map, i32* exp2perm, int first, int T)
//   grid (T, 1, 1)   block (1024, 1, 1)   T <= 64
//
// Row b's block:
//   * the router top-k of columns [0, EXPERTS): k3_router_argmax.cu's
//     kern_k3_router_topk, by warp 0;
//   * meanwhile warps 1..14 land the latent (columns EXPERTS + [0, LATENT)) to
//     bf16 and quantise it to mxfp8 as k3_moe_prefill.cu's kern_k3_moe_quant
//     (four threads per 32-element group), and warps 15..20 run k3_land.cu's
//     land_situ on the shared expert's gate | up (the SH columns after it);
// then the last block to finish (`done`, a zeroed carry it re-zeroes) builds
// the routing tables of this rank's experts `first + j * STRIDE` (j < LOCAL)
// over all T rows: what FlashInfer's routing writes for the batched GEMMs'
// TILE-row CTAs, with every expert's rows in expanded-id order (FlashInfer's
// order within an expert is its atomics'; the GEMMs' rows do not depend on
// it), padding rows routing token 0, exp2perm -1 for another rank's expert.
//
//   nvcc -cubin -arch=sm_103a -DEXPERTS=896 -DLDS=6016 -DSH=768 -DSTRIDE=8 k3_moe_front.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef EXPERTS
#define EXPERTS 896
#endif
#ifndef LDS
#define LDS 6016
#endif
#ifndef SH
#define SH 768
#endif
#ifndef STRIDE
#define STRIDE 8
#endif
#define LATENT 3584
#define TOPK 16
#define TILE 16
#define LOCAL (EXPERTS / STRIDE)
#define THREADS 1024
#define WARPS (THREADS / 32)
#define PER_LANE (EXPERTS / 32)
#define QUANT_WARP0 1
#define QUANT_THREADS (LATENT / 8)
#define SITU_WARP0 (QUANT_WARP0 + QUANT_THREADS / 32)
#define SITU_THREADS (SH / 4)
static_assert(EXPERTS % 32 == 0 && EXPERTS <= THREADS, "one expert per thread, PER_LANE per lane");
static_assert(QUANT_THREADS % 32 == 0 && SH % 4 == 0, "whole warps");
static_assert((SITU_WARP0 * 32 + SITU_THREADS) <= THREADS, "the side work fits the block");
static_assert(LOCAL <= THREADS && TOPK * 64 <= THREADS, "one expanded id per thread");

typedef unsigned int u32;

__device__ __forceinline__ u32 ord_f32(float f) {
  u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <int L, int N>
__device__ __forceinline__ u32 tree_max(const u32 (&o)[PER_LANE]) {
  if constexpr (N == 1) {
    return o[L];
  } else {
    constexpr int h = N > 4 ? (N > 8 ? (N > 16 ? 16 : 8) : 4) : (N > 2 ? 2 : 1);
    return max(tree_max<L, h>(o), tree_max<L + h, N - h>(o));
  }
}

__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float situ_f(float pg, float pu) {
  float g = __bfloat162float(__float2bfloat16(pg));
  float u = __bfloat162float(__float2bfloat16(pu));
  float a = 4.0f * tanh_approx(g * 0.25f);
  float s = __frcp_rn(1.0f + __expf(-g));
  float c = 25.0f * tanh_approx(u * 0.04f);
  return (a * s) * c;
}

// k3_router_argmax.cu's picks over s_sig / s_ord by warp 0, lane t holding
// experts t * PER_LANE + j.
__device__ __forceinline__ void topk(const float* s_sig, const u32* s_ord, int* s_e, float* s_w, const float rs,
                                     int* __restrict__ idx, float* __restrict__ wts, int b, int t) {
  u32 o[PER_LANE];
  float g[PER_LANE];
#pragma unroll
  for (int j = 0; j < PER_LANE; ++j) {
    o[j] = s_ord[t * PER_LANE + j];
    g[j] = s_sig[t * PER_LANE + j];
  }
  for (int r = 0; r < TOPK; ++r) {
    u32 m = tree_max<0, PER_LANE>(o);
    m = __reduce_max_sync(0xffffffffu, m);
    u32 my = 0xffffffffu;
#pragma unroll
    for (int j = PER_LANE - 1; j >= 0; --j)
      if (o[j] == m) my = (u32)(t * PER_LANE + j);
    u32 ew = __reduce_min_sync(0xffffffffu, my);
    if (my == ew) {
      int jw = (int)ew - t * PER_LANE;
      float gw = 0.0f;
#pragma unroll
      for (int j = 0; j < PER_LANE; ++j)
        if (j == jw) {
          gw = g[j];
          o[j] = 0u;
        }
      s_e[r] = (int)ew;
      s_w[r] = gw;
    }
  }
  __syncwarp();
  if (t < TOPK) {
    float sg = s_w[t];
    float den = sg;
#pragma unroll
    for (int off = TOPK / 2; off > 0; off >>= 1) den += __shfl_xor_sync(0x0000ffffu, den, off);
    wts[b * TOPK + t] = (sg / (den + 1e-20f)) * rs;
    idx[b * TOPK + t] = s_e[t];
  }
}

// Eight latent columns of a 32-column group (its four threads are adjacent lanes):
// landed to bf16, the group's UE8M0 scale, e4m3 at nearest.
__device__ __forceinline__ void quant8(const float* __restrict__ x, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                       int u) {
  const float4 a = reinterpret_cast<const float4*>(x)[2 * u], c = reinterpret_cast<const float4*>(x)[2 * u + 1];
  const float raw[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
  float v[8];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[j] = __bfloat162float(__float2bfloat16(raw[j]));
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  int e = amax > 0.f ? (int)ceilf(log2f(amax / 448.f)) : 0;
  e = min(max(e, -127), 127);
  const float s = exp2f((float)-e);
  if ((u & 3) == 0) sf[u >> 2] = (uint8_t)(e + 127);
  uint8_t o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = __nv_cvt_float_to_fp8(v[j] * s, __NV_SATFINITE, __NV_E4M3);
  reinterpret_cast<uint2*>(q)[u] = *reinterpret_cast<const uint2*>(o);
}

// The last block's routing tables over T rows' ids, thread i one expanded id.
__device__ __forceinline__ void route(const int* __restrict__ idx, int first, int T, int* __restrict__ cta_batch,
                                      int* __restrict__ cta_limit, int* __restrict__ num_non_exiting,
                                      int* __restrict__ total_padded, int* __restrict__ route_map,
                                      int* __restrict__ exp2perm) {
  __shared__ int hist[WARPS][LOCAL];
  __shared__ int count[LOCAL], off[LOCAL + 1];
  const int i = threadIdx.x, warp = i / 32, lane = i % 32;
  const unsigned below = (1u << lane) - 1;
  for (int k = i; k < WARPS * LOCAL; k += THREADS) (&hist[0][0])[k] = 0;
  __syncthreads();
  int j = -1;
  if (i < T * TOPK) {
    const int d = __ldcg(idx + i) - first;
    if (d >= 0 && d % STRIDE == 0) j = d / STRIDE;
  }
  const unsigned peers = __match_any_sync(0xffffffffu, j);
  if (j >= 0 && (peers & below) == 0) hist[warp][j] = __popc(peers);
  __syncthreads();
  if (i < LOCAL) {
    int run = 0;
    for (int w = 0; w < WARPS; ++w) {
      const int c = hist[w][i];
      hist[w][i] = run;
      run += c;
    }
    count[i] = run;
  }
  __syncthreads();
  if (warp == 0) {  // off = exclusive scan of the experts' CTA counts
    constexpr int PER = (LOCAL + 31) / 32;
    int c[PER], sum = 0;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int e = lane * PER + k;
      c[k] = e < LOCAL ? (count[e] + TILE - 1) / TILE : 0;
      sum += c[k];
    }
    int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += n;
    }
    int run = incl - sum;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int e = lane * PER + k;
      if (e < LOCAL) off[e] = run;
      run += c[k];
    }
    if (lane == 31) {
      off[LOCAL] = incl;
      num_non_exiting[0] = incl;
      total_padded[0] = incl * TILE;
    }
  }
  __syncthreads();
  if (i < LOCAL) {
    for (int c = off[i]; c < off[i + 1]; ++c) {
      cta_batch[c] = i;
      cta_limit[c] = min((c + 1) * TILE, off[i] * TILE + count[i]);
    }
    for (int r = off[i] * TILE + count[i]; r < off[i + 1] * TILE; ++r) route_map[r] = 0;
  }
  if (i < T * TOPK) {
    int row = -1;
    if (j >= 0) {
      row = off[j] * TILE + hist[warp][j] + __popc(peers & below);
      route_map[row] = i / TOPK;
    }
    exp2perm[i] = row;
  }
}

extern "C" __global__ void __launch_bounds__(THREADS, 1) kern_k3_moe_front(
    const float* __restrict__ S, const float* __restrict__ bias, const __nv_bfloat16* __restrict__ rs,
    int* __restrict__ idx, float* __restrict__ wts, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
    __nv_bfloat16* __restrict__ act, int* __restrict__ done, int* __restrict__ cta_batch,
    int* __restrict__ cta_limit, int* __restrict__ num_non_exiting, int* __restrict__ total_padded,
    int* __restrict__ route_map, int* __restrict__ exp2perm, int first, int T) {
  __shared__ float s_sig[EXPERTS];
  __shared__ u32 s_ord[EXPERTS];
  __shared__ int s_e[TOPK];
  __shared__ float s_w[TOPK];
  __shared__ int s_last;

  const int b = blockIdx.x, t = threadIdx.x, warp = t / 32;
  const float* row = S + (long long)b * LDS;
  if (t < EXPERTS) {
    float sg = 1.0f / (1.0f + expf(-row[t]));
    s_sig[t] = sg;
    s_ord[t] = ord_f32(sg + bias[t]);
  }
  __syncthreads();

  if (warp == 0) {
    topk(s_sig, s_ord, s_e, s_w, __bfloat162float(rs[0]), idx, wts, b, t);
  } else if (warp < SITU_WARP0) {
    quant8(row + EXPERTS, q + (long long)b * LATENT, sf + (long long)b * (LATENT / 32), t - QUANT_WARP0 * 32);
  } else if (t < SITU_WARP0 * 32 + SITU_THREADS) {
    const int k = t - SITU_WARP0 * 32;
    const float4 g = reinterpret_cast<const float4*>(row + EXPERTS + LATENT)[k];
    const float4 u = reinterpret_cast<const float4*>(row + EXPERTS + LATENT + SH)[k];
    __nv_bfloat162 o[2];
    o[0] = __floats2bfloat162_rn(situ_f(g.x, u.x), situ_f(g.y, u.y));
    o[1] = __floats2bfloat162_rn(situ_f(g.z, u.z), situ_f(g.w, u.w));
    reinterpret_cast<uint2*>(act + (long long)b * SH)[k] = *reinterpret_cast<const uint2*>(o);
  }

  __threadfence();
  __syncthreads();
  if (t == 0) {
    s_last = atomicAdd(done, 1) == T - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  route(idx, first, T, cta_batch, cta_limit, num_non_exiting, total_padded, route_map, exp2perm);
  if (t == 0) *done = 0;
}
