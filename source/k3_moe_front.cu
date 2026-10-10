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
//     kern_k3_router_topk, by warps 0..7 (below);
//   * meanwhile warps 8..21 land the latent (columns EXPERTS + [0, LATENT)) to
//     bf16 and quantise it to mxfp8 as k3_moe_prefill.cu's kern_k3_moe_quant
//     (four threads per 32-element group), and warps 22..27 run k3_land.cu's
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
#ifndef TOPK_WARPS
#define TOPK_WARPS 8
#endif
#define SLICE (EXPERTS / TOPK_WARPS)
#define PER ((SLICE + 31) / 32)
#define CPER (TOPK_WARPS * TOPK / 32)
#define QUANT_WARP0 TOPK_WARPS
#define QUANT_THREADS (LATENT / 8)
#define SITU_WARP0 (TOPK_WARPS - SITU_THREADS / 32)  // the slices' warps but the merging one, after their slice
#define SITU_THREADS (SH / 4)
static_assert(EXPERTS % TOPK_WARPS == 0 && TOPK_WARPS * TOPK % 32 == 0 && EXPERTS <= THREADS, "slices, candidates");
static_assert(QUANT_THREADS % 32 == 0 && SH % 4 == 0, "whole warps");
static_assert(SITU_WARP0 > 0 && (QUANT_WARP0 * 32 + QUANT_THREADS) <= THREADS, "the side work fits the block");
static_assert(LOCAL <= THREADS && TOPK * 64 <= THREADS, "one expanded id per thread");

typedef unsigned int u32;

__device__ __forceinline__ u32 ord_f32(float f) {
  u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
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

// k3_router_argmax.cu's picks (largest biased score, ties to the smallest expert, sixteen
// rounds) in two levels: warp w < TOPK_WARPS takes its slice's top 16 (lane l holding
// experts w * SLICE + l * PER + j), then warp 0 the top 16 of those candidates (lane l
// holding candidates l * CPER + j). The order is total, so the second level picks the
// first level's union down to the global picks in the global order.
struct Cand {
  u32 key[TOPK_WARPS * TOPK];
  int e[TOPK_WARPS * TOPK];
  float sig[TOPK_WARPS * TOPK];
};

// A round over a lane's N slots: the largest key (REDUX.MAX), the smallest expert holding
// it (REDUX.MIN); the lane holding that expert blanks the slot and returns true.
template <int N>
__device__ __forceinline__ bool pick(u32 (&o)[N], const int (&e)[N], const float (&g)[N], u32& m, int& ew,
                                     float& gw) {
  m = o[0];
#pragma unroll
  for (int j = 1; j < N; ++j) m = max(m, o[j]);
  m = __reduce_max_sync(0xffffffffu, m);
  u32 my = 0xffffffffu;
#pragma unroll
  for (int j = 0; j < N; ++j)
    if (o[j] == m) my = min(my, (u32)e[j]);
  ew = (int)__reduce_min_sync(0xffffffffu, my);
  bool won = false;
#pragma unroll
  for (int j = 0; j < N; ++j)
    if (e[j] == ew && o[j] == m) {
      gw = g[j];
      o[j] = 0u;  // strictly below ord() of any real float
      won = true;
    }
  return won;
}

__device__ __forceinline__ void topk_slice(const float* s_sig, const u32* s_ord, Cand& c, int w, int lane) {
  u32 o[PER];
  int e[PER];
  float g[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int k = lane * PER + j;
    e[j] = k < SLICE ? w * SLICE + k : 0x7fffffff;
    o[j] = k < SLICE ? s_ord[e[j]] : 0u;
    g[j] = k < SLICE ? s_sig[e[j]] : 0.f;
  }
  for (int r = 0; r < TOPK; ++r) {
    u32 m;
    int ew;
    float gw;
    if (pick(o, e, g, m, ew, gw)) {
      c.key[w * TOPK + r] = m;
      c.e[w * TOPK + r] = ew;
      c.sig[w * TOPK + r] = gw;
    }
  }
}

__device__ __forceinline__ void topk_merge(const Cand& c, int* s_e, float* s_w, const float rs, int* __restrict__ idx,
                                           float* __restrict__ wts, int b, int lane) {
  u32 o[CPER];
  int e[CPER];
  float g[CPER];
#pragma unroll
  for (int j = 0; j < CPER; ++j) {
    o[j] = c.key[lane * CPER + j];
    e[j] = c.e[lane * CPER + j];
    g[j] = c.sig[lane * CPER + j];
  }
  for (int r = 0; r < TOPK; ++r) {
    u32 m;
    int ew;
    float gw;
    if (pick(o, e, g, m, ew, gw)) {
      s_e[r] = ew;
      s_w[r] = gw;
    }
  }
  __syncwarp();
  if (lane < TOPK) {
    float sg = s_w[lane];
    float den = sg;
#pragma unroll
    for (int off = TOPK / 2; off > 0; off >>= 1) den += __shfl_xor_sync(0x0000ffffu, den, off);
    wts[b * TOPK + lane] = (sg / (den + 1e-20f)) * rs;
    idx[b * TOPK + lane] = s_e[lane];
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
    for (int w = 0; w < (T * TOPK + 31) / 32; ++w) {
      const int c = hist[w][i];
      hist[w][i] = run;
      run += c;
    }
    count[i] = run;
  }
  __syncthreads();
  if (warp == 0) {  // off = exclusive scan of the experts' CTA counts
    constexpr int EPL = (LOCAL + 31) / 32;
    int c[EPL], sum = 0;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int e = lane * EPL + k;
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
    for (int k = 0; k < EPL; ++k) {
      const int e = lane * EPL + k;
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
  __shared__ Cand s_cand;

  const int b = blockIdx.x, t = threadIdx.x, warp = t / 32;
  const float* row = S + (long long)b * LDS;
  if (t < EXPERTS) {
    float sg = 1.0f / (1.0f + expf(-row[t]));
    s_sig[t] = sg;
    s_ord[t] = ord_f32(sg + bias[t]);
  }
  __syncthreads();

  if (warp < TOPK_WARPS) {
    topk_slice(s_sig, s_ord, s_cand, warp, t % 32);
    asm volatile("bar.sync 1, %0;" ::"n"(TOPK_WARPS * 32));
    if (warp == 0) topk_merge(s_cand, s_e, s_w, __bfloat162float(rs[0]), idx, wts, b, t);
  } else if (warp < QUANT_WARP0 + QUANT_THREADS / 32) {
    quant8(row + EXPERTS, q + (long long)b * LATENT, sf + (long long)b * (LATENT / 32), t - QUANT_WARP0 * 32);
  }
  if (t >= SITU_WARP0 * 32 && t < SITU_WARP0 * 32 + SITU_THREADS) {
    const int k = t - SITU_WARP0 * 32;
    const float4 g = reinterpret_cast<const float4*>(row + EXPERTS + LATENT)[k];
    const float4 u = reinterpret_cast<const float4*>(row + EXPERTS + LATENT + SH)[k];
    __nv_bfloat162 o[2];
    o[0] = __floats2bfloat162_rn(situ_f(g.x, u.x), situ_f(g.y, u.y));
    o[1] = __floats2bfloat162_rn(situ_f(g.z, u.z), situ_f(g.w, u.w));
    reinterpret_cast<uint2*>(act + (long long)b * SH)[k] = *reinterpret_cast<const uint2*>(o);
  }

  if (warp == 0) __threadfence();  // the ids, the one output the last block reads
  __syncthreads();
  if (t == 0) {
    s_last = atomicAdd(done, 1) == T - 1;
    __threadfence();
  }
  __syncthreads();
  if (!s_last) return;
  route(idx, first, T, cta_batch, cta_limit, num_non_exiting, total_padded, route_map, exp2perm);
  if (t == 0) *done = 0;
}
