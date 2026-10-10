// The glue of a packed prefill chunk's MLA layer around TensorRT-LLM's
// context FMHA, two launches where there were three:
//
//   kern_k3g_mla_prep_gather(partial, gamma_q_a, gamma_kv_a, slot_mapping, slab, page_stride, q_norm,
//                            block_table, max_pages, lens, ns, nseq, latent_g, n, B)
//     blocks [0, B): k3_mla_prep.cu's head for chunk row b (q_norm, the latent row kv_norm | rope
//       appended to the slab), the latent row also written to its place in latent_g;
//     blocks [B, B + ceil(n / 7)): rows [7 g, 7 g + 7) of latent_g, the sequences' cached latent
//       rows back to back (sequence j's at [cum_kv[j], cum_kv[j] + seq_lens_kv[j])), every other row
//       of the first n zero; the rows this chunk appends are the head blocks' own.
//     partial f32 [B, MLA_FUSED] the fused projection (q_a | kv_a | rope | gate);
//     slab, page_stride: the layer's latent pages; lens: kern_k3_fmha_lens_varlen's tables
//     (seq_lens_kv | cum_q | cum_kv, ns words each) of the call's nseq sequences.
//   grid (B + ceil(n / 7), 1, 1)   block (512, 1, 1)
//
//   kern_k3g_mla_gate(out, o, partial, n, B)
//     sigmoid_mul.cu's kern_sigmoid_mul_bf16 on the gate columns of the fused projection, landed to
//     bf16 here (k3_mla_prep.cu landed them into a buffer of their own):
//     out = bf16(o * bf16(sigmoid(bf16(partial[b, HEADC + j])))), o / out [B, n].
//   grid (B, ceil(n / 2048), 1)   block (256, 1, 1)
//
// Every value is the three kernels' bit for bit: the same landings, sums and roundings.
#include <cuda_bf16.h>

#define Q_LORA 1536
#define KV_LORA 512
#define ROPE 64
#define KV_A 576
#ifndef MLA_FUSED
#define MLA_FUSED 14400
#endif
#define HEADC (Q_LORA + KV_LORA + ROPE)  // 2112 head columns
#define EPSV 1e-5f
#define PAGE 64
#define QU (Q_LORA / 4)  // 384 float4 units of q
#define NT 512
#define QW (QU / 32)  // 12 warps of q
#define LANES (KV_A * 2 / 16)  // 72 sixteen-byte pieces a latent row
#define GROWS (NT / LANES)      // 7 latent rows a gather block

typedef __nv_bfloat16 bf16_t;

__device__ __forceinline__ unsigned pack2(float lo, float hi) {
  __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&p);
}
__device__ __forceinline__ unsigned mul2(unsigned a, unsigned g) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a), *reinterpret_cast<const __nv_bfloat162*>(&g));
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ float landf(float x) { return __bfloat162float(__float2bfloat16(x)); }

extern "C" __global__ void __launch_bounds__(NT) kern_k3g_mla_prep_gather(
    const float* __restrict__ partial, const bf16_t* __restrict__ gamma_q_a, const bf16_t* __restrict__ gamma_kv_a,
    const long long* __restrict__ slot_mapping, bf16_t* __restrict__ slab, long long page_stride,
    bf16_t* __restrict__ q_norm, const int* __restrict__ block_table, int max_pages, const int* __restrict__ lens,
    int ns, int nseq, bf16_t* __restrict__ latent_g, int n, int B) {
  const int t = threadIdx.x;
  const int* __restrict__ cum_q = lens + ns;
  const int* __restrict__ cum_kv = lens + 2 * ns;

  if (blockIdx.x >= B) {  // ---- gather: the context's cached latent rows
    const int r = (blockIdx.x - B) * GROWS + t / LANES, lane = t % LANES;
    if (t >= GROWS * LANES || r >= n) return;
    int j = 0;
    while (j + 1 < nseq && cum_kv[j + 1] <= r) ++j;
    const int local = r - cum_kv[j];
    const int cached = lens[j] - (cum_q[j + 1] - cum_q[j]);
    if (local >= cached && local < lens[j]) return;  // this chunk's row: its head block writes it
    uint4 v = make_uint4(0, 0, 0, 0);
    if (local < cached)
      v = *reinterpret_cast<const uint4*>(slab + block_table[j * max_pages + local / PAGE] * page_stride +
                                          (long long)(local % PAGE) * KV_A + lane * 8);
    *reinterpret_cast<uint4*>(latent_g + (long long)r * KV_A + lane * 8) = v;
    return;
  }

  // ---- head: q_norm, kv_norm, rope, the latent row (k3_mla_prep.cu's fast head)
  __shared__ float red[NT / 32];
  const int b = blockIdx.x;
  const float* __restrict__ P = partial + (long long)b * MLA_FUSED;
  const bool isq = (t < QU);
  const int col = isq ? (4 * t) : (Q_LORA + 4 * (t - QU));
  const float4 v = *reinterpret_cast<const float4*>(P + col);
  const float x0 = landf(v.x), x1 = landf(v.y);
  const float x2 = landf(v.z), x3 = landf(v.w);

  float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
  if ((t & 31) == 0) red[t >> 5] = ss;

  const long long slot = slot_mapping[b];
  const bool append = (slot >= 0);
  bf16_t* const row = slab + (slot / PAGE) * page_stride + (slot % PAGE) * KV_A;
  int j = 0;
  while (j + 1 < nseq && cum_q[j + 1] <= b) ++j;
  bf16_t* const grow =
      latent_g + (long long)(cum_kv[j] + lens[j] - (cum_q[j + 1] - cum_q[j]) + (b - cum_q[j])) * KV_A;
  const bf16_t* const gsrc = isq ? (gamma_q_a + col) : (gamma_kv_a + (col - Q_LORA));
  const uint2 g = *reinterpret_cast<const uint2*>(gsrc);
  uint2 rope4 = make_uint2(0u, 0u);
  if (t < ROPE / 4) {
    const float4 rv = *reinterpret_cast<const float4*>(P + Q_LORA + KV_LORA + 4 * t);
    rope4 = make_uint2(pack2(rv.x, rv.y), pack2(rv.z, rv.w));
  }

  __syncthreads();

  float tot = 0.f;
  if (isq) {
#pragma unroll
    for (int w = 0; w < QW; ++w) tot += red[w];
  } else {
#pragma unroll
    for (int w = QW; w < NT / 32; ++w) tot += red[w];
  }
  const float sc = isq ? rsqrtf(tot * (1.f / Q_LORA) + EPSV) : rsqrtf(tot * (1.f / KV_LORA) + EPSV);
  const uint2 o = make_uint2(mul2(pack2(x0 * sc, x1 * sc), g.x), mul2(pack2(x2 * sc, x3 * sc), g.y));
  if (isq) {
    *reinterpret_cast<uint2*>(q_norm + (long long)b * Q_LORA + col) = o;
    if (t < ROPE / 4) {
      *reinterpret_cast<uint2*>(grow + KV_LORA + 4 * t) = rope4;
      if (append) *reinterpret_cast<uint2*>(row + KV_LORA + 4 * t) = rope4;
    }
  } else {
    *reinterpret_cast<uint2*>(grow + (col - Q_LORA)) = o;
    if (append) *reinterpret_cast<uint2*>(row + (col - Q_LORA)) = o;
  }
}

extern "C" __global__ void __launch_bounds__(256) kern_k3g_mla_gate(bf16_t* __restrict__ out, const bf16_t* __restrict__ o,
                                                                    const float* __restrict__ partial, int n, int B) {
  const long long b = blockIdx.x;
  const int j = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
  if (b >= B || j >= n) return;
  const float4* gp = reinterpret_cast<const float4*>(partial + b * MLA_FUSED + HEADC + j);
  const float4 g0 = gp[0], g1 = gp[1];
  const float gx[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
  const uint4 av = *reinterpret_cast<const uint4*>(o + b * n + j);
  const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&av);
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float x0 = landf(gx[2 * i]), x1 = landf(gx[2 * i + 1]);
    const __nv_bfloat162 g = __floats2bfloat162_rn(1.0f / (1.0f + expf(-x0)), 1.0f / (1.0f + expf(-x1)));
    op[i] = __floats2bfloat162_rn(__bfloat162float(ap[i].x) * __bfloat162float(g.x),
                                  __bfloat162float(ap[i].y) * __bfloat162float(g.y));
  }
  *reinterpret_cast<uint4*>(out + b * n + j) = ov;
}
