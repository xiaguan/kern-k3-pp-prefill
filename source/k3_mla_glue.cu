// The glue of a packed prefill chunk's MLA layer around TensorRT-LLM's
// context FMHA, two launches where there were three:
//
//   kern_k3g_mla_prep_gather(partial, gamma_q_a, gamma_kv_a, slot_mapping, slab, page_stride, q_norm,
//                            block_table, max_pages, lens, ns, nseq, latent_g, n, B)
//     blocks [0, B): k3_mla_prep.cu's head for chunk row b (q_norm, the latent row kv_norm | rope
//       appended to the slab), the latent row also written to its place in latent_g;
//     blocks [B, B + ceil(n / 28)): rows [28 g, 28 g + 28) of latent_g, the sequences' cached latent
//       rows back to back (sequence j's at [cum_kv[j], cum_kv[j] + seq_lens_kv[j])), every other row
//       of the first n zero; the rows this chunk appends are the head blocks' own.
//     partial f32 [B, MLA_FUSED] the fused projection (q_a | kv_a | rope | gate);
//     slab, page_stride: the layer's latent pages; lens: kern_k3_fmha_lens_varlen's tables
//     (seq_lens_kv | cum_q | cum_kv, ns words each) of the call's nseq sequences.
//   grid (B + ceil(n / 28), 1, 1)   block (512, 1, 1)
//
//   kern_k3g_mla_prep_rows(partial, gamma_q_a, gamma_kv_a, slot_mapping, slab, page_stride, q_norm,
//                          block_table, max_pages, lens, ns, nseq, row_table, row_lens, bsk, split_max, B)
//     a short chunk's form, for the absorbed attention (k3_mla_absorb → FlashInfer's MLA decode →
//     k3_mla_vup_gate) instead of the expansion + FMHA: blocks [0, B) the head without latent_g;
//     blocks [B, 2B) row b's page table row (its sequence's) and causal length (the cached rows, the
//     chunk's rows up to b); block 2B k3_mla_split_plan.cu's KV split plan over those lengths.
//   grid (2 B + 1, 1, 1)   block (512, 1, 1)
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
#define GROWS (NT / LANES)      // 7 latent rows a gather block pass
#ifndef GITER
#define GITER 4                 // passes: 28 rows a gather block
#endif

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

// k3_mla_prep.cu's fast head for chunk row b: q_norm, kv_norm | rope appended to the slab, and the
// latent row also written to `grow` unless null. The whole 512-thread block.
__device__ __forceinline__ void prep_head(const float* __restrict__ partial, const bf16_t* __restrict__ gamma_q_a,
                                          const bf16_t* __restrict__ gamma_kv_a, const long long* __restrict__ slot_mapping,
                                          bf16_t* __restrict__ slab, long long page_stride, bf16_t* __restrict__ q_norm,
                                          bf16_t* __restrict__ grow, int b, int t) {
  __shared__ float red[NT / 32];
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
      if (grow) *reinterpret_cast<uint2*>(grow + KV_LORA + 4 * t) = rope4;
      if (append) *reinterpret_cast<uint2*>(row + KV_LORA + 4 * t) = rope4;
    }
  } else {
    if (grow) *reinterpret_cast<uint2*>(grow + (col - Q_LORA)) = o;
    if (append) *reinterpret_cast<uint2*>(row + (col - Q_LORA)) = o;
  }
}

#define TILE_KV 128  // the decode attention's KV tile (k3_mla_split_plan.cu)

extern "C" __global__ void __launch_bounds__(NT) kern_k3g_mla_prep_gather(
    const float* __restrict__ partial, const bf16_t* __restrict__ gamma_q_a, const bf16_t* __restrict__ gamma_kv_a,
    const long long* __restrict__ slot_mapping, bf16_t* __restrict__ slab, long long page_stride,
    bf16_t* __restrict__ q_norm, const int* __restrict__ block_table, int max_pages, const int* __restrict__ lens,
    int ns, int nseq, bf16_t* __restrict__ latent_g, int n, int* __restrict__ row_table, int* __restrict__ row_lens,
    int* __restrict__ bsk, int split_max, int short_max, int B) {
  const int t = threadIdx.x;
  const int* __restrict__ cum_q = lens + ns;
  const int* __restrict__ cum_kv = lens + 2 * ns;
  const bool absorbed = B <= short_max;
  const auto seq_of = [&](int b) {
    int j = 0;
    while (j + 1 < nseq && cum_q[j + 1] <= b) ++j;
    return j;
  };
  // row b of sequence j: its place among the sequence's rows (the cached ones first), its causal length
  const auto pos_of = [&](int b, int j) { return lens[j] - (cum_q[j + 1] - cum_q[j]) + (b - cum_q[j]); };

  if (blockIdx.x < B) {  // ---- head: q_norm, kv_norm, rope, the latent row (also to latent_g for the expansion)
    const int b = blockIdx.x, j = seq_of(b);
    prep_head(partial, gamma_q_a, gamma_kv_a, slot_mapping, slab, page_stride, q_norm,
              absorbed ? nullptr : latent_g + (long long)(cum_kv[j] + pos_of(b, j)) * KV_A, b, t);
    return;
  }
  if (blockIdx.x < 2 * B + 1) {  // ---- a short chunk's rows for the absorbed attention
    if (!absorbed) return;
    if (blockIdx.x < 2 * B) {  // row b's page table (its sequence's pages) and causal length
      const int b = blockIdx.x - B, j = seq_of(b), len = pos_of(b, j) + 1;
      for (int p = t; p < (len + PAGE - 1) / PAGE; p += NT) row_table[b * max_pages + p] = block_table[j * max_pages + p];
      if (t == 0) row_lens[b] = len;
      return;
    }
    // k3_mla_split_plan.cu's KV split plan over those lengths
    __shared__ long long wsum[NT / 32];
    const int warp = t >> 5, lane = t & 31;
    long long mine = 0;
    for (int b = t; b < B; b += NT) mine += (pos_of(b, seq_of(b)) + TILE_KV) / TILE_KV;
#pragma unroll
    for (int o = 16; o; o >>= 1) mine += __shfl_xor_sync(0xffffffffu, mine, o);
    if (lane == 0) wsum[warp] = mine;
    __syncthreads();
    long long total = 0;
#pragma unroll
    for (int w = 0; w < NT / 32; ++w) total += wsum[w];
    unsigned nsm;
    asm("mov.u32 %0, %%nsmid;" : "=r"(nsm));
    long long budget = nsm / 2 - B;
    if (budget < 1) budget = 1;
    long long per = (total + budget - 1) / budget;
    if (per < 1) per = 1;
    for (int b = t; b < B; b += NT) {
      const long long tiles = (pos_of(b, seq_of(b)) + TILE_KV) / TILE_KV;
      bsk[b] = (int)min(max((tiles + per - 1) / per, 1ll), (long long)split_max);
    }
    return;
  }
  // ---- gather: the context's cached latent rows for the expansion, GITER of them a thread in flight
  if (absorbed || t >= GROWS * LANES) return;
  const int lane = t % LANES, r0 = (blockIdx.x - 2 * B - 1) * GROWS * GITER + t / LANES;
  uint4 v[GITER];
  bool put[GITER];
#pragma unroll
  for (int k = 0; k < GITER; ++k) {
    const int r = r0 + k * GROWS;
    v[k] = make_uint4(0, 0, 0, 0);
    put[k] = false;
    if (r < n) {
      int j = 0;
      while (j + 1 < nseq && cum_kv[j + 1] <= r) ++j;
      const int local = r - cum_kv[j];
      const int cached = lens[j] - (cum_q[j + 1] - cum_q[j]);
      put[k] = local < cached || local >= lens[j];  // this chunk's rows: its head blocks write them
      if (local < cached)
        v[k] = *reinterpret_cast<const uint4*>(slab + block_table[j * max_pages + local / PAGE] * page_stride +
                                               (long long)(local % PAGE) * KV_A + lane * 8);
    }
  }
#pragma unroll
  for (int k = 0; k < GITER; ++k)
    if (put[k]) *reinterpret_cast<uint4*>(latent_g + (long long)(r0 + k * GROWS) * KV_A + lane * 8) = v[k];
}

// k3_mla_vup_gate.cu's kernel as the absorbed form's blocks: the attention's latent output through W_UV,
// then the gate, read from the fused projection and landed as k3_mla_prep would have.
#ifndef HEADS
#define HEADS 96
#endif
#define NOPE 128
#define LAT 512
#define RB 8      // rows per group
#define GROUPS 4  // row groups per block
#define DS 32     // dv per block
#define JS 8      // j-slices per block
#define JW (LAT / JS)

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

// The decode attention's latent output of row b, head h, columns c .. c + 8: its KV splits merged as
// the DSL's reduction kernel merges them (k3_dcp.cu's kern_k3_dcp_exchange): row b ran
// S = ceil(t / ceil(t / bsk[b])) splits of its t = ceil(len / 128) tiles, l = m + log2(sum_s
// exp2(l_s - m)), o = bf16(sum_s acc_o[s] * exp2(l_s - l)).
#define M_TILE 128
__device__ __forceinline__ uint4 merged_lat(const float* __restrict__ acc_o, const float* __restrict__ acc_lse,
                                            const int* __restrict__ row_lens, const int* __restrict__ bsk,
                                            int split_max, int b, int h, int c) {
  const int tiles = (row_lens[b] + M_TILE - 1) / M_TILE, per = (tiles + bsk[b] - 1) / bsk[b];
  const int splits = (tiles + per - 1) / per;
  const long long rec = (long long)b * M_TILE + h;
  const float* ls = acc_lse + rec * split_max;
  float m = -__int_as_float(0x7f800000);
  for (int i = 0; i < splits; ++i) m = fmaxf(m, ls[i]);
  if (m == -__int_as_float(0x7f800000)) m = 0.f;
  float sum = 0.f;
  for (int i = 0; i < splits; ++i) sum += ex2(ls[i] - m);
  const float l = m + lg2(sum);
  float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const float4* src = reinterpret_cast<const float4*>(acc_o + rec * split_max * LAT + c);
  for (int i = 0; i < splits; ++i, src += LAT / 4) {
    const float w = ex2(ls[i] - l);
    const float4 x = src[0], y = src[1];
    a[0] = fmaf(x.x, w, a[0]), a[1] = fmaf(x.y, w, a[1]), a[2] = fmaf(x.z, w, a[2]), a[3] = fmaf(x.w, w, a[3]);
    a[4] = fmaf(y.x, w, a[4]), a[5] = fmaf(y.y, w, a[5]), a[6] = fmaf(y.z, w, a[6]), a[7] = fmaf(y.w, w, a[7]);
  }
  uint4 v;
  __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k) v2[k] = __floats2bfloat162_rn(a[2 * k], a[2 * k + 1]);
  return v;
}

__device__ __forceinline__ void vup_gate_block(const float* __restrict__ acc_o, const float* __restrict__ acc_lse,
                                               const int* __restrict__ row_lens, const int* __restrict__ bsk,
                                               int split_max, const __nv_bfloat16* __restrict__ w_kv_b,
                                               const float* __restrict__ partial, __nv_bfloat16* __restrict__ gated,
                                               int B, int bx, int h, int dz) {
  __shared__ __align__(16) __nv_bfloat16 lat[RB][LAT];
  __shared__ float red[JS][RB][DS];
  const int dv0 = dz * DS, t = threadIdx.x;
  const int js = t / DS, dl = t - js * DS, dv = dv0 + dl, j0 = js * JW;
  const uint4* __restrict__ w = reinterpret_cast<const uint4*>(w_kv_b + ((size_t)h * 256 + NOPE + dv) * LAT + j0);
  uint4 wv[JW / 8];
#pragma unroll
  for (int k = 0; k < JW / 8; ++k) wv[k] = w[k];
  for (int g = 0; g < GROUPS; ++g) {
    const int b0 = (bx * GROUPS + g) * RB;
    const int nr = min(RB, B - b0);
    if (nr <= 0) break;
    // RB*LAT/8 / 256 = 2 vectors per thread, the splits merged here
    uint4 lv[RB * (LAT / 8) / 256];
#pragma unroll
    for (int k = 0; k < RB * (LAT / 8) / 256; ++k) {
      const int i = t + k * 256, r = i / (LAT / 8), c = i - r * (LAT / 8);
      lv[k] = merged_lat(acc_o, acc_lse, row_lens, bsk, split_max, min(b0 + r, B - 1), h, c * 8);
    }
    if (g) __syncthreads();  // the previous group is done reading lat / red
#pragma unroll
    for (int k = 0; k < RB * (LAT / 8) / 256; ++k) {
      const int i = t + k * 256, r = i / (LAT / 8), c = i - r * (LAT / 8);
      reinterpret_cast<uint4*>(&lat[r][0])[c] = r < nr ? lv[k] : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    float acc[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc[r] = 0.f;
#pragma unroll
    for (int k = 0; k < JW / 8; ++k) {
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[k]);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(w2[j]);
        wf[2 * j] = f.x;
        wf[2 * j + 1] = f.y;
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint4 l = *reinterpret_cast<const uint4*>(&lat[r][j0 + k * 8]);
        const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&l);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(l2[j]);
          acc[r] += wf[2 * j] * f.x + wf[2 * j + 1] * f.y;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) red[js][r][dl] = acc[r];
    __syncthreads();
    for (int i = t; i < nr * DS; i += 256) {
      const int r = i / DS, d = i - r * DS;
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < JS; ++k) a += red[k][r][d];
      const size_t oi = (size_t)(b0 + r) * (HEADS * NOPE) + (size_t)h * NOPE + dv0 + d;
      const float gf = landf(partial[(size_t)(b0 + r) * MLA_FUSED + HEADC + (size_t)h * NOPE + dv0 + d]);
      gated[oi] = __hmul(__float2bfloat16_rn(a), __float2bfloat16_rn(1.0f / (1.0f + expf(-gf))));
    }
  }
}


extern "C" __global__ void __launch_bounds__(256) kern_k3g_mla_gate(
    bf16_t* __restrict__ out, const bf16_t* __restrict__ o, const float* __restrict__ partial,
    const float* __restrict__ acc_o, const float* __restrict__ acc_lse, const int* __restrict__ row_lens,
    const int* __restrict__ bsk, int split_max, const bf16_t* __restrict__ w_kv_b, int n, int short_max, int B) {
  const int gate_blocks = B * (n / 2048);
  if (B <= short_max) {  // a short chunk: o_lat through W_UV, then the gate (blocks past the gate's)
    if (blockIdx.x < gate_blocks) return;
    const int i = blockIdx.x - gate_blocks;
    vup_gate_block(acc_o, acc_lse, row_lens, bsk, split_max, w_kv_b, partial, out, B, i / (HEADS * 4), i / 4 % HEADS,
                   i % 4);
    return;
  }
  if (blockIdx.x >= gate_blocks) return;
  const long long b = blockIdx.x / (n / 2048);
  const int j = (blockIdx.x % (n / 2048) * blockDim.x + threadIdx.x) * 8;
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
