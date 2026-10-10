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
//     partial bf16 [B, MLA_FUSED] the fused projection (q_a | kv_a | rope | gate), landed by its GEMM;
//     slab, page_stride: the layer's latent pages; lens: kern_k3_fmha_lens_varlen's tables
//     (seq_lens_kv | cum_q | cum_kv, ns words each) of the call's nseq + 2 FMHA sequences and its
//     split record. A split sequence's rows go to its pieces: keys c + 1 .. in its own span, keys
//     c .. 1 and 0 .. Lq - 1 in the two spans past the last sequence's (its q rows are copied after
//     q_b by k3_mla_absorb.cu's -DQBF16 launch).
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
//   kern_k3g_mla_gate(out, o, partial, ..., stats, lens, ns, n, short_max, B)
//     sigmoid_mul.cu's kern_sigmoid_mul_bf16 on the gate columns of the fused projection:
//     out = bf16(o * bf16(sigmoid(bf16(partial[b, HEADC + j])))), o / out [B, n]. A split sequence's
//     row first merges its three pieces' o through the FMHA's (max, sum) stats.
//   grid (ceil(B / 8) * n / 2048 + the absorbed form's blocks, 1, 1)   block (256, 1, 1)
//
// Every value is the three kernels' bit for bit (the same landings, sums and roundings), but a split
// row's o: an f32 merge of three bf16 pieces.
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
__device__ __forceinline__ void prep_head(const bf16_t* __restrict__ partial, const bf16_t* __restrict__ gamma_q_a,
                                          const bf16_t* __restrict__ gamma_kv_a, const long long* __restrict__ slot_mapping,
                                          bf16_t* __restrict__ slab, long long page_stride, bf16_t* __restrict__ q_norm,
                                          bf16_t* __restrict__ grow, int b, int t) {
  __shared__ float red[NT / 32];
  const bf16_t* __restrict__ P = partial + (long long)b * MLA_FUSED;
  const bool isq = (t < QU);
  const int col = isq ? (4 * t) : (Q_LORA + 4 * (t - QU));
  const uint2 v = *reinterpret_cast<const uint2*>(P + col);
  const float2 v01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  const float2 v23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  const float x0 = v01.x, x1 = v01.y, x2 = v23.x, x3 = v23.y;

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
    rope4 = *reinterpret_cast<const uint2*>(P + Q_LORA + KV_LORA + 4 * t);
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
    const bf16_t* __restrict__ partial, const bf16_t* __restrict__ gamma_q_a, const bf16_t* __restrict__ gamma_kv_a,
    const long long* __restrict__ slot_mapping, bf16_t* __restrict__ slab, long long page_stride,
    bf16_t* __restrict__ q_norm, const int* __restrict__ block_table, int max_pages, const int* __restrict__ lens,
    int ns, int nseq, bf16_t* __restrict__ latent_g, int n, int* __restrict__ row_table, int* __restrict__ row_lens,
    int* __restrict__ bsk, int split_max, int short_max, int B) {
  const int t = threadIdx.x;
  const int* __restrict__ cum_q = lens + ns;
  const int* __restrict__ cum_kv = lens + 2 * ns;
  const int* __restrict__ rec = lens + 3 * ns;
  const int slq = rec[3], split = slq > 0 ? rec[0] : -1, sc = rec[1], sq0 = rec[2], sp = rec[4];
  const bool absorbed = B <= short_max;
  const auto seq_of = [&](int b) {
    int j = 0;
    while (j + 1 < nseq && cum_q[j + 1] <= b) ++j;
    return j;
  };
  // row b of sequence j: its place among the sequence's rows (the cached ones first), its causal length
  const auto pos_of = [&](int b, int j) { return lens[j] - (cum_q[j + 1] - cum_q[j]) + (b - cum_q[j]); };

  if (blockIdx.x < B) {  // ---- head: q_norm, kv_norm, rope, the latent row (also to latent_g for the expansion)
    const int b = blockIdx.x, j = seq_of(b), i = b - sq0;
    if (j == split)  // its chunk rows are keys sp + i of piece N, which starts at key c + 1
      prep_head(partial, gamma_q_a, gamma_kv_a, slot_mapping, slab, page_stride, q_norm,
                latent_g + (long long)(cum_kv[j] + sp + i - sc - 1) * KV_A, b, t);
    else
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
      int z = 0;
      while (z + 1 < nseq + 2 && cum_kv[z + 1] <= r) ++z;
      const int local = r - cum_kv[z];
      // the piece's sequence and key: its own span, or the split's N (from key c + 1), R (c down), A (0 up)
      const int j = z < nseq ? z : split;
      const int key = z == split ? local + sc + 1 : z == nseq ? sc - local : local;
      const int cached = z == split || z >= nseq ? sp : lens[z] - (cum_q[z + 1] - cum_q[z]);
      put[k] = local >= lens[z] || key < cached;  // this chunk's rows: its head blocks write them
      if (local < lens[z] && key < cached)
        v[k] = *reinterpret_cast<const uint4*>(slab + block_table[j * max_pages + key / PAGE] * page_stride +
                                               (long long)(key % PAGE) * KV_A + lane * 8);
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

// The decode attention's KV splits of row b, head h merged as the DSL's reduction kernel merges them
// (its SASS: lane s holds split s, the max and the sum of exp2(l_s - m) by xor butterflies 16 .. 1,
// l = m + lg2(sum), split s scaled by exp2(l_s - l)): row b ran S = ceil(t / ceil(t / bsk[b])) splits of
// its t = ceil(len / 128) tiles. One warp a (row, head); scale[s] for s < S.
#define M_TILE 128
__device__ __forceinline__ int splits_of(const int* __restrict__ row_lens, const int* __restrict__ bsk, int b) {
  const int tiles = (row_lens[b] + M_TILE - 1) / M_TILE, per = (tiles + bsk[b] - 1) / bsk[b];
  return (tiles + per - 1) / per;
}
__device__ __forceinline__ void split_scales(const float* __restrict__ acc_lse, int split_max, int splits, int b, int h,
                                             int lane, float* __restrict__ scale) {
  const float l = lane < splits ? acc_lse[((long long)b * M_TILE + h) * split_max + lane] : -__int_as_float(0x7f800000);
  float m = l;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (m == -__int_as_float(0x7f800000)) m = 0.f;
  float sum = 0.f + ex2(l - m);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float g = m + lg2(sum);
  if (lane < splits) scale[lane] = ex2(l - g);
}

__device__ __forceinline__ void vup_gate_block(const float* __restrict__ acc_o, const float* __restrict__ acc_lse,
                                               const int* __restrict__ row_lens, const int* __restrict__ bsk,
                                               int split_max, const __nv_bfloat16* __restrict__ w_kv_b,
                                               const bf16_t* __restrict__ partial, __nv_bfloat16* __restrict__ gated,
                                               int B, int bx, int h, int dz) {
  __shared__ __align__(16) __nv_bfloat16 lat[RB][LAT];
  __shared__ float red[JS][RB][DS];
  __shared__ float scale[RB][32];
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
    if (g) __syncthreads();  // the previous group is done reading lat / red / scale
    // warp r: row r's split scales
    if ((t >> 5) < RB) {
      const int b = min(b0 + (t >> 5), B - 1);
      split_scales(acc_lse, split_max, splits_of(row_lens, bsk, b), b, h, t & 31, scale[t >> 5]);
    }
    __syncthreads();
    // RB*LAT/8 / 256 = 2 vectors per thread: the splits' sum in split order, as the reduction runs it
    uint4 lv[RB * (LAT / 8) / 256];
#pragma unroll
    for (int k = 0; k < RB * (LAT / 8) / 256; ++k) {
      const int i = t + k * 256, r = i / (LAT / 8), c = i - r * (LAT / 8), b = min(b0 + r, B - 1);
      const int splits = splits_of(row_lens, bsk, b);
      float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const float4* src = reinterpret_cast<const float4*>(acc_o + ((long long)b * M_TILE + h) * split_max * LAT + c * 8);
      for (int sp = 0; sp < splits; ++sp, src += LAT / 4) {
        const float w = scale[r][sp];
        const float4 x = src[0], y = src[1];
        a[0] = fmaf(x.x, w, a[0]), a[1] = fmaf(x.y, w, a[1]), a[2] = fmaf(x.z, w, a[2]), a[3] = fmaf(x.w, w, a[3]);
        a[4] = fmaf(y.x, w, a[4]), a[5] = fmaf(y.y, w, a[5]), a[6] = fmaf(y.z, w, a[6]), a[7] = fmaf(y.w, w, a[7]);
      }
      __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&lv[k]);
#pragma unroll
      for (int q = 0; q < 4; ++q) v2[q] = __floats2bfloat162_rn(a[2 * q], a[2 * q + 1]);
    }
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
      const float gf = __bfloat162float(partial[(size_t)(b0 + r) * MLA_FUSED + HEADC + (size_t)h * NOPE + dv0 + d]);
      gated[oi] = __hmul(__float2bfloat16_rn(a), __float2bfloat16_rn(1.0f / (1.0f + expf(-gf))));
    }
  }
}


// a split row's three pieces' o (N at row b, R at B + Lq - 1 - i, A at B + Lq + i) merged through the
// FMHA's stats, landed to bf16
__device__ __forceinline__ uint4 merge_split(const bf16_t* __restrict__ o, const float2* __restrict__ stats,
                                             const int* __restrict__ rec, long long b, int j, int n, int B) {
  const int i = (int)b - rec[2], lq = rec[3], h = j / NOPE;
  const long long rows[3] = {b, B + lq - 1 - i, B + lq + i};
  float2 st[3];
  float m = -__int_as_float(0x7f800000);
#pragma unroll
  for (int k = 0; k < 3; ++k) st[k] = stats[rows[k] * (n / NOPE) + h], m = fmaxf(m, st[k].x);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, den = 0.f;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float w = st[k].y * expf(st[k].x - m);
    const uint4 pv = *reinterpret_cast<const uint4*>(o + rows[k] * n + j);
    const __nv_bfloat162* pp = reinterpret_cast<const __nv_bfloat162*>(&pv);
    den += w;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(pp[e]);
      acc[2 * e] = fmaf(w, f.x, acc[2 * e]), acc[2 * e + 1] = fmaf(w, f.y, acc[2 * e + 1]);
    }
  }
  const float inv = 1.f / den;
  return make_uint4(pack2(acc[0] * inv, acc[1] * inv), pack2(acc[2] * inv, acc[3] * inv),
                    pack2(acc[4] * inv, acc[5] * inv), pack2(acc[6] * inv, acc[7] * inv));
}

// GATE_ROWS rows a thread, every load issued before the first use: the kernel's registers and shared
// memory are the absorbed form's, two blocks an SM, so one row's loads alone leave HBM idle
#define GATE_ROWS 8

// 1 / d without the IEEE division's slow-path branch, equal to it for every d in [2^-126, 2^126)
// (k3_moe_route.cu); outside, the division
__device__ __forceinline__ float rcp_fast(float d) {
  float q;
  asm("{\n .reg .f32 r, e;\n rcp.approx.ftz.f32 r, %1;\n fma.rn.f32 e, %1, r, 0fBF800000;\n neg.f32 e, e;\n"
      " fma.rn.f32 %0, r, e, r;\n}"
      : "=f"(q)
      : "f"(d));
  return q;
}

// bf16(a * bf16(sigmoid(g))) for 8 columns; FAST: every denominator in rcp_fast's range
template <bool FAST>
__device__ __forceinline__ uint4 gate8(const uint4& gv, const uint4& av) {
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
  const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&av);
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 x = __bfloat1622float2(gp[i]);
    const float dx = 1.0f + expf(-x.x), dy = 1.0f + expf(-x.y);
    const __nv_bfloat162 g =
        FAST ? __floats2bfloat162_rn(rcp_fast(dx), rcp_fast(dy)) : __floats2bfloat162_rn(1.0f / dx, 1.0f / dy);
    op[i] = __floats2bfloat162_rn(__bfloat162float(ap[i].x) * __bfloat162float(g.x),
                                  __bfloat162float(ap[i].y) * __bfloat162float(g.y));
  }
  return ov;
}

// sigmoid's denominator 1 + e^-x is in rcp_fast's range for every x >= -87 (e^87 < 2^126), NaN not
__device__ __forceinline__ bool gate_ok(const uint4& gv) {
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
  bool ok = true;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 x = __bfloat1622float2(gp[i]);
    ok = ok && x.x >= -87.0f && x.y >= -87.0f;
  }
  return ok;
}

// rows b0 .. b0 + GATE_ROWS of columns j .. j + 8. A warp's whole tile without a split row and with
// every x in rcp_fast's range runs as straight-line code: no division branch to keep its rows apart.
__device__ __forceinline__ void gate_rows(bf16_t* __restrict__ out, const bf16_t* __restrict__ o,
                                          const bf16_t* __restrict__ partial, const float2* __restrict__ stats,
                                          const int* __restrict__ lens, int ns, int n, int B) {
  const long long b0 = (long long)(blockIdx.x / (n / 2048)) * GATE_ROWS;
  const int j = (blockIdx.x % (n / 2048) * blockDim.x + threadIdx.x) * 8;
  uint4 gv[GATE_ROWS], av[GATE_ROWS];
#pragma unroll
  for (int r = 0; r < GATE_ROWS; ++r)
    if (b0 + r < B) {
      gv[r] = *reinterpret_cast<const uint4*>(partial + (b0 + r) * MLA_FUSED + HEADC + j);
      av[r] = *reinterpret_cast<const uint4*>(o + (b0 + r) * n + j);
    }
  const int* __restrict__ rec = lens + 3 * ns;
  bool ok = b0 + GATE_ROWS <= B && (rec[3] == 0 || b0 + GATE_ROWS <= rec[2] || b0 >= rec[2] + rec[3]);
#pragma unroll
  for (int r = 0; r < GATE_ROWS; ++r) ok = ok && gate_ok(gv[r]);
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int r = 0; r < GATE_ROWS; ++r) *reinterpret_cast<uint4*>(out + (b0 + r) * n + j) = gate8<true>(gv[r], av[r]);
    return;
  }
#pragma unroll
  for (int r = 0; r < GATE_ROWS; ++r) {
    const long long b = b0 + r;
    if (b >= B) break;
    if (rec[3] > 0 && b >= rec[2] && b < rec[2] + rec[3]) av[r] = merge_split(o, stats, rec, b, j, n, B);
    *reinterpret_cast<uint4*>(out + b * n + j) = gate8<false>(gv[r], av[r]);
  }
}

extern "C" __global__ void __launch_bounds__(256) kern_k3g_mla_gate(
    bf16_t* __restrict__ out, const bf16_t* __restrict__ o, const bf16_t* __restrict__ partial,
    const float* __restrict__ acc_o, const float* __restrict__ acc_lse, const int* __restrict__ row_lens,
    const int* __restrict__ bsk, int split_max, const bf16_t* __restrict__ w_kv_b, const float2* __restrict__ stats,
    const int* __restrict__ lens, int ns, int n, int short_max, int B) {
  const int gate_blocks = (B + GATE_ROWS - 1) / GATE_ROWS * (n / 2048);
  if (B <= short_max) {  // a short chunk: o_lat through W_UV, then the gate (blocks past the gate's)
    if (blockIdx.x < gate_blocks) return;
    const int i = blockIdx.x - gate_blocks;
    vup_gate_block(acc_o, acc_lse, row_lens, bsk, split_max, w_kv_b, partial, out, B, i / (HEADS * 4), i / 4 % HEADS,
                   i % 4);
    return;
  }
  if (blockIdx.x >= gate_blocks) return;
  gate_rows(out, o, partial, stats, lens, ns, n, B);
}
