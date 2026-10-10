// `k3_kda_rec` — the packed prefill's KDA recurrence and output gate in one
// launch: FlashKDA's kernel 2 (source/flash-kda, `_flash_kda_fwd_recurrence`)
// over its kernel 1's workspace, operation for operation, with the state read
// from and written to each sequence's KDA line, then K11 (k3_kda_out_gate.cu)
// on every output row and the conv windows advanced (K9's
// `kern_k3_span_state_out_packed`).
//
//   extern "C" __global__ void kern_k3_kda_rec(
//       CUtensorMap tm_kd, tm_qd, tm_kr,  // [HEADS * tiles * 16 rows][128] bf16, box 64 x 16, 128B swizzle
//       CUtensorMap tm_v,                 // span_v [rows][INNER], box 64 x 16, 128B swizzle
//       CUtensorMap tm_gate,              // the projections [rows][4 * INNER], box 128 x 16
//       CUtensorMap tm_inv, tm_mqk,       // [HEADS * tiles * 16 rows][16] bf16, box 16 x 16
//       const bf16* v,          // [rows, INNER]  conv'd v (span_v), a partial tile's rows
//       const bf16* beta,       // [HEADS, span]  beta logits (span_beta)
//       const float* ws_gt,     // [HEADS * tiles][128]
//       void* kda_base, const int* line_index, long long line_bytes,  // sequence j's line line_index[j]
//       const bf16* partial,    // [rows, 4 * INNER]  q | k | v | gate projections
//       const float* gamma_o,   // [128]
//       bf16* gated,            // [rows, INNER]  rows span_at[0] + i
//       const int* span_at,
//       const long long* cu_seqlens, const int* tile_prefix,  // sequence j: rows cu[j]..cu[j+1], tiles from prefix[j]
//       int tiles,              // the workspace's head stride (kernel 1's total_tiles)
//       int span);              // rows of the call (beta's row stride)
//
//   grid (nseq, HEADS)   block 512   dynamic smem sizeof(K12Smem) = 215040
//
// Warps 0-7 run the recurrence, warp w the 16 value columns 16w..16w+16 of
// the head's state, held as bf16 in registers as the A operand of
// mma.m16n8k16: FlashKDA's products transposed (S [dv][dk] times kd^T
// instead of kd times S^T), which sums the same products in the same k order
// per output element. Warps 8-15 finish the output rows (K11 to the bit)
// from a four-stage tile ring and advance the windows; warp 8 also streams
// each 16-row tile's inputs into an eight-stage ring (TMA, 128-byte swizzle
// so ldmatrix is conflict-free), seven tiles ahead of the one it gates.
// FlashKDA's roundings are kept:
// the state bf16 between tiles (fma.ftz then RN), u and the output in bf16
// (HADD2 / HMUL2), beta = bf16(sigmoid by tanh.approx).
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef HEADS
#define HEADS 96
#endif
#define K12_INNER (HEADS * 128)
#define K12_FUSED (4 * K12_INNER)
#define K12_REC_BYTES ((long long)HEADS * 128 * 128 * 4)
#define K12_WIN_BYTES ((long long)3 * K12_INNER * 2)
#define K12_STAGES 8
#define K12_OSTAGES 4
#define K12_ROWS_MAX 8192 // a sequence's rows: its betas stay in shared memory
#define K12_ROW 272   // a 128-column bf16 row, padded
#define K12_MMA_WARPS 8
#define K12_EPI_WARPS 8
#define K12_THREADS ((K12_MMA_WARPS + K12_EPI_WARPS) * 32)
#define K12_RMS_EPS 1e-5f

typedef __nv_bfloat16 bf16;

// A 16-row tile of 128 bf16 columns as TMA lands it with the 128-byte
// swizzle: two boxes of 64 columns, 16-byte chunk c of row r at c ^ (r % 8).
struct K12Stage {
  alignas(1024) unsigned char kd[4096];
  alignas(1024) unsigned char qd[4096];
  alignas(1024) unsigned char kr[4096];
  alignas(1024) unsigned char v[4096];
  alignas(128) unsigned char gate[16 * 256];
  alignas(128) unsigned char inv[16 * 32];
  alignas(128) unsigned char mqk[16 * 32];
  alignas(128) float gt[128];
};

struct K12Smem {
  K12Stage in[K12_STAGES];
  alignas(128) unsigned char out[K12_OSTAGES][16 * K12_ROW];
  bf16 beta[K12_ROWS_MAX];  // bf16(sigmoid(logit)), the sequence's rows
  uint64_t full[K12_STAGES], empty[K12_STAGES], ofull[K12_OSTAGES], oempty[K12_OSTAGES];
};

// the launch's dynamic shared memory (scripts/varlen_abi.py REC_SMEM), 1024-aligned by the kernel
static_assert(sizeof(K12Smem) == 215040, "K12Smem");

// A CUtensorMap by value (the manifest's bytes<128> tensormap field).
struct alignas(64) K12Tmap {
  unsigned char bytes[128];
};

__device__ __forceinline__ unsigned k12_smem(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void k12_bar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(k12_smem(bar)), "r"(count));
}

__device__ __forceinline__ void k12_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(k12_smem(bar)) : "memory");
}

__device__ __forceinline__ void k12_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(k12_smem(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void k12_wait(uint64_t* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n"
      "wait_%=:\n"
      " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra wait_%=;\n}" ::"r"(k12_smem(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void k12_bulk(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   k12_smem(dst)),
               "l"(src), "r"(bytes), "r"(k12_smem(bar))
               : "memory");
}

__device__ __forceinline__ void k12_tma(void* dst, const K12Tmap& map, int x, int y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
          "r"(k12_smem(dst)),
      "l"(&map), "r"(x), "r"(y), "r"(k12_smem(bar))
      : "memory");
}

// byte offset of row `row`, 16-byte chunk `chunk` (of 16) in a swizzled tile
__device__ __forceinline__ unsigned k12_sw(int row, int chunk) {
  return (chunk >> 3) * 2048 + row * 128 + (((chunk & 7) ^ (row & 7)) << 4);
}

__device__ __forceinline__ void k12_ldsm(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void k12_ldsm_t(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void k12_stsm_t(unsigned addr, const unsigned (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(r[0]),
               "r"(r[1]), "r"(r[2]), "r"(r[3])
               : "memory");
}

// d = a * b (+ d), m16n8k16, bf16 in, f32 accumulate
__device__ __forceinline__ void k12_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned k12_pack(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *(const unsigned*)&p;
}

__device__ __forceinline__ __nv_bfloat162 k12_b2(unsigned x) { return *(const __nv_bfloat162*)&x; }
__device__ __forceinline__ unsigned k12_u(__nv_bfloat162 x) { return *(const unsigned*)&x; }
__device__ __forceinline__ float k12_lo(unsigned x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float k12_hi(unsigned x) { return __uint_as_float(x & 0xffff0000u); }

__device__ __forceinline__ float k12_fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

// FlashKDA's sigmoid_tanh_approx_f32 as its fast-math build runs it
__device__ __forceinline__ float k12_sigmoid(float x) {
  float h, th;
  asm("mul.ftz.f32 %0, %1, 0f3F000000;" : "=f"(h) : "f"(x));
  asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(h));
  return k12_fma_ftz(th, 0.5f, 0.5f);
}

// 1 / x as IEEE division computes it (its fast path: reciprocal, two Newton
// steps) for x in [1, 2^126); the caller takes a larger x by `/`.
__device__ __forceinline__ float k12_recip(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  r = fmaf(r, fmaf(-x, r, 1.0f), r);
  return fmaf(r, fmaf(-x, r, 1.0f), r);
}

// The state, every value column of this warp: a[kb] the A fragment of dk 16kb..16kb+16.
struct K12State {
  unsigned a[8][4];
};

__device__ __forceinline__ void k12_recurrence(K12Smem& sm, const float* rec_in, float* rec_out, int len) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int m = lane >> 3, r = lane & 7;
  // FlashKDA's fp32 -> bf16 state load: RN
  K12State S;
  {
    const float* row0 = rec_in + (size_t)(16 * w + g) * 128 + 2 * t;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const float2 a0 = *(const float2*)(row0 + 16 * kb), a1 = *(const float2*)(row0 + 8 * 128 + 16 * kb);
      const float2 a2 = *(const float2*)(row0 + 16 * kb + 8), a3 = *(const float2*)(row0 + 8 * 128 + 16 * kb + 8);
      S.a[kb][0] = k12_pack(a0.x, a0.y);
      S.a[kb][1] = k12_pack(a1.x, a1.y);
      S.a[kb][2] = k12_pack(a2.x, a2.y);
      S.a[kb][3] = k12_pack(a3.x, a3.y);
    }
  }
  // per-lane ldmatrix rows: 8(m/2)+r for kd / qd / v / inv / mqk / out, 8(m%2)+r for kr (transposed)
  const int row_a = 8 * (m >> 1) + r, row_t = 8 * (m & 1) + r;
  const unsigned off_sq = row_a * 32 + 16 * (m & 1);
  const unsigned off_out = row_a * K12_ROW + 16 * (m & 1) + 32 * w;
  const int tiles = (len + 15) >> 4;
  for (int c = 0; c < tiles; ++c) {
    const int s = c % K12_STAGES, o = c % K12_OSTAGES;
    K12Stage& st = sm.in[s];
    k12_wait(&sm.full[s], (c / K12_STAGES) & 1);
    // Phase 1: u^T = S kd^T, out^T = S qd^T over dk in FlashKDA's k order
    float u[2][4] = {}, q[2][4] = {};
    const unsigned kd = k12_smem(st.kd), qd = k12_smem(st.qd);
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      unsigned bk[4], bq[4];
      const unsigned off = k12_sw(row_a, 2 * kb + (m & 1));
      k12_ldsm(bk, kd + off);
      k12_ldsm(bq, qd + off);
      k12_mma(u[0], S.a[kb], bk[0], bk[1]);
      k12_mma(u[1], S.a[kb], bk[2], bk[3]);
      k12_mma(q[0], S.a[kb], bq[0], bq[1]);
      k12_mma(q[1], S.a[kb], bq[2], bq[3]);
    }
    // Phase 2: u = (v - bf16(u)) * beta in bf16; the pairs run over two rows i at one dv
    unsigned vv[4];
    k12_ldsm_t(vv, k12_smem(st.v) + k12_sw(row_a, 2 * w + (m & 1)));
    unsigned ua[4];
    {
      const unsigned b0 = *(const unsigned*)&sm.beta[16 * c + 2 * t], b1 = *(const unsigned*)&sm.beta[16 * c + 8 + 2 * t];
      const unsigned uc[4] = {k12_pack(u[0][0], u[0][1]), k12_pack(u[0][2], u[0][3]), k12_pack(u[1][0], u[1][1]),
                              k12_pack(u[1][2], u[1][3])};
#pragma unroll
      for (int k = 0; k < 4; ++k)
        ua[k] = k12_u(__hmul2(__hsub2(k12_b2(vv[k]), k12_b2(uc[k])), k12_b2(k < 2 ? b0 : b1)));
    }
    // Phase 3: U^T = u^T INV^T
    unsigned U[4];
    {
      unsigned bi[4];
      k12_ldsm(bi, k12_smem(st.inv) + off_sq);
      float x[2][4] = {};
      k12_mma(x[0], ua, bi[0], bi[1]);
      k12_mma(x[1], ua, bi[2], bi[3]);
      U[0] = k12_pack(x[0][0], x[0][1]);
      U[1] = k12_pack(x[0][2], x[0][3]);
      U[2] = k12_pack(x[1][0], x[1][1]);
      U[3] = k12_pack(x[1][2], x[1][3]);
    }
    // Phase 4: out = bf16(Mqk U) + bf16(q S)
    unsigned out[4];
    {
      unsigned bm[4];
      k12_ldsm(bm, k12_smem(st.mqk) + off_sq);
      float x[2][4] = {};
      k12_mma(x[0], U, bm[0], bm[1]);
      k12_mma(x[1], U, bm[2], bm[3]);
      out[0] = k12_u(__hadd2(k12_b2(k12_pack(x[0][0], x[0][1])), k12_b2(k12_pack(q[0][0], q[0][1]))));
      out[1] = k12_u(__hadd2(k12_b2(k12_pack(x[0][2], x[0][3])), k12_b2(k12_pack(q[0][2], q[0][3]))));
      out[2] = k12_u(__hadd2(k12_b2(k12_pack(x[1][0], x[1][1])), k12_b2(k12_pack(q[1][0], q[1][1]))));
      out[3] = k12_u(__hadd2(k12_b2(k12_pack(x[1][2], x[1][3])), k12_b2(k12_pack(q[1][2], q[1][3]))));
    }
    // Phase 6: S = bf16(fma.ftz(S, g_total[dk], U^T kr))
    {
      const unsigned kr = k12_smem(st.kr);
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        unsigned bk[4];
        k12_ldsm_t(bk, kr + k12_sw(row_t, 2 * kb + (m >> 1)));
        float x0[4] = {}, x1[4] = {};
        k12_mma(x0, U, bk[0], bk[1]);
        k12_mma(x1, U, bk[2], bk[3]);
        const float2 g0 = *(const float2*)&st.gt[16 * kb + 2 * t], g1 = *(const float2*)&st.gt[16 * kb + 8 + 2 * t];
        unsigned* a = S.a[kb];
        a[0] = k12_pack(k12_fma_ftz(k12_lo(a[0]), g0.x, x0[0]), k12_fma_ftz(k12_hi(a[0]), g0.y, x0[1]));
        a[1] = k12_pack(k12_fma_ftz(k12_lo(a[1]), g0.x, x0[2]), k12_fma_ftz(k12_hi(a[1]), g0.y, x0[3]));
        a[2] = k12_pack(k12_fma_ftz(k12_lo(a[2]), g1.x, x1[0]), k12_fma_ftz(k12_hi(a[2]), g1.y, x1[1]));
        a[3] = k12_pack(k12_fma_ftz(k12_lo(a[3]), g1.x, x1[2]), k12_fma_ftz(k12_hi(a[3]), g1.y, x1[3]));
      }
    }
    __syncwarp();
    if (lane == 0) k12_arrive(&sm.empty[s]);
    // the output tile, rows i, this warp's 16 columns
    k12_wait(&sm.oempty[o], ((c / K12_OSTAGES) & 1) ^ 1);
    k12_stsm_t(k12_smem(sm.out[o]) + off_out, out);
    __syncwarp();
    if (lane == 0) k12_arrive(&sm.ofull[o]);
  }
  // FlashKDA's bf16 -> fp32 state store
  float* row0 = rec_out + (size_t)(16 * w + g) * 128 + 2 * t;
#pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const unsigned* a = S.a[kb];
    *(float2*)(row0 + 16 * kb) = make_float2(k12_lo(a[0]), k12_hi(a[0]));
    *(float2*)(row0 + 8 * 128 + 16 * kb) = make_float2(k12_lo(a[1]), k12_hi(a[1]));
    *(float2*)(row0 + 16 * kb + 8) = make_float2(k12_lo(a[2]), k12_hi(a[2]));
    *(float2*)(row0 + 8 * 128 + 16 * kb + 8) = make_float2(k12_lo(a[3]), k12_hi(a[3]));
  }
}

// Tile c's inputs into its stage, by the whole warp: a whole tile by 2D TMA;
// a sequence's last, partial one has its v rows stored by hand (zero past the
// sequence) and its gate rows by row.
__device__ __forceinline__ void k12_load(K12Smem& sm, int c, int lane, int len, long long bos, long long out0,
                                       long long ws0, int h, const K12Tmap& tm_kd, const K12Tmap& tm_qd,
                                       const K12Tmap& tm_kr, const K12Tmap& tm_v, const K12Tmap& tm_gate,
                                       const K12Tmap& tm_inv, const K12Tmap& tm_mqk, const bf16* v,
                                       const float* ws_gt, const bf16* partial) {
  const int s = c % K12_STAGES;
  K12Stage& st = sm.in[s];
  const int valid = min(16, len - 16 * c);
  const int ws = (int)(ws0 + c) * 16, row0 = (int)(bos + 16 * c), g0 = (int)(out0 + 16 * c);
  if (lane == 0) {
    k12_wait(&sm.empty[s], ((c / K12_STAGES) & 1) ^ 1);
    k12_expect_tx(&sm.full[s], 3 * 4096 + 2 * 512 + 512 + (valid == 16 ? 2 * 4096 : valid * 256));
    for (int b = 0; b < 2; ++b) {
      k12_tma(st.kd + 2048 * b, tm_kd, 64 * b, ws, &sm.full[s]);
      k12_tma(st.qd + 2048 * b, tm_qd, 64 * b, ws, &sm.full[s]);
      k12_tma(st.kr + 2048 * b, tm_kr, 64 * b, ws, &sm.full[s]);
    }
    k12_tma(st.inv, tm_inv, 0, ws, &sm.full[s]);
    k12_tma(st.mqk, tm_mqk, 0, ws, &sm.full[s]);
    k12_bulk(st.gt, ws_gt + (long long)ws * 8, 512, &sm.full[s]);
    if (valid == 16) {
      k12_tma(st.v, tm_v, h * 128, row0, &sm.full[s]);
      k12_tma(st.v + 2048, tm_v, h * 128 + 64, row0, &sm.full[s]);
      k12_tma(st.gate, tm_gate, 3 * K12_INNER + h * 128, g0, &sm.full[s]);
    }
  }
  __syncwarp();
  if (valid < 16 && lane < 16) {
    const int i = lane;
    const uint4* src = (const uint4*)(v + (long long)(row0 + i) * K12_INNER + h * 128);
#pragma unroll
    for (int k = 0; k < 16; ++k)
      *(uint4*)(st.v + k12_sw(i, k)) = i < valid ? src[k] : make_uint4(0, 0, 0, 0);
    if (i < valid)
      k12_bulk(st.gate + i * 256, partial + (long long)(g0 + i) * K12_FUSED + 3 * K12_INNER + h * 128, 256,
               &sm.full[s]);
  }
  __syncwarp();
  if (lane == 0) k12_arrive(&sm.full[s]);
}

extern "C" __global__ void __launch_bounds__(K12_THREADS, 1) kern_k3_kda_rec(
    const __grid_constant__ K12Tmap tm_kd, const __grid_constant__ K12Tmap tm_qd,
    const __grid_constant__ K12Tmap tm_kr, const __grid_constant__ K12Tmap tm_v,
    const __grid_constant__ K12Tmap tm_gate, const __grid_constant__ K12Tmap tm_inv,
    const __grid_constant__ K12Tmap tm_mqk, const bf16* __restrict__ v, const bf16* __restrict__ beta,
    const float* __restrict__ ws_gt, void* __restrict__ kda_base,
    const int* __restrict__ line_index, long long line_bytes, const bf16* __restrict__ partial,
    const float* __restrict__ gamma_o, bf16* __restrict__ gated, const int* __restrict__ span_at,
    const long long* __restrict__ cu_seqlens, const int* __restrict__ tile_prefix, int tiles, int span) {
  extern __shared__ __align__(1024) unsigned char k12_raw[];
  K12Smem& sm = *(K12Smem*)k12_raw;
  const int j = blockIdx.x, h = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long bos = cu_seqlens[j];
  const int len = (int)(cu_seqlens[j + 1] - bos);
  const int ntiles = (len + 15) >> 4;
  const long long ws0 = (long long)h * tiles + tile_prefix[j], out0 = span_at[0] + bos;
  char* line = (char*)kda_base + (long long)line_index[j] * line_bytes;
  if (threadIdx.x == 0) {
    for (int s = 0; s < K12_STAGES; ++s) {
      k12_bar_init(&sm.full[s], 1);
      k12_bar_init(&sm.empty[s], K12_MMA_WARPS + K12_EPI_WARPS);
    }
    for (int o = 0; o < K12_OSTAGES; ++o) {
      k12_bar_init(&sm.ofull[o], K12_MMA_WARPS);
      k12_bar_init(&sm.oempty[o], K12_EPI_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // beta as FlashKDA's recurrence takes it, zero past the sequence's last tile row
  for (int i = threadIdx.x; i < 16 * ntiles; i += K12_THREADS)
    sm.beta[i] = __float2bfloat16(i < len ? k12_sigmoid(__bfloat162float(beta[(long long)h * span + bos + i])) : 0.0f);
  __syncthreads();

  if (warp < K12_MMA_WARPS) {
    float* rec = (float*)(line + (long long)h * 128 * 128 * 4);
    k12_recurrence(sm, rec, rec, len);
    return;
  }

  // Output gate warps. First the conv windows: the gather read them, nothing after it does.
  const int ew = warp - K12_MMA_WARPS, et = threadIdx.x - K12_MMA_WARPS * 32;
  // Epilogue warp 0 also loads, K12_STAGES - 1 tiles ahead of the one it gates.
  const auto load = [&](int c) {
    if (ew == 0 && c < ntiles)
      k12_load(sm, c, lane, len, bos, out0, ws0, h, tm_kd, tm_qd, tm_kr, tm_v, tm_gate, tm_inv, tm_mqk, v, ws_gt,
               partial);
  };
  for (int c = 0; c < K12_STAGES - 1; ++c) load(c);
  if (len > 0 && et < 128) {
    const int col = h * 128 + et;
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      bf16* win = (bf16*)(line + K12_REC_BYTES + s * K12_WIN_BYTES) + col;
      bf16 nt[3];
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int i = len - 3 + k;
        nt[k] = i >= 0 ? partial[(bos + i) * K12_FUSED + s * K12_INNER + col] : win[(i + 3) * K12_INNER];
      }
#pragma unroll
      for (int k = 0; k < 3; ++k) win[k * K12_INNER] = nt[k];
    }
  }
  // Warp ew gates rows 2ew + half, 16 lanes a row: lane k sums the squares of K11's lane group
  // (columns 32(k/4) + 4l + k%4, l < 8) in K11's butterfly order, then finishes columns 8k..8k+8.
  const int half = lane >> 4, k = lane & 15, i = 2 * ew + half, grp = 64 * (k >> 2);
  float gam[8];
  {
    const float4 g0 = *(const float4*)(gamma_o + 8 * k), g1 = *(const float4*)(gamma_o + 8 * k + 4);
    gam[0] = g0.x, gam[1] = g0.y, gam[2] = g0.z, gam[3] = g0.w, gam[4] = g1.x, gam[5] = g1.y, gam[6] = g1.z,
    gam[7] = g1.w;
  }
  for (int c = 0; c < ntiles; ++c) {
    load(c + K12_STAGES - 1);
    const int s = c % K12_STAGES, o = c % K12_OSTAGES;
    k12_wait(&sm.full[s], (c / K12_STAGES) & 1);
    k12_wait(&sm.ofull[o], (c / K12_OSTAGES) & 1);
    const unsigned char* row = sm.out[o] + i * K12_ROW;
    float y[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      // columns 32G + 8q .. +8: l = 2q at +0..3, l = 2q + 1 at +4..7
      const uint4 raw = *(const uint4*)(row + grp + 16 * q);
      const unsigned w0 = (k & 2) ? raw.y : raw.x, w1 = (k & 2) ? raw.w : raw.z;
      const float x0 = (k & 1) ? k12_hi(w0) : k12_lo(w0), x1 = (k & 1) ? k12_hi(w1) : k12_lo(w1);
      y[2 * q] = x0 * x0;
      y[2 * q + 1] = x1 * x1;
    }
    const float sj = ((y[0] + y[4]) + (y[2] + y[6])) + ((y[1] + y[5]) + (y[3] + y[7]));
    const float t1 = sj + __shfl_xor_sync(0xffffffffu, sj, 2);
    const float wsum = t1 + __shfl_xor_sync(0xffffffffu, t1, 1);
    const int base = 16 * half;
    const float tot = __shfl_sync(0xffffffffu, wsum, base) + __shfl_sync(0xffffffffu, wsum, base + 4) +
                      __shfl_sync(0xffffffffu, wsum, base + 8) + __shfl_sync(0xffffffffu, wsum, base + 12);
    const float rr = rsqrtf(tot * (1.0f / 128.0f) + K12_RMS_EPS);
    const uint4 araw = *(const uint4*)(row + 16 * k);
    const uint4 graw = *(const uint4*)(sm.in[s].gate + i * 256 + 16 * k);
    const unsigned aw[4] = {araw.x, araw.y, araw.z, araw.w}, gw[4] = {graw.x, graw.y, graw.z, graw.w};
    float ov[8], den[8], sig[8];
    bool far = false;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float a = (e & 1) ? k12_hi(aw[e / 2]) : k12_lo(aw[e / 2]);
      const float g = (e & 1) ? k12_hi(gw[e / 2]) : k12_lo(gw[e / 2]);
      ov[e] = __bfloat162float(__float2bfloat16(a * rr * gam[e]));
      // K11's sigmoid: __expf (its ex2 wherever the result is not subnormal, and then 1 + it is 1 either way)
      float ex;
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ex) : "f"(-g * 1.4426950408889634f));
      den[e] = 1.0f + ex;
      sig[e] = k12_recip(den[e]);
      far |= den[e] >= 0x1p126f;
    }
    if (__any_sync(0xffffffffu, far)) {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (den[e] >= 0x1p126f) sig[e] = 1.0f / den[e];
    }
    uint4 res;
    unsigned* rw = (unsigned*)&res;
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const unsigned gt = k12_pack(sig[e], sig[e + 1]);
      rw[e / 2] = k12_pack(ov[e] * k12_lo(gt), ov[e + 1] * k12_hi(gt));
    }
    if (16 * c + i < len) *(uint4*)(gated + (out0 + 16 * c + i) * K12_INNER + h * 128 + 8 * k) = res;
    __syncwarp();
    if (lane == 0) {
      k12_arrive(&sm.oempty[o]);
      k12_arrive(&sm.empty[s]);
    }
  }
}
