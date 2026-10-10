// p_fmha_split.cu: one-GPU harness for the prebuilt TRT-LLM gen context FMHA (h192 / v128, causal,
// Q256 x Kv128): the plain causal call vs the packed prefill's three-piece KV split (k3_fmha_plan.cuh:
// N normal | R with q and k reversed | A the leading triangle) merged through the kernel's softmax
// stats, against an f32 reference.
//
//   nvcc -O2 -std=c++17 -arch=sm_103a p_fmha_split.cu -o p_fmha_split -lcuda
//   ./p_fmha_split prebuilt/trtllm_fmha_ctx_h192_v128.cubin check [ROWS PREFIX]   stats and split vs f32
//   ./p_fmha_split prebuilt/trtllm_fmha_ctx_h192_v128.cubin time ROWS PREFIX [C...] plain vs split at each c
//   env: QCAP / KVCAP descriptor capacities, SUSTAIN=N N timed calls first (power cap),
//        EMPTY=1 the plain call with two empty extra sequences
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s); exit(1); } } while (0)
#define RT(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, \
  cudaGetErrorString(e_)); exit(1); } } while (0)

constexpr int H = 96, HQK = 192, HV = 128, KVR = 320, TILE = 256, SMEM = 199296, PARAMS = 1344;
constexpr float SCALE_LOG2 = 0.10411754995584488f;
const char* ENTRY = "fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvCausalVarSeqQ256Kv128PersistentContext";

static void tmap(uint8_t* at, void* ptr, std::vector<uint64_t> dims, std::vector<uint64_t> strides, std::vector<uint32_t> box) {
  CUtensorMap m;
  std::vector<uint32_t> es(dims.size(), 1);
  CK(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dims.size(), ptr, dims.data(), strides.data(), box.data(),
                            es.data(), CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  memcpy(at, &m, 128);
}

struct Call {
  void *q, *kv, *o, *seqlens, *cumq, *cumkv, *stats, *scratch;
  int qcap, kvcap, batch, maxq, sumq;
};

static void params(uint8_t* p, const Call& c) {
  memset(p, 0, PARAMS);
  auto i32 = [&](int at, int v) { memcpy(p + at, &v, 4); };
  auto f32 = [&](int at, float v) { memcpy(p + at, &v, 4); };
  auto ptr = [&](int at, void* v) { memcpy(p + at, &v, 8); };
  tmap(p + 0, c.q, {HQK, 1, H, (uint64_t)c.qcap}, {HQK * 2, HQK * 2, H * HQK * 2}, {64, 1, 1, 128});
  tmap(p + 128, c.kv, {HQK, (uint64_t)c.kvcap, H, 1}, {H * KVR * 2, KVR * 2, 16}, {64, 128, 1, 1});
  tmap(p + 384, (char*)c.kv + HQK * 2, {HV, (uint64_t)c.kvcap, H, 1}, {H * KVR * 2, KVR * 2, 16}, {64, 128, 1, 1});
  tmap(p + 512, c.o, {HV, (uint64_t)c.qcap, H, 1, 1}, {H * HV * 2, HV * 2, H * HV * 2, 16}, {64, 128, 1, 1, 1});
  ptr(912, c.o); ptr(936, c.cumq); ptr(944, c.cumkv); ptr(1024, (char*)c.scratch + 311296); ptr(1032, c.scratch);
  ptr(1096, c.seqlens); ptr(1112, c.stats);
  i32(1128, 0x7FFFFFFF); i32(1132, c.batch); i32(1172, c.maxq); i32(1176, c.kvcap); i32(1180, (c.maxq + TILE - 1) / TILE);
  i32(1184, 1); i32(1192, H); i32(1196, H); i32(1200, 1); i32(1204, 1); i32(1208, INT32_MIN); i32(1216, -1);
  int64_t ho = H * HQK; memcpy(p + 1224, &ho, 8);
  i32(1236, TILE); i32(1240, -1); i32(1244, 1); f32(1248, 1.f); f32(1252, SCALE_LOG2); f32(1260, 1.f);
  i32(1276, c.sumq); i32(1280, c.kvcap);
}

static CUfunction fn;
static void launch(const Call& c, cudaStream_t s = 0) {
  alignas(128) static uint8_t p[PARAMS];
  params(p, c);
  void* args[] = {p};
  CK(cuLaunchKernel(fn, (c.maxq + TILE - 1) / TILE, H, c.batch, 512, 1, 1, SMEM, (CUstream)s, args, nullptr));
}

__global__ void fill(__nv_bfloat16* x, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    float u1 = ((h & 0xffff) + 0.5f) / 65536.f, u2 = ((h >> 16) + 0.5f) / 65536.f;
    x[i] = __float2bfloat16(scale * sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2));
  }
}

// f32 reference: row r of sequence s (rows cq[s].., keys ck[s].. of length kl[s]) sees keys [0, kl - ql + i]
__global__ void reference(const __nv_bfloat16* q, const __nv_bfloat16* kv, const int* cq, const int* ck, const int* kl, int seq,
                          float* o, float* lse, int rstride) {
  int r = blockIdx.x, h = blockIdx.y, i = r - cq[seq], ql = cq[seq + 1] - cq[seq];
  if (i < 0 || i >= ql) return;
  int n = kl[seq] - ql + i + 1;
  extern __shared__ float sm[];
  float* qs = sm; float* red = sm + HQK;
  for (int d = threadIdx.x; d < HQK; d += blockDim.x) qs[d] = __bfloat162float(q[((size_t)r * H + h) * HQK + d]);
  __syncthreads();
  float m = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const __nv_bfloat16* k = kv + ((size_t)(ck[seq] + j) * H + h) * KVR;
    float acc = 0; for (int d = 0; d < HQK; ++d) acc += qs[d] * __bfloat162float(k[d]);
    m = fmaxf(m, acc * SCALE_LOG2);
  }
  red[threadIdx.x] = m; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) { if (threadIdx.x < w) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + w]); __syncthreads(); }
  m = red[0]; __syncthreads();
  float acc_o[HV] = {}; float sum = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const __nv_bfloat16* k = kv + ((size_t)(ck[seq] + j) * H + h) * KVR;
    float a = 0; for (int d = 0; d < HQK; ++d) a += qs[d] * __bfloat162float(k[d]);
    float p = exp2f(a * SCALE_LOG2 - m); sum += p;
    for (int d = 0; d < HV; ++d) acc_o[d] += p * __bfloat162float(k[HQK + d]);
  }
  red[threadIdx.x] = sum; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) { if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w]; __syncthreads(); }
  sum = red[0]; __syncthreads();
  float* out = o + ((size_t)i * H + h) * HV;
  if (threadIdx.x == 0) { for (int d = 0; d < HV; ++d) out[d] = 0; lse[(size_t)i * H + h] = m + log2f(sum); }
  __syncthreads();
  for (int d = 0; d < HV; ++d) atomicAdd(out + d, acc_o[d] / sum);
}

// piece layout of one sequence (ql rows, prefix p, c split): keys of piece N = [c+1, p+ql) normal, R = [1, c] reversed
// (R[t] = key c - t), A = [0, ql) normal; Q of R reversed
__global__ void build_pieces(const __nv_bfloat16* q, const __nv_bfloat16* kv, int ql, int p, int c, __nv_bfloat16* q3,
                             __nv_bfloat16* kv3) {
  size_t qrow = (size_t)H * HQK, kvrow = (size_t)H * KVR;
  int nN = p + ql - c - 1;
  size_t total_kv = (size_t)nN + c + ql;
  for (size_t t = blockIdx.x; t < total_kv + 3 * (size_t)ql; t += gridDim.x) {
    const __nv_bfloat16* src; __nv_bfloat16* dst; size_t len;
    if (t < total_kv) {
      size_t key = t < (size_t)nN ? c + 1 + t : t < (size_t)nN + c ? c - (t - nN) : t - nN - c;
      src = kv + key * kvrow; dst = kv3 + t * kvrow; len = kvrow;
    } else {
      size_t u = t - total_kv, piece = u / ql, i = u % ql;
      size_t from = piece == 1 ? ql - 1 - i : i;
      src = q + from * qrow; dst = q3 + u * qrow; len = qrow;
    }
    for (size_t e = threadIdx.x * 8; e < len; e += blockDim.x * 8) *(uint4*)(dst + e) = *(const uint4*)(src + e);
  }
}

// out[i] = merge of the three pieces' rows (piece 1's row reversed) through their (max, sum) stats
template <int LOG2>
__global__ void merge3(const __nv_bfloat16* o3, const float2* st, int ql, __nv_bfloat16* o) {
  int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  int rows[3] = {i, ql + ql - 1 - i, 2 * ql + i};
  float m[3], s[3], M = -INFINITY;
  for (int k = 0; k < 3; ++k) { float2 v = st[(size_t)rows[k] * H + h]; m[k] = v.x; s[k] = v.y; M = fmaxf(M, m[k]); }
  float acc = 0, den = 0;
  for (int k = 0; k < 3; ++k) {
    float w = s[k] * (LOG2 ? exp2f(m[k] - M) : expf(m[k] - M));
    den += w; acc += w * __bfloat162float(o3[((size_t)rows[k] * H + h) * HV + d]);
  }
  o[((size_t)i * H + h) * HV + d] = __float2bfloat16(acc / den);
}

template <class T> static T* dev(size_t n) { T* p; RT(cudaMalloc(&p, n * sizeof(T))); RT(cudaMemset(p, 0, n * sizeof(T))); return p; }
static int* devi(std::vector<int> v) { int* p = dev<int>(v.size()); RT(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p; }

static void compare(const char* what, const __nv_bfloat16* o, const float* ref, int ql) {
  size_t n = (size_t)ql * H * HV;
  std::vector<__nv_bfloat16> a(n); std::vector<float> r(n);
  RT(cudaMemcpy(a.data(), o, n * 2, cudaMemcpyDeviceToHost)); RT(cudaMemcpy(r.data(), ref, n * 4, cudaMemcpyDeviceToHost));
  double e2 = 0, r2 = 0, mx = 0;
  for (size_t k = 0; k < n; ++k) { double e = __bfloat162float(a[k]) - r[k]; e2 += e * e; r2 += (double)r[k] * r[k]; mx = fmax(mx, fabs(e)); }
  printf("  %-28s relRMS %.3e  max abs %.3e\n", what, sqrt(e2 / r2), mx);
}

int main(int argc, char** argv) {
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, d)); CK(cuCtxSetCurrent(ctx));
  CUmodule mod; CK(cuModuleLoad(&mod, argv[1])); CK(cuModuleGetFunction(&fn, mod, ENTRY));
  CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM));
  bool check = !strcmp(argv[2], "check");
  int ql = check ? atoi(argc > 3 ? argv[3] : "354") : atoi(argv[3]);
  int p = check ? atoi(argc > 4 ? argv[4] : "3000") : atoi(argv[4]);
  int kvl = p + ql;
  size_t qn = (size_t)std::max(ql, 8192) * H * HQK, kvn = (size_t)std::max(kvl, 262144) * H * KVR;
  auto q = dev<__nv_bfloat16>(qn), kv = dev<__nv_bfloat16>(kvn);
  fill<<<1024, 256>>>(q, qn, 1, 1.f); fill<<<1024, 256>>>(kv, kvn, 2, 1.f);
  auto q3 = dev<__nv_bfloat16>(3 * qn), kv3 = dev<__nv_bfloat16>(kvn + (size_t)ql * H * KVR);
  auto o = dev<__nv_bfloat16>((size_t)ql * H * HV), o3 = dev<__nv_bfloat16>(3 * (size_t)ql * H * HV), om = dev<__nv_bfloat16>((size_t)ql * H * HV);
  auto st = dev<float2>((size_t)ql * H), st3 = dev<float2>(3 * (size_t)ql * H);
  void* scratch = dev<uint8_t>(8 << 20);
  int qcap = getenv("QCAP") ? atoi(getenv("QCAP")) : ql, kvcap = getenv("KVCAP") ? atoi(getenv("KVCAP")) : kvl;
  auto one = Call{q, kv, o, devi({kvl}), devi({0, ql}), devi({0, kvl}), st, scratch, qcap, kvcap, 1, ql, ql};
  auto split = [&](int c) {
    int nN = p + ql - c - 1;
    return Call{q3, kv3, o3, devi({nN, c, ql}), devi({0, ql, 2 * ql, 3 * ql}), devi({0, nN, nN + c, nN + c + ql}), st3, scratch,
                3 * ql, kvl + ql - 1, 3, ql, 3 * ql};
  };
  if (check) {
    auto ref = dev<float>((size_t)ql * H * HV); auto lse = dev<float>((size_t)ql * H);
    reference<<<dim3(ql, H), 128, (HQK + 128) * 4>>>(q, kv, devi({0, ql}), devi({0, kvl}), devi({kvl}), 0, ref, lse, 0);
    launch(one); RT(cudaDeviceSynchronize());
    compare("plain causal", o, ref, ql);
    std::vector<float2> s((size_t)ql * H); std::vector<float> l((size_t)ql * H);
    RT(cudaMemcpy(s.data(), st, s.size() * 8, cudaMemcpyDeviceToHost)); RT(cudaMemcpy(l.data(), lse, l.size() * 4, cudaMemcpyDeviceToHost));
    double e_nat = 0, e_l2 = 0;
    for (size_t k = 0; k < s.size(); ++k) {
      e_nat = fmax(e_nat, fabs(1.4426950408889634 * s[k].x + log2(s[k].y) - l[k]));
      e_l2 = fmax(e_l2, fabs(s[k].x + log2(s[k].y) - l[k]));
    }
    printf("  stats[0] (%g, %g) ref lse2 %g; max |lse err| as natural max %.3e, as log2 max %.3e\n", s[0].x, s[0].y, l[0], e_nat, e_l2);
    for (int c : {ql, (p + ql) / 2, p - 1}) {
      if (c < ql || c > p - 1) continue;
      RT(cudaMemset(o3, 0, 3 * (size_t)ql * H * HV * 2));
      build_pieces<<<4096, 256>>>(q, kv, ql, p, c, q3, kv3);
      launch(split(c));
      merge3<0><<<dim3(ql, H), HV>>>(o3, st3, ql, om); RT(cudaDeviceSynchronize());
      char w[64]; snprintf(w, 64, "split c=%d (natural)", c); compare(w, om, ref, ql);
      merge3<1><<<dim3(ql, H), HV>>>(o3, st3, ql, om); RT(cudaDeviceSynchronize());
      snprintf(w, 64, "split c=%d (max read as log2: wrong)", c); compare(w, om, ref, ql);
    }
    return 0;
  }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto time = [&](auto f) {
    f(); RT(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < 5; ++r) { cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms); }
    std::sort(v.begin(), v.end()); return v[2];
  };
  double flop = 2.0 * ql * ((double)p + (ql + 1) / 2.0) * H * (HQK + HV);
  if (getenv("SUSTAIN")) {
    for (int r = 0; r < atoi(getenv("SUSTAIN")); ++r) {
      cudaEventRecord(e0); launch(one); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
      if (r % 10 == 0) printf("  rep %d %.3f ms\n", r, ms);
    }
  }
  float t0 = time([&] { launch(one); });
  if (getenv("EMPTY")) {
    auto three = Call{q, kv, o, devi({kvl, 0, 0}), devi({0, ql, ql, ql}), devi({0, kvl, kvl, kvl}), st, scratch, qcap, kvcap, 3, ql, ql};
    float te = time([&] { launch(three); });
    printf("rows %d prefix %d: plain %.3f ms, with two empty z slices %.3f ms\n", ql, p, t0, te);
    return 0;
  }
  printf("rows %d prefix %d: plain %.3f ms (%.0f TF/s)\n", ql, p, t0, flop / t0 / 1e9);
  for (int k = 5; k < argc; ++k) {
    int c = atoi(argv[k]);
    if (c < ql || c > p - 1) { printf("  c=%d out of range [%d, %d]\n", c, ql, p - 1); continue; }
    build_pieces<<<4096, 256>>>(q, kv, ql, p, c, q3, kv3);
    float t1 = time([&] { launch(split(c)); });
    float t2 = time([&] { merge3<0><<<dim3(ql, H), HV>>>(o3, st3, ql, om); });
    printf("  c=%-7d split %.3f ms + merge %.3f  (%.1f%%)\n", c, t1, t2, 100 * (t1 + t2 - t0) / t0);
  }
}
