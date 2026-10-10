// MVP embedding gather：manifest `embedding` 核的实现（唯一自己写的核）。
// ABI 与 manifest 声明一致：grid.x = tokens，block 256。
// A row is copied in 16-byte vectors, all of a thread's loads in flight before its stores
// (one bf16 a thread an iteration was a DRAM round trip per 512 bytes: 15 us for 8 rows).
//   nvcc -cubin -arch=sm_103a -o kernels/embedding.cubin tools/kernels-src/embedding.cu
#include <cuda_bf16.h>

#define PER 4  // vectors a thread holds at once: 4 x 256 x 16 B covers a 16 KiB row in one pass

extern "C" __global__ void kern_embedding_i64_bf16(
    const long long* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out, int tokens, int hidden) {
  long long row = ids[blockIdx.x];
  const __nv_bfloat16* src = table + row * hidden;
  __nv_bfloat16* dst = out + (long long)blockIdx.x * hidden;
  if (hidden % 8 == 0) {
    const uint4* s = reinterpret_cast<const uint4*>(src);
    uint4* d = reinterpret_cast<uint4*>(dst);
    const int n = hidden / 8;
    for (int j0 = threadIdx.x; j0 < n; j0 += PER * blockDim.x) {
      uint4 v[PER];
#pragma unroll
      for (int k = 0; k < PER; ++k)
        if (j0 + k * (int)blockDim.x < n) v[k] = s[j0 + k * blockDim.x];
#pragma unroll
      for (int k = 0; k < PER; ++k)
        if (j0 + k * (int)blockDim.x < n) d[j0 + k * blockDim.x] = v[k];
    }
  } else {
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) dst[j] = src[j];
  }
}
