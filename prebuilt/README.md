# Prebuilt kernels

These cubins are the exact artifacts `../kernels.toml` pins. `scripts/build.py`
copies them into `build/` and checks each one's SHA-256; CI checks the bundled
bytes as well.

| Cubin | Origin | License |
|---|---|---|
| `bmm_*.cubin` (2) | FlashInfer 0.6.18 TRT-LLM gen batched-GEMM bundle (exact identity in `../kernels.toml`): FC1 with the fused SiTU gate, then FC2 | `LICENSE.trtllm-gen.txt` (Apache-2.0) |
| `trtllm_fmha_ctx_h192_v128.cubin` | FlashInfer 0.6.18 TRT-LLM gen FMHA bundle; the context kernel FlashInfer selects for DeepSeek-style MLA prefill on GB300 | `LICENSE.trtllm-gen.txt` |
| `flash_kda_d128.cubin` | existing build of the vendored FlashKDA in `../source/flash-kda/` | MIT, see that directory |
| `flash_kda_d128_varlen.cubin` | the same sources built with `KERN_DEFINES=-DKERN_VARLEN` (upstream's varlen instantiation, for the packed prefill) | MIT, see that directory |
| `flashinfer_moe_routing.cubin` | built from `../source/flashinfer-moe-routing/` | Apache-2.0, see that directory and its `PROVENANCE.md` |

No source recipe is available here for the three TRT-LLM gen cubins.

`flash_kda_d128.cubin` was built with CUDA 13.1 and an unrecorded CUTLASS 4.x
revision, so a rebuild may not reproduce its bytes. `flash_kda_d128_varlen.cubin`
was built with CUDA 13.0 nvcc (V13.0.88) against the CUTLASS 4.5.0 headers
bundled in FlashInfer 0.6.13, via `KERN_DEFINES=-DKERN_VARLEN
CUTLASS_INCLUDE=<cutlass>/include ../source/flash-kda/build.sh <out>`.

`flashinfer_moe_routing.cubin` rebuilds byte for byte with the recipe in its
`PROVENANCE.md`.
