# kern-k3-pp-prefill

Kimi-K3 prefill as pipeline-parallel stages for the
[kern runtime](https://github.com/pegainfer-project/kern): one manifest per PP
rank, its kernel sources, and the build recipes. This repository follows the
layout of [kern-k3](https://github.com/xiaguan/kern-k3), which holds the
TP4/EP4 tray prefill.

- [`manifests/`](manifests/): the four stages of the **pruned-75pct** K3 (224
  experts, 93 layers) at PP4. The split is 23, 23, 23, 24 layers, the same as
  `SGLANG_PP_LAYER_PARTITION=23,23,23,24`. Each stage is TP1 / EP1 and takes an
  8192-token chunk with a maximum context of 16,384.
- [`kernels.toml`](kernels.toml): every module the manifests pin, with its
  SHA-256 and where it comes from. Handwritten kernels have a `source` and
  compile-time `defines`; vendored builds have a `source` plus a `build`
  script; extracted cubins have a `prebuilt` path and their `upstream`.
- [`source/`](source/): kern's handwritten CUDA (Apache-2.0, `source/LICENSE`),
  vendored FlashKDA (MIT) and vendored FlashInfer MoE routing (Apache-2.0).
  Each vendored tree has its own `LICENSE` and `PROVENANCE.md`.
- [`prebuilt/`](prebuilt/): the cubins with no source recipe here (TRT-LLM gen
  batched GEMM and FMHA), FlashKDA, and the routing build. Licenses and
  origins are in [`prebuilt/README.md`](prebuilt/README.md).
- [`validation/`](validation/): the scripts that checked the stages against
  SGLang PP4.

## A stage

PP does not change any layer's shapes. A stage is the prefill manifest cut
to the layers in `[A, B)`, plus the stream that crosses the boundary:

| | Stage not starting at layer 0 | Stage not ending at the last layer |
|---|---|---|
| hidden stream | input `hidden_in` [T, 7168] | output `hidden` [T, 7168] |
| attention-residual bank | input `blocks_in` [T, 8, 7168], the first ceil(A / 12) snapshots valid | output `blocks` [T, 8, 7168], ceil(B / 12) valid |

These are SGLang's PP proxy tensors: `hidden_states` (the stream head with the
delayed MLP add folded in) and `residual` (the bank, unpadded). The first stage
takes token ids, and the last one ends in the head (`next_token`, and the last
row's logits). The KV and KDA states hold the stage's own layers only.

At EP1 a stage holds all of its layers' experts. The routed MoE runs
TRT-LLM gen's batched mxfp8 × mxfp4 GEMMs (FC1 with the fused SiTU gate, then
FC2). FlashInfer's routing kernels build their routing tables from kern's
router top-k, and kern's combine finishes the layer. This is the path SGLang's
`flashinfer_mxfp4` backend takes.

### Other splits

The manifests are generator output. Generate any split (PP8, PP16, or an
uneven one) from kern's `k3-pp-stage` branch:

```sh
KERN_INDEX_DIR=<kern-kernels index> python3 tools/gen_k3.py \
    --layers 23:46 --ranks 1 --chunk 8192 --max-ctx 16384 > k3-pruned-pp4-l23-46.json
```

Add `--experts 896` for the full checkpoint.

### Running a stage

kern's `crates/kern-run/examples/k3_stage.rs` runs one stage over one chunk.
The boundary tensors come from files and go back to files:

```sh
k3_stage --manifest manifests/k3-pruned-pp4-l23-46.json --weights <kimi-k3-pruned-75pct> \
    --hidden-in hidden.bf16 --blocks-in blocks.bf16 --out <dir> [--gpu 0] [--iters 3]
```

The first stage takes `--tokens ids.i64`. A middle stage writes
`hidden.bf16` and `blocks.bf16` for the next one; the last writes
`next_token.i64` and `logits.f32`.

## Results (2026-10-06, GB300)

Setup: one chunk of 3999 tokens (English prose) over an empty cache, one GPU
per stage.

**Isolation.** Each stage was fed SGLang PP4's received proxy tensors.
- Copied bank rows and the layer-0 snapshot are bit-identical.
- Inside a stage, the difference from SGLang grows about 1% per layer and is
  smallest at row 0, as accumulated precision noise does.
- The head gives SGLang's token and top-5 order, with |Δlogprob| ≤ 0.11.

**Chained.** kern's four stages fed each other. They were judged against the
band two SGLang runs span when they differ only in the KV cache dtype (rel L2,
hidden / bank):

| Boundary | SGLang fp8 ↔ bf16 KV | kern ↔ SGLang fp8 | kern ↔ SGLang bf16 |
|---|---|---|---|
| after l0-23 | 0.159 / 0.071 | 0.169 / 0.080 | 0.167 / 0.079 |
| after l23-46 | 0.196 / 0.100 | 0.201 / 0.107 | 0.199 / 0.105 |
| after l46-69 | 0.290 / 0.201 | 0.288 / 0.206 | 0.289 / 0.206 |
| head, max \|Δlogprob\| over SGLang's top-5 | 0.382 | 0.534 | 0.367 |

- All three runs pick token 2469 at the head.
- The second and third candidates, 603 and 1101, are 0.07 apart in kern and
  swap places.
- Over 4k tokens K3's stream is sensitive enough that the KV dtype alone moves
  the hidden by 16–29%.

**Determinism.** Two runs of every stage wrote byte-identical outputs,
although the routing tables' row order varies between runs.

**Time.** kern per stage, warm, median of 3: 135.8 / 134.9 / 136.9 /
141.4 ms, 549 ms in sum. SGLang PP4 takes 0.605 s end to end for one request,
including hand-offs and scheduling.

## Full K3

The stage form and every kernel cover the full checkpoint (896 experts),
including the 896-expert router top-k (`k3_router_argmax+EXPERTS=896`).
`gen_k3.py --experts 896` generates its stages.

They are not published here yet. A PP8 stage holds about 188 GB of mxfp4
experts. kern loads the checkpoint's tensors and then shuffles them into the
GEMMs' layout, so both copies would sit in the 288 GB of one GB300 at once.

## Build the kernels

Python 3.11+ and CUDA 13.0 nvcc (V13.0.88, `cuda_13.0.r13.0/compiler.36424714_0`)
reproduce the pinned cubins:

```sh
NVCC=/path/to/cuda-13.0/bin/nvcc python3 scripts/build.py
```

`build.py` copies the prebuilt cubins, compiles the handwritten ones with
`nvcc -cubin -arch=sm_103a` and the `defines` in `kernels.toml`, and compares
every result with its pinned SHA-256. The two vendored trees build with their
own `build.sh`; see each directory's `PROVENANCE.md`.

## Check

```sh
python3 scripts/check.py [build]
```

The check verifies that:
- every module a manifest names is pinned in `kernels.toml` with the same hash;
- each module has a source or upstream record;
- every op and launch references something that exists;
- the bundled cubins (and, given a directory, the built ones) match their
  hashes.

CI runs it on every push.

## Validation scripts

- `validation/kcap/` is an SGLang plugin (`SGLANG_PLUGINS=kcap`, the
  directory on `PYTHONPATH`). For the forward of exactly `KCAP_TOKENS` tokens,
  it dumps each PP rank's received and sent proxy tensors.
- `sglang_pp4.sh` starts SGLang at TP1 / PP4 with the plugin.
- `prompt.py` and `request.py` make the prompt and send it.
- `run_iso.sh` and `run_chain.sh` run kern's stages.
- `compare.py`, `blocks.py` and `band.py` do the comparison.

Paths come from the environment: `MODEL`, `OUT`, `DATA`, `WEIGHTS`, `K3_STAGE`.
