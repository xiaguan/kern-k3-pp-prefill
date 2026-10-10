# p-glue: the MoE glue and the MLA glue of a P layer

Target: P stage 1 (`loop/p/bench loop/out/p-stage1.json`), check `loop/p/check loop/out/p-l12.json`.

## For the orchestrator (measurement gap, proposal)

- **P stage 1's bench does not run the MLA attention.** `fmha_lens` (the FMHA's seq_lens_kv |
  cum_q | cum_kv tables) is computed by stage 0's `fmha_lens` call; `kern cut` makes it an `input`
  of stage 1 with no `fill`, so the bench hands stage 1 a table of zeros: `mla_fmha` attends to
  nothing (≈20 us a call at every context) and the context gather copies no cached rows. Measured
  (work/lensx: the same stage 1 recomputing `fmha_lens` from `seq_lens` / `cu_seqlens` at its
  first layer, +1 launch): **97.79 ms/item weighted instead of 52.48**; mla_fmha per call 1.26 ms
  (8192 rows, no prefix), 20.3 / 39.7 / 59.1 ms (8192 rows over 64k / 131k / 197k), 10.5 ms (4096
  over 64k), 2.29 ms (10 rows over 131k), 4.11 ms (354 over 131k). The real P stage spends ~45% of
  its time in the MLA FMHA. loop/p/check's p-l12 has stage 0's call, so numerics are checked on
  real tables. Fix: recompute the tables per stage (as work/lensx) or a `fill` for the cut table.
- **Stage 1's MoE routing is degenerate too:** its `hidden_in` / `blocks_in` arrive as zeros, so
  every row routes to the same 16 experts (FC1 1.7 ms/layer at 8192 rows there vs 3.0 ms on stage
  0's real routing; 0.1 ms vs 1.3 ms at 354 rows). Since 19:54 the bench target is stage 0.
- Stage 0 (main 9b1cce0 + this branch, 225 launches, 127.7 ms/item): per call at 8192 rows over
  131k: mla_fmha 45.8 ms (×3), fc1 2.8 ms, fc2 1.3 ms, flash_kda 1.0 ms, finalize 165 us,
  res_mlp 147 us, res_in 155 us, route 115 us, mla_gate 130 us, prep_gather 97 us. At 10 rows over
  131k (21.7 ms an item): mla_fmha 2.35 ms + expand ~2.3 ms per MLA layer = ~14 ms of the item.
- `work/ab.sh A B OUT`: A, B, A, B in one lease over work/ab.toml (8192@0, 10@131k, 354@131k);
  `work/swapmod.py MANIFEST main OUT` re-pins a manifest's modules to main's cubins (the A side of
  a cubin-only change). The lease's own A-to-A spread is 0.5-1.2% at 8192 rows.
- **Absorbed MLA for short chunks (9f15357 + e9d8b2c), offered separately (it skips the expand
  GEMM for chunks of <= 128 rows).** Stage 0 A/B in one lease: 10 rows over 131k 21.8 -> 7.6 ms
  (-65%), 8192 / 354 rows inside noise; ~ -1.9% weighted. Launches 225 -> 231 (every `when`
  alternative counted; the decode attention's reduction is merged by the gate kernel, the prep and
  the gate each carry both forms' blocks). Numerics: KL <= 2.1e-3 (check), <= 2.8e-3 with 120-row chunks (kern test against the
  expanded form via work/pcheck.sh on --max-ctx 16384 manifests: at 64k, saving kv_exp for every
  span runs kern test out of device memory past ~10 chunks). The decode kernel's bf16-rounded
  softmax scale gave KL 8.8e-3; the short form uses the FMHA's f32 scale. Why not for 354 rows:
  per context token B x 128 x 1088 x 2 FLOP (heads padded to 128, latent dims) against 576 x 30720
  x 2 + B x 96 x 320 x 2 for expand + FMHA: the crossover is ~160 rows. Measured (A/B, threshold 512
  vs 128): 200 rows over 131k −4.2% absorbed, 354 rows +9%: the real crossover is ~250 rows. The
  threshold stays 128: the decode kernel's split workspace is rows × 16 splits × 256 KB (512 MB at
  128 rows; 256 would be 1 GB of a stage's memory for items the bench does not weigh).
  (Answer to 20:49: moving the threshold does not pay on the 354-row shape: 354 rows absorbed is +9%
  (A/B, threshold 512); a threshold of 256 leaves the 354-row shape on the expanded form.)
- The split merge in the gate kernel is the DSL reduction's own arithmetic (4ccb71c, read off its
  SASS: lane = split, butterfly max / sum, MUFU ex2 / lg2, FMA accumulation in split order):
  bit-identical end-to-end logits to the reduction-launch form; +0.2% on the 10-row item (~5 us a
  gate call for the warp butterflies and a barrier).
- **Proposal (a GEMM epilogue, so not done): sh_down accumulates onto lat_up's output.** With
  `cublaslt_bf16_tn_acc` (beta 1, D in place) the shared expert's down projection adds into
  `routed_partial`; the next layer's `land_add2_attnres_rms` then reads one partial (two = 0):
  −234 MB a MoE layer at 8192 rows (~30 us, ~0.35% of the 8192-row item). Numerics: hidden =
  bf16(prefix2 + bf16(bf16(latup) + shdown)) instead of bf16(prefix2 + bf16(latup) + bf16(shdown)).
- **Robustness note:** `moe_route`'s grid barrier needs its 304 blocks (two an SM, the whole
  register file) co-resident. A kernel running beside it on another stream (a PP transfer) only
  delays it (its blocks wait for that kernel to drain), it cannot deadlock it unless that kernel
  waits on this stream.

## State

| commit | launches/layer (stage1) | cost ms/item | note |
|---|---|---|---|
| main 2026-10-10 | 27.1 | 56.82 | |
| moe route | 24.2 | 56.69 | router_topk + moe_quant + 3 FlashInfer routing → 1 (+1 init per stage) |
| situ in route | 23.2 | 56.64 | shared expert's situ made by the route kernel |
| (main 8f3d6a4, with p-kda) | 20.9 | 54.19 | |
| residual on 14 warps | 20.9 | 53.96 | K1a/K1b/K1d 448 threads a row, bit-identical |
| (main 4df7918, with p-kda) | 19.4 | 52.99 | |
| mla prep + gather | 19.2 | 52.87 | prep head + varlen gather one launch; gate reads the f32 projection |
| residual sw from L1, 3 rows/SM | 19.2 | 52.75 | 60 → 40 regs |
| route reciprocals | 19.2 | 52.65 | rcp/div fast paths without the IEEE slow-path branch |
| finalize smem picks + FFMA2 | 19.2 | 52.53 | 190 → 163 us at 8192 rows |
| route: pieces to warps without a top-k row | 19.2 | 52.48 | short chunks: top-k beside quant/situ |
| route: batched piece loads | 19.2 | 52.42 | stage 0 A/B: inside noise |
| gather: 4 rows a thread | 18.8 (stage 0) | stage 0 A/B −0.2..−0.3% | with the batched route |
| absorbed MLA ≤ 128 rows | 19.2 (stage 0, 231) | stage 0 A/B: 10 rows −65% | offered separately (skips expand) |

A MoE layer is now: `land_add_attnres_rms_bf16` → router* → lat_down* → wsh* → `moe_route` →
fc1* → fc2* → `moe_finalize_rms` → lat_up* → sh_down* → next `land_add2_attnres_rms`.
An MLA layer: wfu* → `mla_prep_gather` → expand* → q_b* → `mla_fmha` → `mla_gate` → o_proj*.
Non-GEMM launches: 4 a layer (+ `moe_route_init` once a stage). Without touching the GEMMs that is
the floor: every remaining glue kernel sits between two GEMMs.

Shares (ablate, main 8f3d6a4): moe_finalize_rms 2.03%, land_add_attnres_rms_bf16 2.02%,
land_add2_attnres_rms 1.96%, moe_route 1.52%, mla_prep 0.30%, mla_gate 0.29%, latent_gather 0.08%.
Per call at 8192 rows (bracketed): residual ~195 us (floor ~100 us at 8 TB/s), finalize 190 us
(floor ~125 us), route 136 us, mla_prep 113 us, mla_gate 113 us (floor ~75 us).

## Tooling

- `work/ops.py REPORT [op...]`: per-op in-program time per scenario (bracketed, inflated for small ops).
- `work/hb/`: harness of k3_moe_route.cu against the old top-k / quant / situ kernels and a CPU
  build of the tables (`loop/withgpu -n 1 ./hb`).
- `work/hr/`: harness of the residual kernels (K1b / K1d) of a candidate source against the
  committed one, bit-compared and timed at 10 / 354 / 8192 rows, nb 1 / 2 / 4.

## Done

### k3_moe_route.cu: top-k + quant + routing tables (+ situ) in one launch
- Persistent grid (304 blocks of 512, 64 up to 256 rows; two an SM, 64 regs) with one grid
  barrier; barrier and tables live in a workspace zeroed by `moe_route_init` where a stage starts
  (a carry is refused by `kern cut`: "writes carry buffer before the last stage").
- Top-k by threshold (16th largest lane max) + candidate ranking instead of 16 serial rounds;
  contiguous experts per lane; sigmoid recomputed for the picks. Bit-identical picks / weights.
- Token order inside an expert matters to moe_fc1: atomic order cost +40 us/layer of FC1 at 8192
  rows. Tables are built from per-expert row bitmaps + 256-row chunk counts: exact row order,
  deterministic.
- situ (and the quant) as 256-column pieces spread over every warp of the grid; at a row a warp
  it made 354-row items slower (one warp per row, latency-bound).
- Role split (half the warps top-k, half quant/situ): better at small B, worse at 8192 (top-k warps
  become the tail). Not kept.

### Residual kernels on 14 warps (bit-identical)
- 7 → 14 warps a row: 80 → 60 regs, no spills, 2 blocks/SM: 197.6 → 173.6 us (K1d, 8192, nb 2).
- Tried on top (harness, all bit-identical, none kept):
  - packed fp32 (FFMA2/FADD2: (sq, dp) chains in lockstep, mix pairs): no gain alone; with the
    first two snapshot candidates prefetched into registers it spilled (72 regs).
  - TMA bulk staging of the row's operands + 2 candidates in smem (mbarrier): 168.7 us at nb 2,
    but slower at 354 rows (11.5 vs 10.1 us) and nb 4 (231 vs 218 us).
  - 28 warps (1 vector a thread): 243 us; launch bounds 3 blocks: spills.
  - What remains is half issue (≈ 890 warp-instr a row a warp: unpacking, 7 butterflies of 2
    groups, the scoring and the mix) and half DRAM at 58% of peak.

## Out of scope (measured, kept on branch `wip/gemm-merge`, not in main)

The orchestrator ruled GEMM merges out of scope (18:13). Measured before that:
- lat_down + wsh as one GEMM (weight [lat_down; gate; up], 15872 columns): −1 launch/layer at equal
  weighted cost (10 rows −0.8%, 354 rows +0.26%); cuBLASLt picks another kernel, a few landed
  bf16 values move one ulp: end-to-end KL ≤ 2.5e-3, 2 near-tie flips.
- lat_up + sh_down as one K-concat GEMM ([routed latent normed | shared act] @ [lat_up | sh_down]^T,
  K = 9728, weight concatenated by columns in the derive program): −1 launch/layer more and −0.33%
  weighted (one bf16 partial instead of two; `land_add2_attnres_rms` with two = 0). Numerics:
  hidden = bf16(prefix2 + bf16(latup + shdown)) instead of bf16(prefix2 + bf16(latup) + bf16(shdown)).

### MLA glue (k3_mla_glue.cu, bit-identical)
- `kern_k3g_mla_prep_gather`: grid = B head blocks (mla_prep's fast head minus the gate) + ceil(ctx/7)
  gather blocks (7 rows × 72 lanes of 512 threads); the chunk's own latent rows go from the head
  blocks to latent_g, the gather skips them. Packed chunk only.
- `kern_k3g_mla_gate` reads the gate's f32 columns of the fused projection and lands them itself:
  mla_prep no longer writes a bf16 gate copy (−403 MB read, −201 MB write per MLA layer at 8192).
- The fused kernel looked +10 us/MLA layer slower at 354 rows than prep + gather separately
  (variant B, within noise); kept for the launch.
- rcp_fast in the gate (sigmoid): −2% of the gate kernel, no measurable change: not kept.

### Reciprocals without the slow-path branch (route)
- nvcc's `1.0f / d`, `__frcp_rn(d)` and `a / 448.f` each emit a fast path + FCHK/exponent test +
  BSSY/CALL slow path per element; in the route kernel 116 BSSY / 70 CALL. `rcp_fast` (rcp.approx,
  fma residual, fma correction) equals both for every float in [2^-126, 2^126) (exhaustive check,
  work/hx/t.cu); taken when a warp vote says every lane is in range. `div448` equals a / 448.f for
  every bf16 magnitude but inf (passed through). Route 126.8 → 109.5 us at 8192, 21 → 15.4 at 354.

### Finalize
- 16 (row, weight) picks staged in smem by 16 threads instead of 32 broadcast global loads a thread,
  FFMA2 accumulation: 182.8 → 157.6 us (harness; a pure gather of the same rows is 148.8 us).
- Tried: 16 / 8 / 4 loads in flight per thread (104 / 64 / 56 regs): slower at 8192 (occupancy);
  2 or 4 rows a block: slower.

### Tried, not kept
- PDL (`pdl: true` + griddepcontrol.wait at the top) on every glue launch: 10 rows −0.5%, but 8192
  rows +0.9% (the early-resident blocks slow the GEMMs' tails); weighted +0.7%.
- Route phase 2 with the first row's places computed before the scan + a tight barrier spin:
  −0.7 us at 354 rows in the harness, no change in the bench.
- Residual with packed FFMA2 mix + FADD2 butterflies on top of the 40-register form: spills, slower.

## Tooling gap (fixed by the orchestrator in a405bd0)
`kern test` diffs ops, not module bytes: a kernel-only change read "nothing to test". loop/p/check
now replays a rebuilt module under its old name (`work/forcediff.py` did the same by hand).

## Next

- Launches: every glue kernel left sits between GEMMs / the prebuilt FMHA; the only one left is
  `moe_route_init` (1 a stage), which needs a zeroed workspace (a carry is refused by `kern cut`).
- mla_gate: only neighbours are the prebuilt FMHA and cuBLAS o_proj: no fusion without our own FMHA.
- Time: residual K1b / K1d (153 / 163 us at 8192, nb 2; floor ~86 / 100 us): issue-bound half the
  time (58% issue, ~14k warp-instr a row); finalize at the gather floor + 9 us; route 108 us
  (situ ~45 us of it, MUFU-bound: 4 MUFU an element).
