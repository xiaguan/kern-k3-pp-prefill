# p-glue: the MoE glue and the MLA glue of a P layer

Target: P stage 0 since 19:54 (`loop/p/bench loop/out/p-stage0.json`), check `loop/p/check
loop/out/p-l12.json`. Every timing claim goes through `loop/p/ab A B OUT [N]` (A, B, A, B in one
lease; A-to-A spread 0.5-1.2% at 8192 rows); `work/swapmod.py MANIFEST main OUT` re-pins a manifest's cubins to main's.

## Session 2026-10-10 23:00- (main 96fb15b, rebased f03a989)

Stage 0 now: 200 launches (16.7/layer). Non-GEMM launches a layer: MoE `land_add_attnres_rms_bf16`,
`moe_route`, `moe_finalize_rms`, next `land_add2_attnres_rms`; KDA `span_gather`, FlashKDA prepare,
`k3_kda_rec`; MLA `mla_prep_gather`, (absorb / decode attention short | FMHA long), `mla_gate`.

Stage 0 baseline this session (one bench): 124.12 ms/item. Glue per 8192-row item (in-program,
bracketed): residuals 3.9 ms (K1b 171 + K1d 170 us a call), finalize 2.0, route 1.4, mla_gate 0.6,
situ 0.35. Harness floors at 8192 rows, nb 1, two 0: K1b / K1d 128 us (bytes 587 MB: 73 us),
finalize 157 us (a pure gather of its rows 149), route 98 us (420 MB: 52 us), gate 140 us (603 MB:
75 us), situ 280 us (1.66 GB: 207 us).

Committed (both bit-identical, P check PASS at every span):
- `the p mla gate runs a whole tile branch-free`: the gate spent ~48 instructions an element, each
  sigmoid's IEEE division in its own BSSY/BSYNC slow-path region, so a thread's 8 rows x 8 columns
  ran serially (ncu: stalls "wait" + "branch resolving", DRAM 28%). A warp's tile without a split
  row and with every x >= -87 runs straight-line with rcp_fast. Harness 139.7 -> 94.1 us at 8192;
  stage-0 A/B in-program gate 579 -> 382..396 us per 8192-row item (item-level pairs within drift).
  A separate long-form entry (its own registers) is 94 vs 100 us: not worth a generator change.
- `the p situ kernel takes the reciprocal without its slow-path branch`: same pattern in layer 0's
  situ, 280 -> 259 us at 8192 rows. Module shared with D.

- `a p chunk's kda layer projects q | k | v | g and wsm in one gemm`: qkvg (N 49152) and wsm (N 256)
  read the same normed rows; one GEMM with D's wkda weight layout, span_gather / k3_kda_rec read
  their slices by stride (-DK9_LD / -DK12_LD, defaults unchanged: p-kda's builds byte-identical).
  Stage 0 209 -> 200 launches (16.7/layer). A/B 2 pairs +0.11 / +0.28%, 3 pairs -0.04 / -0.01 /
  -0.59%; the GEMMs' own in-program time -40..-423 us an item in 6 of 7 scenarios. P check PASS,
  KL <= 3.92e-3, 1 near-tie flip (cuBLASLt picks another kernel for N 49408). It was the last pair
  of GEMMs in a P layer sharing an input with matching output dtypes: router (f32 out) and front
  (bf16) differ; a router in the front GEMM would need f32 for both (+260 MB write and read a call
  at 8192 rows) or bf16 routing logits.

Tried, not kept (harness, bit-identical unless said):
- Residual K1b/K1d persistent (grid 3/SM) with the next row's operands and candidate 0 staged by
  cp.async in shared memory (each thread copies its own slots: wait_group, no barrier): slower
  everywhere, 128 -> 145 us at 8192, +1-2 us even with one row a block. ncu of the committed K1d
  (nb 1): 10.9k warp-instr a row, issue 56%, DRAM 54%, 61% occupancy, 154 spill ld/st a row; the
  dynamic block scheduler already overlaps rows better than a one-row prefetch.
- Residual with nb a template constant (nb == 1 dispatched): nb 1 -3..-5% at 8192, -7% at 354, but
  the dynamic path in the same kernel +5% (registers are per kernel). As per-nb module variants it
  would be ~0.1 ms a stage-0 item: not worth the ops-per-nb generator change.
- Route phases (harness, 8192 rows): whole 97.5 us; without situ 51.3, without quant 84.5, without
  the post-barrier tables 86.7, top-k + tables alone 33.5, top-k alone 23.1. Each piece sits near its
  own floor (situ alone ~46 us vs 38 us of bytes); the loss is that top-k (compute, DRAM idle) and
  the streaming do not overlap. Staggering the order by warp parity (odd warps stream first): 99.2
  vs 97.5 us. Prefetching into shared memory cannot cover it (~128 KB an SM against ~1 MB an SM of
  DRAM time idle during the top-k).
- Finalize: at the gather's floor (above), not retried.
- mla_prep_gather's head two rows a block (256 threads a row, each thread two of k3_mla_prep.cu's
  4-column lanes with their own butterflies: bit-identical, check PASS): stage-0 A/B in-program, three
  calls an item, 8192@0 145 / 145 -> 143 / 145 us, 354 rows 160 / 160 -> 174 / 176 us (half the
  head blocks). The head (~48 us a call at 8192 rows) is not short of rows in flight; the gather part
  runs ~7 TB/s at 196k. Not kept.
- Harness lesson: the first gate harness fed hashed bit patterns as bf16 (huge / NaN gates): every
  element took the division's slow path and the kernel read 256 us. Use normal data.

## Where it stands (main aa144e0, all of this run's commits merged)

- Stage 0: 206 launches (17.2/layer), ~125 ms/item. A MoE layer: `land_add_attnres_rms_bf16`
  → router* → front* (lat_down | shared gate | up) → `moe_route` (top-k, mxfp8, routing tables,
  situ) → fc1* → fc2* → `moe_finalize_rms` → back* (lat_up | sh_down by K) → next
  `land_add2_attnres_rms`. An MLA layer: wfu* → `mla_prep_gather` → expand* (long) → q_b* →
  absorb (short) → mla_fmha (long) / decode attention (short) → `mla_gate` → o_proj*.
  Every glue kernel left sits between GEMMs, the prebuilt FMHA or the decode attention. A stage's
  setup (the routing state's zeroing, stage 0's FMHA tables) rides in its first launch (b038d30:
  stage 0 206 launches, 17.2/layer); the stage-end `land_add2` stays (nothing after it in a stage).
- Cost now is MLA attention and GEMMs: at 8192 rows over 131k mla_fmha is 45.8 ms a layer (68
  TFLOP at ~1.5 PF: at peak), at 354 rows expand 2.6 + FMHA 4.5 ms a layer (192 CTAs, 1.26
  waves, each head's 84 MB of expanded K/V streamed per 256-row Q tile), fc1 1.3 ms a layer
  (streaming all 896 experts' weights). The 10-row item is 7.6 ms (GEMM weight streaming).
- Next, if anyone: an absorbed MLA *prefill* kernel (rows × heads in M, the latent read once) for
  129..~1000-row chunks: ~9.7 TFLOP at 354 rows over 131k against 21 ms of expand + FMHA today,
  but only as a tcgen05 kernel (mma.sync peaks at 605 TFLOP/s here: work/mma).

## Power cap (22:16 target; main aa144e0, 8192@0 unless said)

Tools (uncommitted, work/): `ptrace.sh MANIFEST OUT N SCENARIO...` (one kern bench per scenario,
nvidia-smi beside it), `pw/` (one kernel or a cuBLAS GEMM looped at 8192 rows; `mix:<filler>:<n>`
times a GEMM after n fillers), `ncu_inner.sh` + `ncu_sum.py` (DRAM bytes per kernel of one stage pass).
Pitfalls: nvidia-smi's index is not CUDA's device 0 on a lease (sample by the CUDA device's UUID,
`pw/uuid`); NVML's clock and power refresh every ~100 ms whatever the poll rate (nvidia-smi -lms 10
or a tight NVML loop alike), so no trace resolves a kernel; nsys is not installed.

- Stage scenarios (40 samples each, active span): 1400 W limit, 2070 MHz max.

  | scenario | power median | SM clock median / mean / min | capped | at 2070 |
  |---|---|---|---|---|
  | 8192@0 | 1368 W | 1507 / 1570 / 1425 | 88% | 13% |
  | 8192@131k | 1352 W | 1657 / 1681 / 1462 | 92% | 9% |
  | 10@131k | 1323 W | 1740 / 1820 / 1477 | 57% | 42% |
  | 354@131k | 1337 W | 1687 / 1750 / 1462 | 77% | 23% |

- One kernel alone, 5 s loops: a cuBLAS bf16 GEMM 8192×12288×7168 is capped by itself (1353 W,
  1357 MHz, 889 us). DRAM-bound glue is not: K1d 1119 W, K1b 1132 W (153.6 us, 2070 MHz: 0.17 J a
  call), finalize 965 W, a 235 MB D2D copy 964 W (6.4 TB/s). Idle with the clocks up: 231 W. So a
  byte through DRAM costs ~(964 − 231) W / 6.4 TB/s ≈ 115 pJ (HBM + L2 + the copy's issue).
- The controller is fast enough that a low-power gap pays the next GEMM back (`pw mix`, median of
  300 GEMMs): back-to-back 851 us; after 129 us idle 773; after ≥0.3 ms idle 730-740; after a
  152 us copy 799; after one K1d (190 us) 793; after 4 K1d (660 us) 748. A filler's effective
  cost (its time minus the GEMM time it gives back): idle 0.40 of its time, copy 0.66, K1d 0.69.
  So under the cap 1 us less of DRAM-bound glue is worth ~0.7 us of step, and an energy-equal
  speedup is worth only its static share. p-kda's -0.6 ms gather is then ~-0.4 ms: inside the
  stage A/B's spread.
- Lost to throttling at 8192 rows: GEMM-type kernels are ~120 of the item's 145 ms in-program
  (gemm_bf16 65, fc1 35, fc2 16, gemm_f32 4) and run ~14% slower capped than at the max clock
  (851 vs 730 us for the probe GEMM): ~17 ms an 8192@0 item (~12%), all of it GEMM time.
- DRAM per 8192@0 pass (ncu, 189 kernels): 316 GB read + 54 GB written. GEMMs and attention move
  most of it (fc1 111 GB, fc2 66, nvjet 118, kda/flash-kda/gather 40). This lane's glue moves 31.6 GB
  (8.5%):

  | kernel | calls | read GB | write GB | in-program ms (dcf6ec0) |
  |---|---|---|---|---|
  | moe_finalize_rms | 11 | 10.35 | 0.57 | 2.02 |
  | land_add_attnres_rms | 12 | 4.11 | 2.41 | 2.01 |
  | land_add2_attnres_rms | 11 | 3.88 | 2.25 | 1.86 |
  | moe_route (top-k, mxfp8, tables, situ) | 11 | 3.21 | 1.13 | 1.40 |
  | mla_gate (bf16 gate since cb84767: was 1.81 + 0.54) | 3 | 1.21 | 0.50 | 1.01 |
  | situ (layer 0) | 1 | 1.11 | 0.52 | 0.36 |
  | embed_rms, land_add2, mla_prep_gather | 5 | 0.34 | 0.37 | 0.27 |

  Energy: the item is ~187 J (145 ms × 1292 W mean). The glue's bytes are ~3.6 J (2%); the glue
  whole is ~9 ms × ~1.05 kW ≈ 9.5 J (5%), worth ~6.5 ms of capped step.
- Bytes the glue moves that it need not: none worth a commit.
  - Every glue read and write is its contract minimum. finalize reads the FC2 output once (16
    picks × 3584 bf16 a row, 0.94 GB); the residuals read partial, prefix and the nb blocks once
    and write prefix2 and normed once; route reads S f32 (29 MB, routing precision) and the normed
    row once.
  - The one write-then-read pair a fusion could still remove is o_proj's partial (the GEMM adding
    prefix as C, beta 1). That saves one 117 MB write a call (~1.4 GB, ~0.16 J, ~0.1 ms an
    item), and it is a GEMM ABI change in the runtime: out of scope.
  - The f32 intermediate in the MLA glue (wfu's gate columns) went bf16 with d-attn's cb84767
    (gate −0.6 GB read a pass).
- Sized, not done (each below the stage A/B's ~0.5% spread):
  - `discard.global.L2` on the glue's dead inputs. These are route's front-GEMM rows (latent +
    gate/up, 260 MB a call), the residuals' partials (117 MB) and the FC2 output's tail. Only the
    lines still dirty in L2 are saved, ~1-2 GB of write-back a pass, ~0.1 ms. p-kda's kda_rec
    discard (de539b9, 1.7 GB a pass) measured flat.
  - The residuals' second read of each block candidate (score, then mix) comes from L2, not DRAM:
    ncu L2 1.2 GB against DRAM 0.54 GB a call. Keeping the candidates in registers spilled before;
    it would be worth ~15 mJ a call, ~0.25 ms an item.
  - Route at 10 rows is 9.1 us back to back (CUDA graph, harness work/hb2; empty launch 0.7). The
    bench's 20 us is the per-op bracketing. All of it is ~0.1 ms of a 7.6 ms item.
- Where an 8192-row item's energy goes: GEMMs and their weight and activation streaming. The glue's
  ~5% is near its byte floor; the throttling (~12% of the item) is the GEMMs' own power.
- Threshold 256 rows with 8 splits vs main's 128 with 16, same-lease `loop/p/ab` on stage 0 (main
  aa144e0): A 124.52 / 124.57, B 124.78 / 124.44 ms/item; the 10-row item (the only one it
  changes: 354 stays long) 7.595 → 7.605 ms (+0.13%). No gain: not landed.

## For the orchestrator (measurement notes)

- Stage 1's bench gets zero `fmha_lens` tables (cut input without a fill: the FMHA attends nothing)
  and zero `hidden_in` (every row routes to the same 16 experts). With per-stage tables
  (work/lensx) stage 1 costs 97.8 ms/item instead of 52.5.
- Process lesson: after a cherry-pick, `loop/build` before committing (a cherry-picked
  kernels.toml carried a stale k3_moe_route pin into 5faa214; the orchestrator repinned it).
- Robustness: `moe_route`'s grid barrier needs its 304 blocks (two an SM, the whole register
  file) co-resident; a kernel beside it on another stream only delays it unless that kernel waits
  on this stream.
- Absorbed MLA for short chunks (merged): 10 rows over 131k 21.8 -> 7.6 ms. Threshold 128: the
  measured crossover is ~250 rows (200 rows −4.2%, 354 rows +9%); 256 would double the decode
  kernel's split workspace (rows × 16 × 256 KB) for shapes the bench does not weigh. The split
  merge in the gate kernel is the DSL reduction's own arithmetic (bit-identical logits).
  Option measured, not committed (bench-neutral): threshold 256 with 8 splits (same 512 MB
  workspace): A/B 4 rows over 131k +3.7% (fewer splits), 10 rows -0.06%, 200 rows -4.8%; the
  bench has no shape between 10 and 354. On stage 0 after d-attn's FMHA split it is flat (see
  "Power cap"): not landed.
- Proposal not done (a GEMM epilogue): sh_down accumulating onto lat_up's output is moot now that
  the back GEMM sums them by K.

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

## GEMM merges, offered again (322c829, 5faa214) after the absorbed MLA was taken

Ported onto the stage-0 era: stage 0 231 -> 209 launches (17.4/layer), A/B in one lease -0.12 /
-0.58 / -0.22% (8192@0 / 10@131k / 354@131k); KL <= 2.28e-3 against the branch before them, 4
near-tie flips. The measurements below are the first round's (stage 1).

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
- PDL on P's batched GEMMs (as d-mlp's e29458a does for D), with and without the route kernel
  triggering its dependents after its barrier: A/B in one lease 8192@0 +0.19 / +0.69%, 10 rows
  -0.10 / -0.09%, 354 rows -0.04 / +0.14%: nothing measurable.
- PDL (`pdl: true` + griddepcontrol.wait at the top) on every glue launch: 10 rows −0.5%, but 8192
  rows +0.9% (the early-resident blocks slow the GEMMs' tails); weighted +0.7%.
- Route phase 2 with the first row's places computed before the scan + a tight barrier spin:
  −0.7 us at 354 rows in the harness, no change in the bench.
- Residual with packed FFMA2 mix + FADD2 butterflies on top of the 40-register form: spills, slower.

### Mid-size chunks (129 .. ~1000 rows): what is left is the MLA attention
- At 354 rows over 131k (stage 0, 47.4 ms an item) the expand (2.6 ms) and FMHA (4.5 ms) take 21 ms
  of the item; the FMHA there streams each head's 84 MB of expanded K/V once per 256-row Q tile on
  192 CTAs (1.26 waves). The absorbed decode kernel is worse (+9%: heads padded to 128, every row its
  own pass). An absorbed prefill kernel with rows × heads in M would need ~9.7 TFLOP (6.5 ms at
  1.5 PF) but only as a tcgen05 kernel: mma.sync peaks at 605 TFLOP/s here (work/mma), which would
  make it slower than today's path.

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
