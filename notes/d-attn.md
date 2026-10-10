# d-attn: the attention half of a D layer (KDA, MLA, DCP exchange)

The D lane closed at 21:04 UTC (sections from "State" on). The orchestrator
then moved this run to P's MLA attention (`mla_fmha`): see "P: MLA attention"
at the end.

## State

| | launches/layer (d-l16-tp8) | D weighted cost |
|---|---|---|
| main (2026-10-10) | 24.7 | 4.562 ms/step |
| after kda conv fold | 23.9 | 4.556 ms/step |
| after dcp exchange kernel | 23.2 | 4.409 ms/step |
| exchange merges the DSL splits | 22.9 | 4.400 ms/step |
| exchange runs v-up + gate | 22.7 | 4.394 ms/step |
| exchange latency (loads in flight, grid 256) | 22.7 | 4.366 ms/step |
| main 1242531 (all of the above + d-mlp, p-*) | 13.2 | 4.175 ms/step |
| exchange latency, rebased | 13.2 | 4.135 ms/step |
| absorb: 384 blocks, W frags held | 13.2 | 4.133 ms/step |
| main 7d6ea4f + the above (measured with PDL, = without) | 13.2 | 4.144 ms/step |
| exchange: split loads at once, half-head merge items, gate in smem | 13.2 | 4.120 ms/step |
| launch_dependents at entry of kda_core / mla_prep / dcp_exchange | 13.2 | 4.101 ms/step |
| rebased on main 049ff65+ (without embedding/argmax) | 13.2 | 4.099 ms/step |
| MLA split slots 32 → 16 | 13.2 | 4.066 ms/step |
| main ad97493 (all of the above merged, with d-mlp's handshake + top-k) | — | 4.026 ms/step |
| main 0c3c812 (+ d-mlp's PDL triggers, embedding, argmax, residual work) | 13.0 | 3.930 ms/step |

## Status (2026-10-10 ~21:30 UTC): lane closed at its launch floor

Per layer the half launches: KDA `kda_core`; MLA `mla_prep`, `absorb`, the
DSL split kernel, `dcp_exchange` (+ `mla_split_plan` once a step). Every
neighbour pair is split by a cuBLAS GEMM or the prebuilt attention (see
"Launch floor"). Time: the DSL attention is ~55% of the half and sits at
~4.7 TB/s in a prebuilt cubin (no other build, no PDL); kda_core resists
every prologue reshuffle (register cliff at 128); the exchange is within
~2x of its NVLink floor at 48 rows. Open lever needing a runtime feature:
tiered split slots (see "MLA split slots"). D's other non-GEMM ops are
d-mlp's (MLP, all-reduces, embedding, argmax, head).

## Done

- **KDA: `conv_silu` folded into `kda_core`** (`-DKDA_CONV=1`): thread d of the
  (row, head) block runs the conv + SiLU of column base + d for q, k, v and
  shifts the window in place; conv_q/k/v buffers gone. Same f32 sums and bf16
  landings, so bit-identical (check: bit-identical vs main). Cost 4.556 vs
  4.562 (noise). 128 regs vs 160 before, no spills.

- **MLA DCP: fixup + pack + NCCL all-to-all + combine → `dcp_exchange`**
  (source/k3_dcp.cu): one launch, grid 152 x 256, all blocks resident. Each
  block pushes its (row, head) records (64 o vectors + one lse vector, lse
  one byte per word so it can never look like the 0x8000 poison) into the
  owner's Lamport stage (`dcp_lamport`, own 3-stage rotation and
  `dcp_state`), re-poisons its share of the previous stage, then merges its
  (row, local head) records exactly as combine did. Bit-identical. 4.556 →
  4.409 ms/step: the NCCL SendRecv alone was 23 us + a 3 us gap at 48 rows.
  The receiver waits for every vector even when its weight is 0, so no
  write of call c can land after the stage's re-poison in call c + 1.

- **DSL split reduction folded into the exchange's send side**: the MLA op
  keeps only the split kernel (`reduce=False`); the sender merges acc_o /
  acc_lse as the DSL reduction does (S = ceil(t / ceil(t / bsk)), ex2/lg2
  approx, fma in split order, lse / log2 e). Bit-identical even at 4 rows
  (several splits). 4.409 → 4.400. Needs `__launch_bounds__(256, 1)`:
  with plain (256) ptxas squeezes it to 40 regs and spills.

- **`mla_vup_gate` folded into the exchange's receive side**: a merge item
  is (local head j, 4 rows); the rows' merged o land in bf16 in smem, then
  a warp per dv (32 lanes x 16 columns, butterfly sum) does W_UV · o and the
  gate. W_UV of the head is staged by cp.async in 4 quarters through two
  32 KB buffers (64 KB dyn smem), so it streams while the block waits on
  its peers. 7 near-tie flips vs main, relRMS max 0.6% (sum order of the
  512-long dot). 4.400 → 4.394.
  Pitfalls hit: (1) the lat slice is loop-invariant over dv, so a layout
  with 8 lanes x 64 columns made ptxas hoist 128 regs of it and spill;
  (2) a 128 KB staging buffer (1 block/SM, 152 blocks = every SM) PASSed a
  one-GPU 8-virtual-rank harness (work/t/xchg8.cu) bit for bit but broke its
  one real TP8 check (garbage, 10x slower steps) — most likely the startup
  init race found later, not residency; the 64 KB form was kept anyway.
  (3) `cmd | tail` hides a FAIL exit status: chain on the check's own rc.

- **Exchange latency**: at 48 rows @128k it was 29.6 us (nsys). Its send
  loop was a chain of dependent loads per record (lse, then acc_o) and the
  receiver polled its 8 peers' lse, then their o, one vector at a time (16
  round trips per item). Now split 0's acc_o is loaded with its lse, the
  receiver loads all 8 peers' vectors at once and re-polls only the late
  ones, and the grid is 256 (2 blocks/SM fit: 120 regs, 68 KB smem; 304
  would be exactly full). Same numerics as before. 4.394 → 4.366.

- **Absorb**: 768 blocks but 5/SM fit (regs + 40 KB smem) → 8 blocks in a
  second wave; now 384 x 256 threads, W fragments in registers across row
  tiles, rope copy spread over every block. 48 rows 11.9 → 9.9 us (one GPU,
  work/t/absorb.cu). Bit-identical.
- **Lamport deadline on clock64()** (the exchange counts 2 * timeout_ns SM
  cycles). Suspected %globaltimer jumps turned out a red herring: the step-0
  timeouts were the startup init race (below), which is also the likely
  cause of the 128 KB-staging exchange failing its one real check.

- **Exchange, round 2** (instrumented with %globaltimer printf from a
  debug cubin swapped into a copied manifest, work/t/k3_dcp_dbg.cu): at 16
  rows the send took ~5 us (2 records a block, each a chain of dependent
  split loads: lse, max, sum, then o one split at a time), the first
  peer data arrived at ~8 us, and the merge + v-up took ~8 us more (16
  serial dv passes, a global gate load per pass, two mid-item waits on W
  quarters). Now the first 8 splits' lse load at once and o loads 4
  splits at a time, a merge item is half a head (8 passes, its 64 KB of
  W_UV staged before the send), the item's gate rows sit in smem. 4.144 →
  4.120. Bit-identical.

- **`griddepcontrol.launch_dependents` at entry** of the kernels followed
  by a cuBLAS GEMM (kda_core → o_proj, mla_prep → q_b, dcp_exchange →
  o_proj): cuBLASLt launches its GEMMs as programmatic dependents and waits
  inside, so the GEMM's blocks land while ours run (d-mlp's find). The
  exchange's trigger fires only once every block ran it, so all its
  blocks are resident before a GEMM block takes an SM. 4.120 → 4.101.
  (absorb is followed by the DSL kernel, which has no PDL: no trigger.)

- **Embedding / argmax (dropped from this branch: d-mlp owns the head)**:
  I had vectorized the embedding gather (P kern test 37.1 → 13.4 us a
  call) and made the decode argmax one launch (per-row atomicMax of the
  same 64-bit keys into a zeroed carry, last block writes the token;
  16 rows: 5.8 us vs partial 7.4 + final 2.6; next tokens identical to
  main over 640 steps); together D 4.101 → 4.080. d-mlp landed its own
  versions first; numbers sent to d-mlp. Kept in git history only
  (2e4d9fc, 9a9a427 on the old branch).

- **MLA split slots 32 → 16** (`--mla-split-max` default): the DSL grid
  is (2, rows, split_max) clusters and every slot past a row's planned
  splits is a cluster that launches and exits; at 24-48 rows that churn
  cost 12-20 us per MLA layer. Bench (rebased branch, 4.099 at 32): cap 16
  4.066, cap 8 4.048; `when` tiers (≤7: 32, 8-15: 16, 16-31: 8, ≥32: 4)
  would be ~4.04 but loop/count counts every `when` launch (13.2 → 14.9 a
  layer for zero extra launches run), and kern's expressions have no `min`
  to shrink the slots with the batch in one launch. 16 keeps ≥2x headroom
  over a uniform 8-row batch's need (and lets a long row among short ones
  spread 16 ways), halves acc_o (537 → 268 MB). Below 8 rows a lone long
  row now spreads over 16 clusters instead of up to 32 (not in the bench).
  **Runtime proposal**: `min`/`max` in launch expressions (grid z =
  min(32, ceil_div(128, tokens))) or not counting mutually exclusive
  `when` launches, to tier this properly.

## Resolved: intermittent check failure (orchestrator, 20:00 UTC)

Fixed by d-mlp's first-call Lamport handshake (main d116f05); main
ad97493 checked 3/3 bit-identical with the exchange on top. The analysis:

main + my 1a53ac8 56d8cf6 05d3ad6 (exchange loads-in-flight, absorb, clock64
deadline) failed 1 of 3 d/checks with ~2-3 s of extra step time (one
Lamport deadline). Analysis so far:
- No concurrent kernels during a step (k3_step: one stream, no NCCL per
  step), and the exchange has no PDL wait, so it starts on drained SMs;
  even without full residency it cannot deadlock (every block sends before
  it waits, non-merging blocks exit).
- Stage reuse: each call re-poisons exactly the previous call's footprint
  (NRANKS * R * HL * REC vectors, R of that call); a member can be at most
  one call ahead of another, and the one-ahead member writes the stage the
  other is not clearing. Partial 16-byte arrivals: every 16-bit half is
  checked.
- **Startup race (not mine, affects the all-reduces):** each rank poisons
  its Lamport stages in the once-program right after the handle exchange,
  and step/'s handles() polls the files every 200 ms with no barrier
  after; a rank that finishes polling early can write its layer-0
  all-reduce partial into a peer that has not run its init yet, which then
  poisons it and waits until the deadline. The exchange is behind three
  all-reduces at layer 3, so its first write cannot precede a peer's init.
- main's k3_ar_fused.cu still times out on %globaltimer.
Diagnostic tree (work/out-dbg, not committed): every Lamport timeout
prints kernel, block, missing member and the elapsed cycles; work/rep.sh
runs the check N times. **Result, 6 checks of main 9b1cce0 + this branch:
4 PASS, 2 FAIL (9.5 and 7.0 ms/step). Every timeout was in k3_ar_fused
(run 1: ranks 0-6 x 3584 waiting on rank 7, rank 7 none; run 6: rank 2 x
3584 waiting on rank 0); zero dcp_exchange timeouts.** That is the startup
race above: the early rank's first all-reduce push is poisoned by its
peers' late init. d-mlp is adding a one-time ready handshake to its
all-reduce (and a trap on its deadline); it covers dcp_lamport too, since
tp_init poisons both before a rank's first all-reduce and the first
exchange sits behind three of them. The exchange now traps on its
deadline as well (a member's data never came: fail loudly, not garbage).

## Tried and dropped

- **MLA split plan for a full GPU** (work/k3_mla_split_plan.cu): at 39-48
  rows kern's plan gives every row one split (48 of 76 clusters busy); a
  makespan search picked 3 splits (2 waves of 43 tiles). The DSL split
  kernel stayed at 190 us (48 rows @128k, 906 MB of latent per rank: ~4.8
  TB/s either way, it is at its bandwidth), the exchange grew with the
  splits and the plan kernel itself took 14 us. 4.394 → 4.412: reverted.
  "Fewer, longer splits" in kern's comment holds.
- **kda_core: L2 bulk prefetch of the head's state at entry** (so the
  prologue overlaps the HBM read): +30 regs (158, 3 blocks/SM); with
  ROWS_PER_ITER=2 87 regs. Timed on one GPU (work/t/kdat.cu, state over 4
  rotating line sets): 48 rows 16.2 us now vs 18.6 / 17.7 / 17.4 for the
  variants; 24 rows 8.8 vs 8.0 best. Not worth it.

- **PDL on kda_core / mla_prep / absorb** (the kernels right behind a cuBLAS
  GEMM; `griddepcontrol.wait` at entry, absorb's W_UK loads before it;
  work/pdl.diff): bit-identical, but every shape within ±0.02 ms of the
  build without it (4.144 vs 4.133, main moved in between). Dropped. The
  DSL attention cubin has no ACQBULK/PREEXIT, so the exchange behind it
  could only gain the launch latency.
- **Launch floor of the half**: KDA `kda_core`; MLA `mla_prep`, `absorb`,
  the DSL split kernel, `dcp_exchange` (+ `mla_split_plan` once a step).
  Every pair left is split by a cuBLAS GEMM (wfu, q_b, o_proj) or the
  prebuilt attention. The plan could ride in the embedding launch as an
  extra block, but that is two unrelated loops in one grid: not done.

- **kda_core split in two 128-thread halves** (upper half streams its 64
  rows during the lower half's prologue; work/kda_split_v2.cu): one GPU,
  8/16/24 rows 6.09/7.57/8.81 → 5.98/7.42/8.52 us, 36/48 rows much worse
  (2 blocks/SM). Not worth a `when` split for 0.1-0.3 us.

- **bf16-output GEMMs for qkvg / wfu / q_b** (their readers land the f32
  partial to bf16 first, so cublasLt's bf16 epilogue would be the same
  rounding): one-GPU timing (work/t/gemm.cu, W over 4 copies) of cublas
  f32-out vs cublasLt bf16-out heuristics #0-3 at 8-48 rows: within ±0.5 us
  everywhere (qkvg 18.5-20 us, wfu 13, q_b 11). Same kernels, same split-K.
  Not pursued (and GEMMs are out of scope).
- **kda_core prologue loads up front** (w_f_b tile by cp.async into 32 KB
  smem, the scalars in registers): 150 regs, slower at every size. Dropped.

- **kda_core phase breakdown** (clock64 stamps, one GPU, 16 rows, cycles
  per block): conv 2000, l2norm 400, gate (wsm loads + w_f_b GEMV) 2400,
  delta rule 4000-5300, epilogue 1000: the prologue is half. Register-free
  prefetches (w_f_b tile + wsm row to L1 at entry, the state to L2 once the
  line is known) still moved ptxas to 156 regs and were slower at every
  size (16 rows 7.6 → 8.5 us, 48 rows 16.3 → 21.4). Every prologue
  reshuffle so far has cost registers past the 128 that keeps 4 blocks an
  SM; the kernel is left as it is. Forcing 128 regs with
  __launch_bounds__(128, 4): with the L2 state prefetch 32 B of spills and
  slower everywhere; L1-only prefetch 16 rows 7.6 → 7.9, 48 rows 16.2 → 16.8.

- **kda_core launched early (pdl) to pull its weights into L2 before
  griddepcontrol.wait** (they are cold in the real step: one GPU says 7.6
  us at 16 rows, the graph 10.3; work/kda_pdl_prefetch.diff). The prefetch
  had to live in a `__noinline__` helper to keep 128 regs (inline it
  pushed ptxas to 152: a trick worth remembering). Bit-identical, but D
  4.025 (main ad97493, same pool, same hour) → 4.050: the early blocks
  waiting on the SMs slow the GEMM ahead of them. Dropped. The same L2
  prefetch at entry without pdl (noinline helper, 128 regs): 4.047 vs
  4.025, no gain either.

- **DSL tensor maps with 256-byte L2 promotion** (was 128; a generator
  field, not the kernel): 3.928 vs main 3.930 on the same pool/hour, every
  shape within 0.01 ms. No effect.

- **kda_core knob sweep** at 8-48 rows (one GPU; ROWS_PER_ITER 4/2,
  KDA_STREAM 1/0, KDA_SWIZZLE 1/0, all math-neutral): the shipped 4/1/1 is
  best at 48 rows (16.3 us); swizzle off is better at 8-36 (24 rows 8.96 →
  8.53, 36 rows 13.42 → 12.99) and much worse at 48 (18.9). A runtime
  branch on gridDim.x would save ~5 us a step at 24-36 rows (~0.1%, below
  the bench's resolution): not done.

- **kda_core prefetching o_proj's 22 MB weight into L2** (bulk prefetch,
  each block its share, noinline helper, 128 regs): o_proj 7.0 → 5.8 us but
  kda_core 11.1 → 12.6 us at 16 rows; D 3.942 vs main 3.930. The same HBM
  bytes, moved from one kernel to the other. Dropped.

## Where the half stands (16 rows @128k, nsys, after the commits above)

Per 16-layer step: DSL split kernel 254 us (prebuilt, ~4.7 TB/s, the only
build in the index), kda_core 123 (12 x 10.3), dcp_exchange 55 (4 x 13.9),
absorb 27, mla_prep 12, split plan 2. Exchange breakdown (debug cubin,
work/t/k3_dcp_dbg2.cu): 16 rows send 1.3 us per item, first peer data at
~7 us, merge done ~11-13 us; 48 rows send ~8 us (4.8 MB a rank over
NVLink is ~5 us by itself), merge items 288 > 256 blocks so 32 blocks take
two. The GEMMs around (qkvg + splitK reduce 20 us, wfu 14, q_b 12, o_proj
7) are cuBLAS's.

- **Exchange grid 304** (2 blocks on every SM, so 48 rows' 288 merge items
  run in one round; at 256 32 blocks take two, exchange 23 us at 48 rows):
  48-row steps -0.02-0.03 ms, weighted 4.066 → 4.063 (noise). Not worth
  giving up the residency slack: kept 256.

## Profile (nsys, rank 0, 48 rows @128k, before the exchange kernel)

work/prof.sh runs the bench with nsys on the first rank (`--capture-range-end
stop`; the default stop-shutdown kills the rank and hangs the other 7).
MLA layer: mla_prep 4.7, q_b 12.8, absorb_mma 11.0, DSL split 187.7 (grid
2x48x32: the split plan gives 1 split per row at 48 rows, 48 of 76 clusters),
DSL reduce 9.2, fixup 1.1, pack 2.2, NCCL 23 (+2.8 gap), combine 2.8,
vup_gate 14.8. KDA layer: qkvg 18.5 + splitK reduce 4, kda_core 15.4.
Gaps between graph nodes are ~0.2 us: a launch costs its ramp, not a gap.

## Ideas / queue (for a next session)

1. Tiered MLA split slots (≤7 rows 32, 8-15 16, 16-31 8, ≥32 4: ~-0.6%
   more) need either `min` in kern's launch expressions or loop/count not
   counting mutually exclusive `when` launches; proposed, not done.
2. Everything else measured in this lane is in "Tried and dropped". The
   one-GPU harnesses in work/t/ (xchg8.cu exchange on 8 virtual ranks,
   kdat.cu kda_core, absorb.cu, gemm.cu) and work/prof.sh (nsys of one rank
   of the TP8 bench; `check` profiles the teacher-forced run) are the
   tools; a debug cubin can be swapped into a copied manifest by sha
   (cache blobs) to printf from inside a real TP8 run.

# P: MLA attention (`mla_fmha`), from 21:04 UTC

Bench `loop/p/bench loop/out/p-stage0.json` (on the P pool), check
`loop/p/check loop/out/p-l12.json`. The check's 2048-row chunks never take
the split path below; `work/p/check.sh` is the same check with
`PREFILL=3000 CHUNK=400` (chunks 3-6 split). `work/p/bench.sh` benches any
workload (`work/p/wl-waves.toml`), `work/p/ops.py` / `work/p/diff.py` read
per-op times out of a report.

## State

| | P stage 0 weighted (same host A/B) | launches/layer (p-l12, counted) |
|---|---|---|
| main 9850876 | 123.47 / 124.08 ms/item | 17.7 |
| fmha split for 129-512-row calls | 122.63 / 123.52 (-0.57%) | 18.0 (+1 exclusive `when` launch a MLA layer) |

## What the kernel is and what it reaches

- `fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvCausalVarSeqQ256Kv128PersistentContext`
  from FlashInfer 0.6.18's TRT-LLM gen bundle: an sm_103a tcgen05/TMEM
  kernel, 512 threads, 199 KB smem (one CTA an SM), Q tile 256 x KV tile
  128, grid (ceil(rows / 256), heads, seqs), one CTA a (q tile, head). The
  only build we have: kernelMetaInfo.h lists Static / SkipsSoftmax / Dense
  variants, but every cubin under $K3_REF/trtllm/.../fmha/cubin is a
  130-byte git-LFS pointer. No context variant splits KV over CTAs.
  FlashInfer's cake_fmha (generated CUDA source) is head_dim 128 only.
- It writes softmax stats when ptrSoftmaxStats (param byte 1112) is set:
  float2 (max of the scaled score in natural-log units, sum) per (token,
  head); lse2 = log2(e) * max + log2(sum) matches an f32 reference to 2e-6.
- FLOP rate (rows x (ctx + causal half) x 96 x 320 x 2 over its time): main's
  stage-0 bench 1.33 PF at 8192 @0, 1.50 PF at 8192 @64k-196k, 1.39 PF at
  4096 @64k, 0.63 PF at 354 @128k. cuBLAS reaches 2.0 PF on the stage's big
  GEMMs (kern's calibration 1.91). One GPU, harness: 1.61-1.65 PF at 8192
  rows, linear in rows (7680 / 7936 / 8192 rows: 57.96 / 60.82 / 62.81 ms,
  no wave step); sustained for 150 calls it is power-capped: 1.35-1.41 kW,
  SM clock 1.80-1.97 GHz, 63.5-64 ms. The stage bench reads 66-69.6 ms for
  the same call (more power from the rest of the stage, other GPUs).
- So at 8192 rows it is at the power cap, about 95% of the "1.6 PF
  effective" number. Splitting KV did not help there (harness: 8192 @196k
  -0.6..+0.9% with the merge; empty extra FMHA sequences cost +3%).
- Small calls are the loss: a 129-512-row call is 96 or 192 CTAs of one
  whole context each on 152 SMs (256 / 354 / 512 / 768 rows @128k: 2.29 /
  4.10 / 4.14 / 4.12 ms).

## Done

- **KV split of a short call's heaviest sequence** (5a65c15): three causal
  pieces, N keys [c + 1, P + Lq) as is, R keys c .. 1 reversed with its q
  rows reversed (reversing both turns the end-aligned mask into a moving
  start: row i sees [i + 1, c]), A keys [0, Lq) with a second q copy (row i
  sees [0, i]); c = ctx / 3 (a greedy list-scheduling model says 1/3 for
  1 and 2 q tiles; harness optimum c ~ 0.3 ctx at 354 rows: 4.10 -> 3.00
  ms). Any such split needs Lq - 1 duplicated keys and two extra q row
  ranges (cum_q / cum_kv are spans: the mask is aligned to the cum_kv
  span, so spans cannot overlap or have gaps). The lens kernel plans it,
  the prep gathers in piece order and writes the extra q_norm rows (q_b
  runs T + 1024 rows), the gate merges with the stats. `when` routes the
  split launch to 129-512 rows.
  Bench FMHA at 354 @128k 4.47 -> 3.24 ms a layer. Overheads: q_b +27-33
  us a call at every size (fixed +1024 rows: kern's expressions cannot
  shrink it for big calls), expand +130 us at 354 rows.

## Ideas / not done

- Expansion without the k_pe identity block (w_aug is [30720, 576]: 6144
  output columns are k_pe copies, K 576 vs 512 for the rest: ~30% of the
  expand's FLOPs, the expand is ~4.4% of the weighted P score). Needs a
  per-head strided (batched) GEMM to leave the 64 k_pe columns of each
  320-column head alone; kern's cublaslt extern has row strides but no
  batch. Runtime proposal: a batched/strided-C GEMM extern.
- Packed multi-sequence calls run grid z = seqs with x = ceil(tokens / 256)
  each, so most CTAs are empty; the harness shows empty FMHA sequences cost
  3-5% at long contexts. The bench has one sequence a call; real traffic
  packs up to 16. Worth a look by whoever owns the packing (grid x per
  sequence is the kernel's, not ours).
