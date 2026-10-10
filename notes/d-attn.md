# d-attn: the attention half of a D layer (KDA, MLA, DCP exchange)

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
  one-GPU 8-virtual-rank harness (work/t/xchg8.cu) bit for bit but broke the
  real TP8 check (garbage, 10x slower steps): presumably not all 152 blocks
  were resident at once -> Lamport waits time out. Keep this kernel at
  >= 2 blocks/SM worth of resources.
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
- **Lamport deadline on clock64()**: d-mlp found %globaltimer jumps early in
  a run (resync to host time) and fires the 2 s timeout with nothing late
  (tp_err = 1 + rank at step 0, ranks diverge). That is the likely cause of
  my 128 KB-staging exchange failing the real check while passing the
  8-virtual-rank harness. The exchange now counts 2 * timeout_ns SM cycles.

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
ad97493 checked 3/3 bit-identical with the exchange on top.


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

## Ideas / queue

1. MLA DCP: fixup + pack + nccl all-to-all + combine → one peer-memory
   Lamport kernel; then fold the DSL's split reduction into its send side,
   and maybe the vup_gate into its receive side.
2. MLA prep: `mla_prep` writes q_norm for the q_b GEMM, so it cannot merge
   with `mla_absorb` (the GEMM sits between).
