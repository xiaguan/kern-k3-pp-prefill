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
