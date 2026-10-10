# d-attn: the attention half of a D layer (KDA, MLA, DCP exchange)

## State

| | launches/layer (d-l16-tp8) | D weighted cost |
|---|---|---|
| main (2026-10-10) | 24.7 | 4.562 ms/step |
| after kda conv fold | 23.9 | 4.556 ms/step |
| after dcp exchange kernel | 23.2 | 4.409 ms/step |
| exchange merges the DSL splits | 22.9 | 4.400 ms/step |

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
