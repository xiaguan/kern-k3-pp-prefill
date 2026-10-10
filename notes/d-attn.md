# d-attn: the attention half of a D layer (KDA, MLA, DCP exchange)

## State

| | launches/layer (d-l16-tp8) | D weighted cost |
|---|---|---|
| main (2026-10-10) | 24.7 | 4.562 ms/step |
| after kda conv fold | 23.9 | 4.556 ms/step |

## Done

- **KDA: `conv_silu` folded into `kda_core`** (`-DKDA_CONV=1`): thread d of the
  (row, head) block runs the conv + SiLU of column base + d for q, k, v and
  shifts the window in place; conv_q/k/v buffers gone. Same f32 sums and bf16
  landings, so bit-identical (check: bit-identical vs main). Cost 4.556 vs
  4.562 (noise). 128 regs vs 160 before, no spills.

## Ideas / queue

1. MLA DCP: fixup + pack + nccl all-to-all + combine → one peer-memory
   Lamport kernel; then fold the DSL's split reduction into its send side,
   and maybe the vup_gate into its receive side.
2. MLA prep: `mla_prep` writes q_norm for the q_b GEMM, so it cannot merge
   with `mla_absorb` (the GEMM sits between).
