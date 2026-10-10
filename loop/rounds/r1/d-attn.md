Run `d-attn` · **target: the attention half of a D layer, KDA and MLA, including the DCP exchange.** Notes go in `notes/d-attn.md`.

What it launches today, per layer (`loop/count`, `loop/out/d-l16-tp8.json`):

- KDA (12 of every 16 layers): `attnres_rms` → qkvg* → `conv_silu` → `kda_core` → o_proj* → (all-reduce: the `d-mlp` run's)
- MLA (every 4th): `attnres_rms` → wfu* → `mla_prep` → q_b* → `mla_absorb` → `mla_attn` (2 launches: split + merge) → `dcp_fixup` → `dcp_pack` → `nccl_alltoall_bf16` → `dcp_combine` → `mla_vup_gate` → o_proj*

(* = GEMM, stays.) Directions, ranked by launches saved:

1. MLA DCP: 7 non-GEMM launches between q_b and o_proj. Hand-write the exchange as a peer-memory kernel (the ranks' buffers are mapped like the Lamport all-reduce's, `peer` buffers in the manifest) that writes each rank's partials straight into the owner's slot, so pack + all-to-all + combine (+ fixup) become one or two launches; fold the attention's own split merge into it if the LSE math lines up.
2. KDA: `conv_silu` + `kda_core` in one kernel (the conv window is tiny at decode; vLLM / SGLang have fused conv + gating + delta-rule decode kernels, see `$K3_REF/insights.md`); fold `attnres_rms` into the qkvg prologue only if that does not mean replacing the GEMM.
3. MLA prep: `mla_prep` + `mla_absorb` (+ rope) into fewer launches.

Measured context (the full 93-layer D at 48 rows, nsys, 33.5 ms per step): MoE bmm 13.6 ms, other GEMMs 6.6, all-reduce 4.2, routing + finalize 1.9, attn-res 1.3, MLA small kernels 1.3. SGLang runs the same layout; its captured graphs are in `decode/sglang-tp8-dcp8-ep8/`.

Check: `loop/d/check loop/out/d-l4-tp8.json` (its 4 layers hold one MLA layer: layer 3). Bench: `loop/d/bench loop/out/d-l16-tp8.json`. Both need the whole pool (8 GPUs); another run (`d-mlp`) shares it, so batch your changes before measuring.
