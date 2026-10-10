Run `p-kda` · **target: the KDA half of a P layer.** Notes go in `notes/p-kda.md`.

What it launches today, per KDA layer (9 of a stage's 12; `loop/count`, `loop/out/p-stage1.json`):

`land_add2_attnres_rms` → qkvg* → wsm* → `span_gather` (2 launches) → `span_state_load` → span_g* → `flash_kda` (3 launches: FlashKDA prepare / recurrence / tile prefix, varlen) → `span_state_store` → `kda_out_gate` → o_proj*

(* = GEMM, stays.) Shares of the stage's weighted cost: flash_kda 7.6%, span_gather 6.5%, kda_out_gate 1.2%, state load/store 0.2%.

Directions:

1. NVlabs KDA² (`$K3_REF/nvkda`, branch with kda-cake-cute / kda-tirx / kda-cake-ptx): reported 2.5-3.6x FlashKDA at 8192 tokens on B300, fp32 V-first state, the K3 gate formula, ABI `run(q,k,v,g,beta,A_log,dt_bias,scale,initial_state,cu_seqlens)`. `kda-cake-ptx` is static PTX for sm_103a, the closest to kern's cubin model. Port it as a kern op, fold the state load/store (and, if the layout allows, the span gather / conv) into it, and keep the varlen packing. Check its state layout against kern's (`kda.<l>` state, line index).
2. `span_gather` (6.5%) is memory traffic: fuse it into the producer or the consumer.
3. `kda_out_gate` into the recurrence's epilogue.

Check: `loop/p/check loop/out/p-l12.json`. Bench: `loop/p/bench loop/out/p-stage0.json`. One GPU each, from your pool.
