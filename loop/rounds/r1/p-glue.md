Run `p-glue` · **target: the MoE glue and the MLA glue of a P layer.** Notes go in `notes/p-glue.md`.

What it launches today (`loop/count`, `loop/out/p-stage1.json`), every layer after o_proj*:

`land_add_attnres_rms_bf16` → router* → `router_topk` → lat_down* → `moe_quant` → `moe_routing` (3 launches) → moe_fc1* → moe_fc2* → `moe_finalize_rms` → lat_up* → wsh* → `situ` → sh_down* → next layer's `land_add2_attnres_rms`

and in an MLA layer (3 of a stage's 12): wfu* → `mla_prep` → q_b* → `latent_gather` → expand* → `mla_fmha` → `mla_gate` → o_proj*

(* = GEMM, stays.) Shares of the stage's weighted cost: land_add_attnres_rms_bf16 2.0%, moe_finalize_rms 2.0%, land_add2_attnres_rms 1.9%, router_topk 0.8%, situ 0.6%, moe_routing 0.3%, mla_gate 0.3%, mla_prep 0.3%, moe_quant 0.2%. The short items (10-700 rows, more than half the recorded P items) are where launch count shows most: a 10-row item takes 11.4 ms for 325 launches.

Directions:

1. Routing: `router_topk` + `moe_quant` + the three FlashInfer routing launches into fewer (TRT-LLM's routing kernels, SGLang's fused gates: `$K3_REF`).
2. `moe_finalize_rms` + `situ` / shared-expert combine + the next layer's `land_add2_attnres_rms`.
3. MLA: `mla_gate` into the FMHA output path or o_proj's input; `latent_gather` into `mla_prep` or the expand's input.

Check: `loop/p/check loop/out/p-l12.json` (layers 0-11: three MLA layers). Bench: `loop/p/bench loop/out/p-stage1.json`. One GPU each, from your pool.
