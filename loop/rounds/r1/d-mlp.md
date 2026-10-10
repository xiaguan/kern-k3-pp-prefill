Run `d-mlp` · **target: the MLP half of a D layer and the TP all-reduces around it.** Notes go in `notes/d-mlp.md`.

What it launches today, per layer, after o_proj* (`loop/count`, `loop/out/d-l16-tp8.json`):

`tp_allreduce_bf16` → `land_add_attnres_rms_bf16` → front* → `router_topk` → `land_n3584` → `moe_quant` → `moe_routing` (3 launches, FlashInfer) → moe_fc1* → moe_fc2* → `moe_finalize` → `situ` (shared expert) → sh_down* → `tp_allreduce_bf16` → `rms` → lat_up* → `land_add2` → next layer's `attnres_rms`

(* = GEMM, stays.) About 12 non-GEMM launches. Directions:

1. All-reduce fusion: the all-reduce is ours (`source/peer_allreduce_bf16.cu`, TensorRT-LLM's Lamport one-shot protocol). Make it also add the residual and run the attn-res mix + RMSNorm that follows (TRT-LLM / FlashInfer `allreduce_fusion`), for both all-reduces of the layer: `tp_allreduce_bf16` + `land_add_attnres_rms_bf16`, and `moe_finalize` / `situ` / `tp_allreduce_bf16` / `rms` / `land_add2` / next `attnres_rms` as far as the data flow allows.
2. Routing: `router_topk` + `land_n3584` + `moe_quant` + the three routing launches into one or two (the routing tables for 112 local experts at ≤64 rows fit one CTA; TRT-LLM's routing kernels and SGLang's fused gates are in `$K3_REF`).
3. `moe_finalize` + shared-expert add.

Measured context (the full 93-layer D at 48 rows, nsys, 33.5 ms per step): MoE bmm 13.6 ms, other GEMMs 6.6, all-reduce 4.2, routing + finalize 1.9, attn-res 1.3. The all-reduce is ~186 launches per step: its latency, not its bytes, is the cost.

Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`. Both need the whole pool (8 GPUs); another run (`d-attn`) shares it, so batch your changes before measuring.
