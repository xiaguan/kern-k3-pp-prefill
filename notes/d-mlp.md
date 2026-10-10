# d-mlp: the MLP half of a D layer and the TP all-reduces around it

Check: `loop/d/check loop/out/d-l4-tp8.json` (~20 s). Bench: `loop/d/bench loop/out/d-l16-tp8.json` (~20 s when
the pool is free). main 2026-10-10: 24.7 launches/layer, 4.562 ms/step.

## Per-layer MLP half on main (DCP decode)

`tp_allreduce_bf16` → `land_add_attnres_rms_bf16` → front* → `router_topk_front` → `land_n3584` → `moe_quant`
→ `moe_routing` (3) → fc1* → fc2* → `moe_finalize` → `situ_front` → sh_down* → `tp_allreduce_bf16` → `rms`
→ lat_up* → `land_add2` → next `attnres_rms`

## Log

| change | launches/layer | cost ms/step | numerics |
|---|---|---|---|
| main | 24.7 | 4.562 | |
| closing `land_add2` + next `attnres_rms` → `kern_k3_land_add2_attnres_rms` (1024-thread row, snapshot layers too) | 23.7 | 4.558 | bit-identical |
| `router_topk_front` + `land_n3584` + `moe_quant` + `moe_routing` (3) + `situ_front` → `kern_k3_moe_front` (row per block; warp 0 picks while warps 1-20 quantise + situ; the last block builds the tables, deterministic order) | 18.1 | 4.394 | bit-identical |
