# d-mlp: the MLP half of a D layer and the TP all-reduces around it

Check: `loop/d/check loop/out/d-l4-tp8.json` (~20 s on a free pool). Bench: `loop/d/bench loop/out/d-l16-tp8.json`
(~20 s). main 2026-10-10 start: 24.7 launches/layer, 4.562 ms/step.

## Per-layer MLP half now (DCP decode)

`ar_attnres_rms` (o_proj sum + landing + mix + norm) → front* → `moe_front` (top-k, latent mxfp8, shared situ,
routing tables) → fc1* → fc2* → sh_down* → `ar_finalize_rms` (combine + sum of both + latent norm) → lat_up*
→ `land_add2_attnres_rms` (closing add + next layer's mix + norm). 4 non-GEMM launches, all in
`source/k3_ar_fused.cu` / `source/k3_moe_front.cu`.

## Log

| change | launches/layer | cost ms/step | numerics |
|---|---|---|---|
| main (start) | 24.7 | 4.562 | |
| closing `land_add2` + next `attnres_rms` → `kern_k3_land_add2_attnres_rms` (1024-thread row, snapshot layers too) | 23.7 | 4.558 | bit-identical |
| `router_topk_front` + `land_n3584` + `moe_quant` + `moe_routing` (3) + `situ_front` → `kern_k3_moe_front` (row per block; warp 0 picks while other warps quantise + situ; the last block builds the tables, deterministic order) | 18.1 | 4.394 | bit-identical |
| `tp_allreduce_bf16` + `land_add_attnres_rms_bf16` → `kern_k3_ar_attnres_rms`; `moe_finalize` + `tp_allreduce_bf16` + `rms` → `kern_k3_ar_finalize_rms` (all 152 blocks push a grid-stride share, block b < B polls row b and runs the epilogue at 1024 threads) | 15.2 | 4.332 | bit-identical |
| (main after merges with d-attn: 13.2/layer, 4.175) | 13.2 | 4.175 | |
| Lamport hang detector on clock64 instead of %globaltimer (fix, below) | 13.2 | 4.174 | bit-identical, 3/3 runs |
| moe_front top-k in two levels (8 warps x 112 experts, then warp 0 over the 128 candidates); 10.5 -> 6.5 us | 13.2 | 4.122 | bit-identical |
| the residual rows (ar_attnres_rms, land_add2_attnres_rms) cp.async their snapshot rows into smem up front, sw/gamma in registers; the all-reduce scores the snapshots before polling | 13.2 | 4.098 | bit-identical |
| layer 0's dense down partial summed by layer 1's mix: `tp_allreduce_bf16` + `land_add2_attnres_rms` → `ar_attnres_rms` (prefix = prefix2, out = hidden) | 13.1 | 4.093 | bit-identical |
| my kernels execute `griddepcontrol.launch_dependents` at their start: the next cuBLAS GEMM (launched with programmatic serialization) gets its CTAs on SMs while the kernel finishes | 13.1 | 4.061 | bit-identical |

## Findings

- **%globaltimer trips spurious timeouts.** The Lamport kernels' 2 s timeout fired at step 0 in ~2/3 of the
  4-layer checks of the fused all-reduces (tp_err = 1 + rank 5 or 6; seen with `--dump` of a copy of the
  manifest whose `tp_err` is [tokens]-shaped and has no fill). Then a rank sums a slot that has not arrived
  and the ranks diverge for good ("ranks disagree", hundreds of flips). The same manifest with a 30 s
  timeout: 3/3 clean and no extra step time, so the wait was not real: the global timer is resynced to the
  host clock and jumps. All Lamport kernels now count clock64 cycles (2 per ns). d-attn moved k3_dcp.cu too.
- **withgpu -H 2 livelocks** when two 8-GPU jobs wait at once (each pass takes one host, fails the other,
  both release, retry in lockstep every 5 s). `work/lease8.sh` (not committed) retries single passes with a
  random 2-9 s pause; `work/dcheck.sh` / `work/dbench.sh` are loop/d/check / bench through it.
- **Launch gaps inside the graph are ~0.15 us**: fusing saves the kernel ramp and tail, not launch latency.
- **`pdl: true` on my kernels (griddepcontrol.wait at their start) is unsafe for K1d**: with it on
  land_add2_attnres_rms alone the check FAILs (430 flips, ranks agree); on the two AR kernels alone or
  moe_front alone it is bit-identical. Unexplained (its predecessor is lat_up's cuBLASLt GEMM); the whole
  gain was 4.061 -> 4.049, so dropped. d-attn saw no gain from it either. The trigger alone (committed) is safe.
- Profiling: nsys is on the hosts (`/usr/local/bin/nsys`), not in the container. `work/dprof.sh` runs rank 0
  under nsys with k3_step `--profile R@CTX` and `--cuda-graph-trace=node`; `work/kstats.py`, `work/kmin.py`,
  `work/ktimeline.py` read the export.
- Profile at 8 rows, per KDA layer ~150 us: GEMMs ~104 (qkvg 17+3 splitK, front 17+3 splitK, fc1 26, fc2 15,
  lat_up 10, sh_down 6.6, o_proj 6); ar_attnres_rms ~11 (min 10.4); ar_finalize_rms 9-22 (waits on the
  slowest rank's MoE: expert-load skew); moe_front 10.5; land_add2_attnres_rms 5.5. At 48 rows the ARs'
  medians are 16 / 33 us, fc1 84, fc2 42.
- moe_front at 10.5 us regardless of rows: ~6 us was the single warp's 16 serial rounds over 28 experts a
  lane, ~1.5 us the routing tables (single-GPU harness `work/h/mf_bench.cu`).
