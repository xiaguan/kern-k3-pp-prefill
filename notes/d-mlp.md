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
| first Lamport call handshakes with every peer (init race, below); a passed deadline (30 s) traps | 13.2 | 4.177 | bit-identical, 3/3 on main, 12/12 on the series |
| moe_front top-k in two levels (8 warps x 112 experts, then warp 0 over the 128 candidates); 10.5 -> 6.5 us | 13.2 | 4.122 | bit-identical |
| the residual rows (ar_attnres_rms, land_add2_attnres_rms) cp.async their snapshot rows into smem up front, sw/gamma in registers; the all-reduce scores the snapshots before polling | 13.2 | 4.098 | bit-identical |
| layer 0's dense down partial summed by layer 1's mix: `tp_allreduce_bf16` + `land_add2_attnres_rms` → `ar_attnres_rms` (prefix = prefix2, out = hidden) | 13.1 | 4.093 | bit-identical |
| my kernels execute `griddepcontrol.launch_dependents` at their start: the next cuBLAS GEMM (launched with programmatic serialization) gets its CTAs on SMs while the kernel finishes | 13.1 | 4.061 | bit-identical |
| embedding copies 16-byte vectors, 4 in flight a thread (was one bf16 a thread an iteration: 15 us for 8 rows) | 13.1 | 4.050 | bit-identical (D and P check) |
| head argmax in one launch (`kern_k3_argmax_f32_fused`: 16 busy 256-thread blocks a row, the row's last block takes the max) | 13.1 | 4.041 | bit-identical |
| embedding gather + layer 0's `attnres_rms_first` → `kern_k3_embed_rms` (nb = 0: snapshot + rms of the gathered row) | 13.0 | 4.040 | bit-identical |

## Findings

- **Step-0 deadline misses were an init race.** The Lamport kernels' 2 s deadline fired at step 0 in some
  4-layer checks (tp_err = 1 + a rank; seen with `--dump` of a copy of the manifest whose `tp_err` is
  [tokens]-shaped with no fill), then a rank summed a missing slot and the ranks diverged for good. I first
  blamed %globaltimer jumping and moved the deadline to clock64 (wrong; dropped from the branch). With the
  deadline at 2000 s, 2 of 3 runs HUNG after load: data lost, not late. Cause (found by d-attn, confirmed by
  their debug build: ranks 0-6 waiting on rank 7's AR partial): the once-program poisons the stages right
  after step/'s 200 ms-polled handle exchange, with no barrier, so a fast rank's first push can land in a
  peer's stage before that peer's init poisons it. Fix (first commit of the branch, on main): the first
  Lamport call of a process handshakes (block 0 sets ready[rank] in every peer's words past the three
  stages, every block waits for all of its own; state[6] marks it done) and a passed deadline (30 s) traps
  instead of summing a missing slot. It also covers d-attn's dcp_lamport: a peer's ready flag is written
  after its whole once-program, and layer 0's all-reduce precedes the first exchange. Seen since: runs
  that wait 5-7 s at step 0 (start skew) and pass.
- withgpu -H 2 used to livelock two waiters (fixed in main 049ff65). Keep D runs to <= 3 in a row with a
  60 s pause before the next lease (orchestrator, 20:36).
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
- **What bounds the fused all-reduces** (stamped debug build: `work/h/mkstamps.py` adds a stamps param,
  `--dump stamps` of a [tokens, 64] i64 output): at 4 rows the attention AR is ~1.4 us before the first
  push, 2.4 us of pushing (row blocks only: 1024 vectors x 8 peers through one SM), 0.6 clear, 0.7
  snapshots, ~2.3 us polling, ~2.3 us epilogue (including other threads' late vectors). Across the 4 ranks
  of one host the AR ends ~10-11 us after the last rank arrives; ar_finalize additionally waits a median
  ~22 us at 24 rows on the rank with the most active experts (MoE weight bandwidth; inherent to the EP
  layout). Tried, no gain, reverted: pushes spread over all 152 blocks in 8-lane groups (4.061 -> 4.124:
  the combine's 16-gather chain repeats per lane group, stores lose 512 B coalescing); contiguous chunk per
  block staged in smem (4.104: pushes finish earlier, arrival does not); push before the state barrier and
  the prefetch (4.059, equal); AR grid 64 / 96 instead of 152 (equal). The end is set by arrival (peer
  skew + NVLink latency), not by the push.
- Two-shot all-reduce for 36-48 rows estimated at ~0.4% weighted (one-shot pushes 7x the partial; only the
  large row counts are bandwidth-bound). Not done.
- Per-step leftovers: lm_head GEMM 370 us (replicated on every rank, out of scope), mla_split_plan (d-attn).
