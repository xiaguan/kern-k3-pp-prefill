# d-mlp: the MLP half of a D layer and the TP all-reduces around it

Check: `loop/d/check loop/out/d-l4-tp8.json` (~20 s on a free pool). Bench: `loop/d/bench loop/out/d-l16-tp8.json`
(~20 s). main 2026-10-10 start: 24.7 launches/layer, 4.562 ms/step.

## Status (2026-10-10 23:15 UTC)

D 16 layers: 24.7 -> 12.9 launches/layer, 4.562 -> 3.52 ms/step weighted (all runs' work; main dcf6ec0 3.581,
plus this branch's L2 prefetch in the all-reduces).
Real 93-layer step at 128k: 23.88 -> 19.54 ms (24 rows), 30.87 -> 26.42 ms (48 rows) before the prefetch.

## Roofline of the real D step (93 layers, 128k context, 2026-10-10 23:00 UTC)

nsys of rank 0 on main dcf6ec0 (`work/dprof93.sh`, `work/roofline.py`): every kernel aligned to its manifest
call and charged its critical-path time (its end minus the previous kernel's end, so a PDL overlap is not
counted twice); the shorter of two replays (one 48-row replay carried a 1.1 ms member start skew in layer 0's
first all-reduce: host side, excluded). Bytes and FLOPs are one member's.

Bounds, measured on this pool where it matters:
- HBM 8 TB/s. The best kernels here reach 6.2-7.0 TB/s (MoE fc1/fc2, lm_head).
- Tensor: ~1.6 PF bf16 and ~2.8 PF fp8 reachable on one GB300 (earlier calibration); the mxfp8 x mxfp4 MoE
  runs at the fp8 rate. bf16 weights reach the ridge at ~200 rows, so at 8-48 rows every GEMM is HBM-bound.
  MLA decode is the exception: 96 heads x 1088 MACs per 1152-byte KV row is 181 FLOP/B, at the ridge.
- NVLink: 16-byte pushes from every SM with Lamport polling (`work/h/nvl_push8.cu`: 8 GPUs, 2 trays, fabric
  handles as kern maps them): ~2.5 us a hop, ~680 GB/s marginal egress a GPU (raw push + poll of 2.4 MB a GPU,
  the 24-row attention sum: 8.4 us from the last start; 4.8 MB: 11.9 us). For comparison NCCL picks RING_LL
  here and takes 37 / 58 us an all-reduce at 24 / 48 rows (16-layer manifest without `--peer-ar`).
- glue: ~0 bytes; floor ~2 us a launch (0.7 us empty launch + one dependent row pass).

MoE bytes: active experts x 17.5 MB (fc1 11.7 MB, fc2 5.9 MB, mxfp4 + scales). The bench feeds pseudo-random
tokens with many repeats (181 / 275 distinct experts in the last layer at 24 / 48 rows: 22.6 / 34.4 a member,
used below). Real text routes wider: over 48 rows x 56 steps x 92 layers of teacher-forced text
(`work/route93.sh`, per-layer `topk_idx` dumped) 230 / 342 distinct experts, 28.8 / 42.7 a member, the busiest
member 35.4 / 49.9 (1.23x / 1.17x the mean). At the bound that is 7.1 / 10.1 ms of MoE a step on real text
(busiest member) against 4.6 / 6.9 ms for the bench's batch.

**24 rows** (step 19.66 ms under nsys; bench p50 19.54):

| class | calls | ms | % | bytes | bound | at bound ms | achieved | gap ms |
|---|---|---|---|---|---|---|---|---|
| MoE grouped GEMMs (fc1, fc2) | 184 | 5.62 | 28.6 | 36.5 GB | HBM | 4.56 | 0.81 | 1.06 |
| dense GEMMs (cuBLAS, incl. 185 split-K reduces) | 489 | 5.53 | 28.1 | 25.1 GB | HBM | 3.14 | 0.57 | 2.39 |
| all-reduce + combine + latent norm (`ar_finalize_rms`) | 92 | 2.53 | 12.9 | 332 MB NVL | fabric | 0.72 | 0.28 | 1.82 |
| MLA attention (DSL split kernel, prebuilt) | 24 | 2.05 | 10.4 | 10.9 GB | HBM | 1.36 | 0.66 | 0.69 |
| glue (moe_front, land_add2_attnres_rms, mla_prep, embed, head) | 211 | 1.43 | 7.3 | ~0 | launch | 0.42 | 0.30 | 1.01 |
| all-reduce + mix + norm (`ar_attnres_rms`) | 94 | 1.22 | 6.2 | 226 MB NVL | fabric | 0.57 | 0.47 | 0.65 |
| KDA core | 69 | 0.76 | 3.9 | 2.6 GB | HBM | 0.33 | 0.43 | 0.43 |
| DCP exchange | 24 | 0.33 | 1.7 | 50 MB NVL | fabric | 0.13 | 0.40 | 0.20 |
| MLA absorb | 24 | 0.19 | 1.0 | 0.6 GB | HBM | 0.08 | 0.40 | 0.11 |
| **step** | 1211 | **19.66** | | | | **11.31** | 0.58 | 8.35 |

**48 rows** (step 26.77 ms under nsys; bench p50 26.42):

| class | calls | ms | % | bytes | bound | at bound ms | achieved | gap ms |
|---|---|---|---|---|---|---|---|---|
| MoE grouped GEMMs | 184 | 8.42 | 31.5 | 55.5 GB | HBM | 6.94 | 0.82 | 1.48 |
| dense GEMMs | 489 | 5.87 | 21.9 | 25.1 GB | HBM | 3.14 | 0.54 | 2.72 |
| MLA attention | 24 | 4.48 | 16.8 | 21.7 GB | HBM (tensor 2.46, 3.28 with 96 heads padded to the 128-row MMA) | 2.72 | 0.61 | 1.77 |
| `ar_finalize_rms` | 92 | 3.22 | 12.0 | 665 MB NVL | fabric | 1.21 | 0.37 | 2.02 |
| `ar_attnres_rms` | 94 | 1.55 | 5.8 | 453 MB NVL | fabric | 0.90 | 0.58 | 0.65 |
| glue | 211 | 1.47 | 5.5 | ~0 | launch | 0.42 | 0.29 | 1.04 |
| KDA core | 69 | 1.01 | 3.8 | 5.2 GB | HBM | 0.65 | 0.65 | 0.36 |
| DCP exchange | 24 | 0.53 | 2.0 | 101 MB NVL | fabric | 0.21 | 0.39 | 0.33 |
| MLA absorb | 24 | 0.22 | 0.8 | 0.6 GB | HBM | 0.08 | 0.35 | 0.14 |
| **step** | 1211 | **26.77** | | | | **16.27** | 0.61 | 10.50 |

Dense GEMMs one by one (achieved HBM rate, 24 / 48 rows): front 86 MB 4.43 / 3.98 TB/s, qkvg 92 MB 4.93 / 4.58,
lat_up 51 MB 5.32 / 5.24, o_proj 22 MB 3.82 / 3.63, sh_down 11 MB 2.77 / 2.71, wfu 52 MB 4.09 / 3.70, q_b 57 MB
5.14 / 4.92, lm_head 294 MB 6.27 / 5.95. The f32-output GEMMs (`cublas_bf16_tn_f32`: front, qkvg, wfu) run
cuBLAS split-K with a separate `splitKreduce_kernel` (185 a step, ~2.9 us each) that `loop/count` does not see.
The small ones (o_proj, sh_down) pay a fixed ~3 us of ramp. GEMMs are out of scope; this is the largest gap.

Non-GEMM items over 1% of the step, per call (24 / 48 rows):

| item | % of step | median us | bound us | what is in the gap |
|---|---|---|---|---|
| `ar_finalize_rms` | 12.9 / 12.0 | 27.2 / 33.8 | 7.8 / 13.1 + ~2 epilogue | p10 11.2 / 16.4 = at the bound; the median waits ~16 us for the member with the most active experts (EP imbalance: the MoE GEMMs of the busiest member) |
| MLA attention | 10.4 / 16.8 | 85 / 187 | 57 / 113 (HBM), 137 at 48 rows if 96 heads pad to 128 | prebuilt DSL cubin at 5.3 / 4.85 TB/s; the 48-row case is near the padded tensor bound |
| `ar_attnres_rms` | 6.2 / 5.8 | 13.0 / 16.6 | 6.0 / 9.6 + ~2.3 epilogue | raw push + poll alone takes 8.4 / 11.9 us here: within ~1-2 us of the fabric |
| KDA core | 3.9 / 3.8 | 11.0 / 14.6 | 4.7 / 9.4 | 43% / 65% of HBM (d-attn: register cliff at 128) |
| `moe_front` | 3.5 / 2.5 | 7.4 / 7.5 | ~2 | dependent chain logits -> top-k -> last-block tables (harness floor 6.4) |
| `land_add2_attnres_rms` | 3.2 / 2.4 | 6.9 / 7.3 | ~2 | two dependent block reductions; 2.3 us at nb 0 to 8.4 at nb 8 (harness) |
| DCP exchange | 1.7 / 2.0 | 14.0 / 22.2 | 5.6 / 6.2 | one-shot all-to-all + LSE merge + v-up (d-attn, closed) |
| MLA absorb | 1.0 / 0.8 | 7.8 / 9.1 | 3.1 | 25 MB weight at 3.2 / 2.8 TB/s |

What it says: a step at every bound would be 11.3 / 16.3 ms against 19.7 / 26.8. Of the 8.4 / 10.5 ms gap,
the GEMMs hold 3.4 / 4.2 ms, the MoE all-reduce 1.8 / 2.0 ms (of which ~1.5 ms is the busiest member's
experts, i.e. MoE GEMM time again), MLA attention 0.7 / 1.8 ms, glue 1.0 ms (a latency floor of launches
that sit between GEMMs), the attention all-reduce 0.65 ms (fabric-bound in fact), KDA 0.4 ms.

Worked from it: the all-reduces leave HBM idle for 10-30 us while the next cuBLAS GEMM streams its weight cold
at 4-5 TB/s, so their idle blocks now pull that weight into L2 (log, 3.581 -> 3.522 ms/step).

My half is at its launch floor: per layer `ar_attnres_rms`, `moe_front`, `ar_finalize_rms`, `land_add2_attnres_rms`
between cuBLAS / TRT-LLM GEMMs, which bound every pair. Time: the all-reduces are fabric-bound (cross-member
stamps below), ar_finalize additionally waits on the member with the most active experts (MoE weight
bandwidth, fixed EP layout), the residual epilogues sit at two dependent block reductions, moe_front's
chain (logit load -> sigmoid -> picks -> last-block tables) at ~6.5 us. Every further trial is in the log
and findings below with its measurement.

## Proposals that need the runtime

- **A head without the logits gather in serving.** Member 0's copy of every member's logit slice exists
  only because the caller (k3_step) reads member 0's full logits; it costs 9-50 us a step (member 0's
  NVLink ingress, 8-48 rows). A program variant or a readback hint that says "logits not read" drops it.
- **Concurrent launch groups.** sh_down (cuBLAS) and the MoE batched GEMMs are independent and both
  memory-bound; on the member with the most active experts they serialize. A manifest construct for two
  streams joined before the all-reduce would let the shared expert hide under the routed one.
- **A startup barrier after the once-programs** in k3_step / kern-serve. The first-call handshake makes the
  Lamport kernels safe without it, but a slow member's setup (6 s seen) still lands in the first step.

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
| vocab-parallel lm_head (SGLang's layout for this group): each member's 20480-row slice GEMM, `kern_k3_head_argmax` copies the slice into member 0's full logits and exchanges one argmax key per row (3 stages, zero = not arrived) | 13.0 | 3.742 | lm_head GEMM at N = 20480 (another cuBLAS kernel): logits relRMS 1e-5 vs main, 0 flips |
| (main 0c3c812 + the head: 3.930 -> 3.632) | 13.0 | 3.632 | |
| (main 6bf21b5: 3.584) | 13.0 | 3.584 | |
| the MLA split plan (`mla_split_plan`, one launch a step) computed by each `embed_rms` block for its row → `kern_k3_embed_rms_plan` | 12.9 | 3.586 | bit-identical (integer plan) |
| the MoE batched GEMMs launch as programmatic dependents (`pdl: true` on fc1 / fc2: their cubins wait with griddepcontrol and trigger early) | 13.0 | 3.582 | unchanged (launch attribute only) |
| (main dcf6ec0: 3.581) | 12.9 | 3.581 | |
| the two all-reduces' blocks past B prefetch.global.L2 the next GEMM's weight (front 86 MB, lat_up 51 MB, capped 96 MB) after their pushes | 12.9 | 3.522 (same-session main 3.535-3.581) | bit-identical |

L2 prefetch trials (same-session A/B against the committed one-range per-line form, 3.535):
- prefetch at kernel entry in every block, and in `land_add2_attnres_rms` (B blocks: ~100k lines an SM at 8
  rows): 3.834. An SM issues prefetches at ~1 a clock or slower; keep the count per SM small.
- a second range (`ar_finalize` -> lat_up + the next layer's qkvg / wfu, 112 MB budget): 3.564, ~1% worse at
  <= 24 rows. The extra issue keeps the idle blocks alive past a short all-reduce.
- `cp.async.bulk.prefetch.L2` (64 KB an instruction): faulted when issued from `land_add2_attnres_rms` (cause not
  isolated); in the all-reduces, chunks dealt consecutively by thread put ~650 on one SM: 13.9 ms/step (the
  bulk unit drains slowly, the GEMMs after it crawl too); dealt over the blocks: 3.564 one range / 3.590 two.
  No better than per-line.

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
- The head was 370 us a step (every member computing all 163840 logits from a 2.35 GB weight). SGLang's
  captured graph for this group runs it vocab-parallel (a 1/8 GEMM, then a logits all-gather); now so do we,
  with the logits gathered to member 0 only (the caller reads member 0's) and the argmax by key exchange.
- **The slow checks (26.6 ms/step, orchestrator 21:24) are one member's late start, not the kernels.**
  Caught with an instrumented build (work/dbg, work/catch_slow.sh: device printf when the first-call
  handshake, an all-reduce poll or the head's key wait passes 5 ms, with the rank waited on): in a 12.7
  ms/step run every member waited 6.0 s in the first-call handshake for member 4, and nothing else waited
  over 5 ms in the whole run (6 s / 640 steps = the extra 9.4 ms/step). All members logged "loaded" at
  8.1 s; the 6 s went into member 4's setup after load (leases, tables, the first graph capture and
  instantiation) before its first all-reduce. Before the handshake that same skew tripped the 2 s
  deadline and gave wrong values; now it costs only startup time. Steady state: in normal runs a few
  dozen all-reduce polls a run wait 5-9 ms (members waiting for member 0, whose host reads and scans
  4 x 163840 logits after every check step); the head's key wait never passed 5 ms.
- **Residual-row epilogue floor** (single-GPU harness work/h/k1d_bench*.cu on land_add2_attnres_rms): empty
  launch 0.7 us; nb = 0 2.3 us; nb = 1 4.3; nb = 2 4.9; nb = 8 8.4. Of nb = 1's extra 2 us, the snapshot copy is
  ~0.2 and the mix ~1.3 (score of the prefix, two block barriers, combine, softmax, mix: two dependent block
  reductions with a softmax between them, and rms needs the bf16-rounded mix, so no algebraic shortcut).
  Interleaving the snapshots' butterflies: no gain. p-glue's 14-warp form (2 vectors a thread, same trees):
  4.16 vs 4.29 us at nb = 1, ~3%. moe_front with pdl: 3.580 vs 3.581 (nothing). Two-shot attention
  all-reduce from 33 rows (reduce-scatter to the row's owner, owner sums in rank order, gathers the bf16
  sum): 36 rows unchanged, 48 rows -0.5%; reverted (and a `when` pair counts as two launches in loop/count).
- Remaining D cost at 24 rows, per KDA layer ~180 us: MoE bmm ~75 (weight-bandwidth bound, varies by the
  member's active experts: ar_finalize waits ~20 us median on the slowest member), other GEMMs ~55 (front
  incl. cuBLAS split-K reduce, qkvg, o_proj, sh_down, lat_up), kda_core 11 (d-attn), my four kernels ~38.
  MLA layers add the DSL attention (96 us at 24 rows / 128k, ~4.7 TB/s, prebuilt) and the exchange.
- **Cross-member stamps of the attention all-reduce** (work/dbg/arf_t*.cu: block 0 and 5 of every member
  printf %globaltimer at entry / push issued / poll done for 4 calls; members of one host share the
  clock), 24 rows: members enter within 0-2 us; issuing the pushes takes ~4.9 us (blocks 0-20 each 1024
  vectors x 8 peers); polls complete ~10.5 us after the first entry, ~4 us after the last member's pushes
  were issued. Chunked pushes (every block a contiguous share, staged in smem): issue 3.6 us, last issue
  1.3 us earlier, polls only 0.3 us earlier. Delivery is fabric-bound (2.4 MB a member at 24 rows lands at
  an effective ~270 GB/s, plus a ~3 us hop seen at 4 rows), which also explains why the two-shot (half
  the bytes, two hops) did not pay. Pushing each chunk with cp.async.bulk (one bulk copy a peer from the
  staged smem) instead of 16-byte stores: polls done 10.0 vs 10.2 us, so it is not per-packet overhead.
  The all-reduce line is closed.
- **Real model, 93 layers, 128k** (work/d93bench.sh; 2 min weight load): start 3ad4ec9 23.88 / 30.87 ms
  a step at 24 / 48 rows, this branch 19.54 / 26.42 (-18% / -14%; every run's work). nsys at 24 rows:
  ar_finalize median 27 us (MoE skew), ar_attnres 12.9, kda_core 10.9, moe_front 7.3, K1d 6.9 (snapshot
  counts up to 8: +2.3 us over the bench's nb 1-2), dcp_exchange 13.7.
- moe_front top-k by threshold (the 16th largest ord by a 32-step block-wide __syncthreads_count search,
  then the survivors compacted and bitonic-sorted in one warp; quant/situ moved before it): 9.4 vs 6.4 us
  at 8-48 rows (harness). The counting barriers cost more than the two levels of warp rounds. Discarded.
- K1d over a 4-CTA cluster (a row's snapshots read by 4 SMs, group partials through DSMEM into the leader's
  slots, the same trees: bit-identical to the one-CTA kernel for nb 0-8, both snapshot flags, 8-48 rows in a
  harness, and in loop/d/check with every K1d forced onto it). Harness: nb 8 9.1 -> 7.1 us, nb 2-3 ~0.4 us
  better, nb 0 1 us worse. Used for nb >= 3 (69 calls of the 93-layer step): 19.54 -> 19.52 / 26.42 ->
  26.40 ms at 24 / 48 rows, nothing. Reverted.
- moe_front routing by slots claimed with atomics in each row's block (the last block then only scans the
  112 counts and places the picks; rows within an expert in arrival order, as FlashInfer's tables had
  them): 6.7 vs 6.4 us (harness), the atomic round trip lands on every row's path. Discarded.

