# p-kda: the KDA half of a P layer

## Where it stands

| | launches (stage 1) | weighted P cost | 8192 rows, empty prefix |
|---|---|---|---|
| main (2026-10-10) | 325 (27.1/layer) | 56.82 ms/item | 105.9 ms |
| 6463ad5 packed gather | 298 (24.8/layer) | 54.34 ms/item | 100.6 ms |
| 6abb2fe k3_kda_rec | 280 (23.3/layer) | 53.37 ms/item | 98.6 ms |
| main 40c1cda (p-glue merged) | 230 (19.2/layer) | 52.87 (52.70 re-measured) | 97.28 |
| 26f8f07 gate on idle SMs | 230 (19.2/layer) | 52.40 ms/item | 96.61 ms |
| 64d3f13 one TMA box an array | 230 (19.2/layer) | 52.07 ms/item | 95.85 ms |

Stage 0 now (main 96fb15b + 53e41ef, 2026-10-11 00:30): 17.4 launches/layer,
~122 ms/item weighted on a normal GPU (p/ab). A KDA layer's non-GEMM
launches: span_gather, FlashKDA prepare, k3_kda_rec. That is 3, the floor
(see "Closed" below). In isolation (ncu, 8192 rows, one layer): gather 211
us, FlashKDA prepare 276 us, k3_kda_rec ~400 us (recurrence end ~490 in
the bench, the gate CTAs ~10 us after).

## Closed: why the KDA half stays at 3 launches and this time

- The recurrence is mma.sync-bound for chunk 16, and UMMA doesn't help at
  N=16..64 (below). Every reorder / pack / ring / publisher variant was flat.
- The gather and FlashKDA's prepare are each issue/MUFU-bound, not
  memory-bound (ncu: gather 154M warp-instr + 19M MUFU, prepare 209M +
  11M). Fusing them saves only DRAM traffic. A bit-exact fused prepare
  (work/prep2: conv + SiLU + FlashKDA kernel 1 in one kernel, the gather
  reduced to beta / flow, launch-neutral because extern:cublaslt_bf16_tn has
  lda = k, so f_b can't read the flow in place; bit-identical logits and
  states) ran 1194 -> 872 us after fixing its load indexing, against 487
  for the pair. Conv + exact SiLU alone is ~220M instructions there; the
  floor is ~400 us. Dropped.
- Stage 0 is power-capped: kernel speedups in this half don't reach the item
  time. A tanh-based SiLU made the gather 26% faster (2.52 -> 1.86 ms an
  8192 item) with the weighted item flat (-0.1%), and the check went
  INCONCLUSIVE (KL 1.18e-2): rejected. The f32 state (-3% flash_kda) was flat
  on the item too.
- What did move the score: 53e41ef (counters on their own lines), -0.65%
  on a GPU whose placement made the recurrence 2x slower. Anything in the
  P stage that many CTAs poll / release on one line is suspect (moe_route's
  grid barrier? p-glue's lane).

Stage 0 (scored from 2026-10-10 19:48; real data): flash_kda 6.9% -> 6.6%
of 8192@0 with 64d3f13; per layer at 8192@0 span_gather ~334 us (~283 after
eaf2849), flash_kda ~900-1000 us (FlashKDA prepare ~300 + k3_kda_rec), qkvg
3.0 ms.

Stage 0 is power-capped: with real activations the GPU sits at its power
limit, so an item's time follows its energy more than its kernels' latency.
eaf2849 cut span_gather by 20% a call (-640 us an 8192 item, same lease)
and the item total did not move: the GEMMs after it ran that much slower
(lower clocks). A memory-bound kernel finishing sooner buys little there;
fewer DRAM bytes / instructions overall is what can count. Per-call p50s
(`work/opsum.py OP SCENARIO runs...`) show a kernel's own change; the
weighted A/B is what is scored.

A KDA layer now (non-GEMM launches): `span_gather` (1: conv, beta / flow,
tile prefix, zeroes rec's progress counters) → span_g* → `flash_kda` (2:
FlashKDA prepare, then source/k3_kda_rec.cu: 96 recurrence CTAs (state
from/to the line, windows) + 56 gate CTAs on the idle SMs (K11, a warp pair
per quarter of a head's rows, 381c3b5)).

Per-layer times at 8192 rows (l13, `work/calls.py`): span_gather 263 us,
FlashKDA prepare 276 us + k3_kda_rec ~515 us.

## KDA^2: tried, not precise enough for P (branch agent/p-kda-kda2)

NVlabs KDA^2's static-PTX stream kernel (kda-cake-ptx, the 2-CTA-cluster
`varlen_mixed` build) vendored in source/kda2 with a patch so it reads and
writes each sequence's state in its KDA line (bit-identical to upstream on
random inputs). Layer: gather (conv, flow, KDA^2's piece / CTA tables and
descriptors) -> f_b GEMM -> KDA^2 -> k3_kda_gate (K11, windows). Fast: 371
us for 8192 rows x 96 heads on stage 1 (FlashKDA prepare + recurrence ~710);
stage 0 same-lease A/B at 8192@0: 140.2 / 138.8 -> 138.0 / 136.3 ms.

Numerics, the blocker:
- Its partial last chunk is inaccurate on real data (check INCONCLUSIVE at
  a 5990-row prefill, PASS at 6016 / 6144; random data in a harness never
  showed it). The last branch commit stops every chain at the sequence's
  last whole chunk and runs the rest (< 32 rows) in the gate kernel, token
  by token in f32 from the state KDA^2 left in the line.
- That f32 path over every row (`K14_ALL_ROWS` + `K9_NO_CHUNKS`) is an f32
  reference build of the packed KDA. `kern test --reference <it>` over
  seeds 0-3: per span, from the same inputs, main's gated output (FlashKDA +
  k3_kda_rec + K11) is within 1e-3 .. 1e-2 max abs of it, KDA^2's within
  0.2 .. 0.96 (growing with depth). End to end KDA^2 is a little closer on
  the prefill's last rows but spikes on single decode tokens that main and
  the reference agree on: vs main max KL 9.7e-3 (seed 0, PASS), 3.6e-2 with
  a flip (seed 2, FAIL); vs the reference 1.0e-2 / 3.2e-2 where main is
  4.0e-3 / 6.2e-3. Upstream states a 5e-2 numeric contract and falls back
  to an exact M64 schedule on large states or failed decay proofs; FlashKDA
  (bf16 state every 16 rows) is far closer to f32 than KDA^2 is.

So FlashKDA's chunked math stays; a faster P KDA has to keep its precision.
The reference build is the tool to judge one (a recipe: build the kda2
branch with those two defines, `loop/gen`, keep the manifest and the two
cubins next to `build/`, pass it as the check's reference, vary `--seed`).

## Tools (work/, not committed)

- `work/pbench`, `work/pcheck`: loop/p/bench / check with `KERN_CACHE_DIR`
  and `LD_LIBRARY_PATH` passed through `withgpu`'s ssh (the pool host is
  remote; without it kern cannot find freshly built cubins and tries to
  fetch them from the registry: HTTP 401).
- `work/qbench`: the same over `work/w8192.toml` (one scenario, 13 s).
- `work/ncu.sh MANIFEST OUT REGEX COUNT [ncu args]`: ncu a kernel inside a
  kern bench (`WL=w8192`), report on the shared fs.
- `work/calls.py A.json B.json [scenario] [layer]`: per-call p50 diff.
- `work/ab.sh A B OUT [WL]`: A B A B in one lease over a work/ workload,
  prints the GPU (`WITHGPU_GPUS=3` pins one). loop/p/ab is the number of record.
- `work/sassprof.sh MANIFEST OUT REGEX`: ncu SourceCounters of one launch,
  executed SASS instructions by 40-instruction region (where the
  instructions go).
- `work/slowsm.py LOG`: per-CTA end times / slow SMs from a k3_kda_rec
  build that prints `K12END cta smid start end` (globaltimer; the
  instrumentation is in the 53e41ef message's history, re-add as needed).
- `work/mb/umma.cu` (tcgen05 phase-1 cost), `work/mb/smspeed.cu` (per-SM
  FMA / MMA / L2 / LDS speed, all GPUs of a tray).

## Tried

- span_gather (842 us at 8192 rows, 17% DRAM): issue-bound. The IEEE `/` in
  SiLU has a per-element slow-path branch whose BSSY/BSYNC serializes the
  thread's columns (exact `/`: 920 us; `__fdividef`: 233 us). Kept exact:
  reciprocal + 2 Newton steps over d * 2^-64 (correctly rounded, scaled
  back exactly; d = inf -> -0; negated residual keeps -0). A fallback
  branch to `/` for "out of range" values cost a lot: the bench data has
  many sb <= -80 (it was 315-360 us with fallbacks / spills).
  Then latency-bound (long scoreboard, 36% DRAM) -> each thread streams
  its 48 rows through a 12-deep cp.async ring in smem: 263 us, 58% DRAM,
  65% issue. Floor ~175 us (issue) / ~160 us (DRAM).
- Folding the window update into the gather is unsafe (another block may
  still read the old window); it moved to the state store, which runs
  after FlashKDA.

- k3_kda_rec (FlashKDA kernel 2 + K11 + state + windows, bit-exact):
  first version 2080 us (loader starved: ~112 small cp.async.bulk per
  chunk + a dependent global beta load per chunk); 8-stage ring + beta
  of the whole sequence in smem: 1900; 2D TMA tensor copies with 128B
  swizzle (12 per chunk): 1300; gate epilogue 4 rows at once, no `/`
  slow path, 4 out stages: 770; 8 epilogue warps (2 rows each, 17 warps
  -> 96 regs with spills): 686; loader folded into epilogue warp 0 (16
  warps, 128 regs): 714 (the loader warp lags the others); epilogue with
  K11's tree by 16 threads a row (6 shuffles), 16B loads/stores: 634.
  Register budget: 17 warps cap at 96 regs (5 warps on one SMSP share
  its 16K registers), 16 warps at 128.
- Own port of FlashKDA's kernel 1 (k3_kda_prep, bit-exact: logits and
  every state identical; all FTZ ops, FMUL.D2 form of the sigmoid argument,
  the gate's first FFMA with RZ): 345 us vs FlashKDA's 276. With K9's conv
  fused in (gather left with the flow copy only): 1490 -> 900 us after
  cp.async staging + elementwise gate activation; still a loss vs gather
  263 + prep 276. It is instruction-bound (~550M warp instructions: the
  bit-exact SiLU alone ~20 instr/element x 3 streams) with __syncthreads
  phases where 1-4 warps work. Kept in work/k3_kda_prep_fused.cu (and the
  rec / abi / gen that go with it: work/*_fused.py, k3_kda_rec_wsv.cu).
  The conv contraction is fma(x,w3, fma(t2,w2, fma(t0,w0, t1*w1))) (nvcc's
  choice; spelling it out rebuilds the gather to the same cubin).

- Gate on idle SMs (26f8f07): rec CTAs publish output tiles (st.release,
  every 4 tiles) to q's buffer, 56 gate CTAs poll (ld.acquire by lane 0,
  nanosleep 256) and stage rows by cp.async. Lessons: ld.acquire.gpu
  compiles to LDG.STRONG + CCTL.IVALL (invalidates L1): 32 lanes polling
  made the kernel 3x slower; a per-tile __threadfence also; the gate CTA
  reading rows by plain LDG (L1 invalidated) was too slow, cp.async to smem
  fixed it; gate CTAs FIRST in the grid cost 2 ms/item (the recurrence lands
  on other SMs); a separate 200 MB raw-output buffer cost ~2 ms/item across
  all memory-bound kernels (footprint / layout), reusing span_q fixed it.
- Software-pipelining the recurrence (phase 6 of tile c interleaved with
  phase 1 of tile c+1): no gain (476 vs 450 us MMA-only). HMMA m16n8k16 on
  GB300: 8 cycles issue per SMSP, 20 cycles latency (work/mb/hmma.cu); the
  recurrence's tensor floor is ~832 cycles/tile on 96 SMs (213 us), it runs
  ~1000-1760.
- kern supports `pdl: true` on a launch (griddepcontrol.wait) and `cluster`.
- Where k3_kda_rec's time goes at 8192 rows (clock64 / globaltimer printf
  from one MMA warp, in the bench): 4 us prologue; the tile loop ~1665
  cycles a tile (full wait 84, phase 1 520, phases 2-4 210, phase 6 670,
  output-ring wait 90); the MMA warps issue ~345 instructions a tile each.
  With whole heads per gate CTA, the 40 two-head gate CTAs ended ~140 us
  after the recurrence: double-buffered loads (each warp's next 4-tile
  group in flight) and quarter-head units (<= 7/4 heads a CTA) fixed it
  (381c3b5; visible in one lease of five, flat in the others).
- The MMA loop's instruction budget a warp a tile (ncu, 330): 52 HMMA, 64
  FFMA + 64 unpacks (SHF / LOP3) + 48 F2FP for FlashKDA's bf16 state update
  (fma.ftz on f32 then RN), 27 LDSM, 18 LDS. Phase 1 runs at the tensor rate
  (64 HMMA an SMSP in ~520 cycles); phase 6 is issue-bound on the state
  update. What is left needs a different formulation, not scheduling.
- Rejected on k3_kda_rec (all bit-identical, all slower or flat):
  - K11 inside the recurrence CTAs (4 idle warps gating from the output
    ring, the gate's projection by cp.async; no raw round trip, no gate
    CTAs): the MMA phases slowed 30-60% (issue / shared-memory contention
    on the MMA warps' SMSPs), flash_kda 976 -> 1168 us.
  - Starting warps 4-7 half a tile late so the two MMA warps of an SMSP are
    in different phases: the offset random-walks (thousands of cycles
    either way by tile 400) and no phase got shorter.
  - Phase 1's q MMAs (out = S qd^T) moved into phases 2-4 (whose chain only
    u feeds): +250 us an item, 24 B of stack.
  - Phase 6's fma.ftz pairs as fma.rn.ftz.f32x2: +250 us an item (packing
    movs, 8 B of stack). f32x2 paid off in the gather (eaf2849), not here.
  - The loader / store warps sleeping (nanosleep 128) between failed
    barrier tries instead of spinning: +100 us an item (late refills).
- f32 state between tiles (22a1f9b, in main): phase 6 is 32 FFMA2, no
  unpacks; per-phase cycles a tile after it (clock64, instrumented): full
  wait ~110, phase 1 ~765 (it now packs the bf16 A fragments), phases 2-4
  ~185, phase 6 ~490, output-ring wait ~245-318, rest ~130.
- tcgen05 (UMMA) for the recurrence: no (work/mb/umma.cu, checked against
  the CPU). M128 x N16 x K16 costs ~50 cycles an instruction, the same at
  N=32 and ~54 at N=64, and a round trip (issue, commit, mbarrier,
  tcgen05.ld) adds ~140. Phase 1 (8 K-steps, kd|qd as N=32) comes to ~550
  cycles, which is the mma.sync time. Phase 6 (one M128 x N128 x K16) would
  only break even with its 256 HMMA cycles. Chunk 16 is too narrow for
  UMMA; KDA^2 gets its speed from chunk 64.
- Phase 4 (out = Mqk U + qS, off the state chain) moved after phase 6:
  flat to slightly worse (7.56 -> 7.66 ms flash_kda an item).
- The store warp's st.release.gpu every 4 tiles is a membar of 1.6k-9.3k
  cycles (it varies by SM), paid inline. With the 4-deep output ring
  backing up, the MMA warps waited ~245-318 cycles a tile. An 8-deep ring
  alone: -1% flash_kda. A publisher warp (warp 11, idle in recurrence CTAs)
  that st.release.gpu's whatever the store warp has copied (smem counter,
  release/acquire at cta scope, cumulative) drops the ring wait to ~82
  cycles (the barrier itself), about -7% on the tile loop.
  On a normal GPU the publisher was flat; on GPU 3 (pre-53e41ef) it was +12%
  flash_kda. Not kept; the real cause of the ring stalls was the next item.
- Placement / lease sensitivity, solved by 53e41ef. d-attn saw span_kda
  770 -> 1180 us from an unrelated buffer resize, and I saw flash_kda 7.8
  vs 8.9-10.4 ms an 8192 item for the same manifest on different GPUs.
  Per-CTA %smid + globaltimer (instrumented build, work/slowsm.py): on GPU 3
  of the tray, the recurrence CTAs on 34 fixed SMs (17 TPC pairs) ran at
  half speed (rec end 983 vs median ~500 us) in every call. ncu with
  --clock-control none: rec 846 us there vs 404 on GPU 1, same clocks. SMs
  uniform in isolation (work/mb/smspeed.cu). Gate CTAs exiting at once: no
  slow SMs (428 us). Gate CTAs that only poll: slow again (570). Dropping
  the gate's DRAM loads, stores, raw re-reads, discard, or polling 8x less
  often: still slow. Cause: the 96 progress counters on 3 L2 lines. Fix:
  one 128-byte line each.
- Short items: at 10 rows a KDA layer costs ~20 us flash_kda + ~13 us
  gather, at 354 rows ~63 + 24 us; next to the 8192-row items' weight in
  the score this is noise.

## Constraints learned

- FlashKDA is a prebuilt cubin (CuTe; loop/build only runs bare nvcc
  without include paths), so its kernels cannot be edited: fusing the
  output gate / state line into it means writing our own recurrence.
- kern expressions: `add` takes two expressions (`seqs` + a ceil_div is fine).

## Next

1. Own recurrence (done: k3_kda_rec). Own prepare with the conv in its load:
   done and dropped (work/prep2; above).
2. KDA^2: done, rejected on precision (above).
3. What is left needs a different algorithm: a chunk-64 recurrence (UMMA at
   N=64, a quarter of the serial steps) with FlashKDA-level precision. That
   means FLA-style 16-row sub-blocks with relative decays inside the chunk,
   which is where KDA^2 lost precision. Days, not a session.
4. Short items (10 / 354 rows, 13% of the weight) cost ~33 / ~87 us a KDA
   layer: <0.2% of the score.
