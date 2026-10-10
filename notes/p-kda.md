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

Stage 0 now (main dcf6ec0, 2026-10-10 22:10): 17.2 launches/layer, ~125
ms/item weighted. A KDA layer's non-GEMM launches: span_gather, FlashKDA
prepare, k3_kda_rec (3; the floor while the prepare stays FlashKDA's
prebuilt cubin). At 8192@0 per layer: span_gather ~283 us, flash_kda ~880 us.

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
- Short items: at 10 rows a KDA layer costs ~20 us flash_kda + ~13 us
  gather, at 354 rows ~63 + 24 us; next to the 8192-row items' weight in
  the score this is noise.

## Constraints learned

- FlashKDA is a prebuilt cubin (CuTe; loop/build only runs bare nvcc
  without include paths), so its kernels cannot be edited: fusing the
  output gate / state line into it means writing our own recurrence.
- kern expressions: `add` takes two expressions (`seqs` + a ceil_div is fine).

## Next

1. Own recurrence kernel replacing FlashKDA kernel 2 (reads kernel 1's
   workspace): state straight from / to the KDA line (no staging), the
   output gate (rms, gamma_o, sigmoid gate) in its epilogue, the window
   update, more CTAs than 96 (FlashKDA runs one CTA per (seq, head) on
   152 SMs; each warp owns 32 dv columns independently -> split dv over
   CTAs; the per-row rms over 128 dv then needs a cluster / DSMEM sum).
2. Own prepare with the conv in its load (q/k never written).
3. KDA^2: done, rejected on precision (above).
