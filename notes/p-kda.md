# p-kda: the KDA half of a P layer

## Where it stands

| | launches (stage 1) | weighted P cost | 8192 rows, empty prefix |
|---|---|---|---|
| main (2026-10-10) | 325 (27.1/layer) | 56.82 ms/item | 105.9 ms |
| 6463ad5 packed gather | 298 (24.8/layer) | 54.34 ms/item | 100.6 ms |
| 6abb2fe k3_kda_rec | 280 (23.3/layer) | 53.37 ms/item | 98.6 ms |
| main 40c1cda (p-glue merged) | 230 (19.2/layer) | 52.87 (52.70 re-measured) | 97.28 |
| 26f8f07 gate on idle SMs | 230 (19.2/layer) | 52.40 ms/item | 96.61 ms |

A KDA layer now (non-GEMM launches): `span_gather` (1: conv, beta / flow,
tile prefix, zeroes rec's progress counters) → span_g* → `flash_kda` (2:
FlashKDA prepare, then source/k3_kda_rec.cu: 96 recurrence CTAs (state
from/to the line, windows) + 56 gate CTAs on the idle SMs (K11)).

Per-layer times at 8192 rows (l13, `work/calls.py`): span_gather 263 us,
FlashKDA prepare 276 us + k3_kda_rec ~515 us.

## PROPOSAL for the orchestrator: KDA^2 (branch agent/p-kda-kda2, c92ec75)

NVlabs KDA^2's static-PTX stream kernel (kda-cake-ptx, the 2-CTA-cluster
`varlen_mixed` build), vendored in source/kda2 with a patch so it reads and
writes each sequence's state in its KDA line (bit-identical to upstream on
random inputs). On GB300: 371 us for 8192 rows x 96 heads (FlashKDA prepare
+ recurrence 983, ours 790), 192 us at 4096, 58 us at 1000, 12 us at 17.
The layer: gather (conv, flow, KDA^2's tables + descriptors) -> f_b GEMM ->
KDA^2 -> k3_kda_gate (K11 + windows): same 230 launches, 8192 rows 96.60 ->
93.89 ms (~ -3%).

It FAILs loop/p/check vs main: 19/23 argmax agree, 3 near-tie flips within
the KL limit, 1 beyond (the prefill's last row, KL 5.9e-2, main's margin
0.19). Cause: FlashKDA rounds the state to bf16 every 16 tokens, KDA^2 keeps
fp32. Against an fp64 token-by-token reference (work/kda2/ref2.cu, random
inputs) KDA^2's output is 4e-3 and final state 3e-3 relative RMS; NVlabs'
README measures FlashKDA's final state ~4% off on a real 8K prefill. So the
difference is main's error, but the A/B check cannot tell; it needs a
precision oracle (e.g. an fp64 / fp32-state reference P manifest) or the
orchestrator's call. The fixed-h96 build is 4% faster but stores a wrong
final state after a partial last chunk (do not use it).

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
3. KDA^2 (nvkda kda-cake-ptx): fixed_h96 PTX needs host-built piece/CTA
   tables from cu_seqlens (58 KB Python scheduler); no varlen h96 PTX.
