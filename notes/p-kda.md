# p-kda: the KDA half of a P layer

## Where it stands

| | launches (stage 1) | weighted P cost | 8192 rows, empty prefix |
|---|---|---|---|
| main (2026-10-10) | 325 (27.1/layer) | 56.82 ms/item | 105.9 ms |
| 6463ad5 packed gather | 298 (24.8/layer) | 54.34 ms/item | 100.6 ms |

A KDA layer now (non-GEMM launches): `span_gather` (1) → span_g* →
`flash_kda` (2: FlashKDA prepare / recurrence) → `span_state_store` (1) →
`kda_out_gate` (1).

Per-layer times at 8192 rows (l13, `work/calls.py`): span_gather 263 us,
FlashKDA prepare 276 us + recurrence 707 us (980 total), out_gate 154 us,
state store 8 us.

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
