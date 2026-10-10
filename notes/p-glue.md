# p-glue: the MoE glue and the MLA glue of a P layer

Target: P stage 1 (`loop/p/bench loop/out/p-stage1.json`), check `loop/p/check loop/out/p-l12.json`.
Main (2026-10-10): 325 launches (27.1/layer), 56.82 ms/item.

## State

| commit | launches/layer (stage1) | cost ms/item | note |
|---|---|---|---|
| main | 27.1 | 56.82 | |
| moe route | 24.2 | 56.69 | router_topk + moe_quant + 3 FlashInfer routing → 1 (+1 init per stage) |

## Tooling notes

- `loop/p/bench` / `loop/p/check` run on the leased host over ssh, where `KERN_CACHE_DIR` is not
  inherited: a new cubin is fetched from the registry (401). `work/p.sh bench|check` is the same
  command with the cache forwarded (kept out of main).
- `work/ops.py REPORT [op...]`: per-op in-program time per scenario (bracketed, inflated for small ops).
- `work/hb/`: standalone harness of k3_moe_route.cu against the old top-k / quant kernels and a
  CPU build of the tables (`withgpu -n 1 ./hb`).

## Done

### k3_moe_route.cu: top-k + quant + routing tables in one launch
- Persistent grid (≤ 304 blocks of 512, two an SM, 64 regs) with one grid barrier; the barrier
  and the tables live in a workspace zeroed by `moe_route_init` once where a stage starts (a
  carry is refused by `kern cut`: "writes carry buffer before the last stage").
- First version (atomic ranks, 128 regs, the old 16-round top-k) was slower: 125 us at 8192
  rows vs ~108 us of the five old kernels. Fixes: top-k by threshold (16th largest lane max)
  + candidate ranking instead of 16 serial rounds; contiguous experts per lane (float4 loads);
  sigmoid recomputed for the picks instead of kept; quant 8 elements a lane; 64 regs.
- Token order inside an expert matters to moe_fc1: atomic order cost +40 us/layer of FC1 at
  8192 rows (1756 vs 1716 us). Sorted order (tested with a throwaway sort kernel) gives it back.
  The tables are now built from per-expert row bitmaps + 256-row chunk counts: exact row order,
  deterministic tables.
- Result: 8192 rows 105.76 (main 105.90), r10 11.26 (11.39), r354 14.24 (14.34); weighted 56.69.
