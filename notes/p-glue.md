# p-glue: the MoE glue and the MLA glue of a P layer

Target: P stage 1 (`loop/p/bench loop/out/p-stage1.json`), check `loop/p/check loop/out/p-l12.json`.

## State

| commit | launches/layer (stage1) | cost ms/item | note |
|---|---|---|---|
| main 2026-10-10 | 27.1 | 56.82 | |
| moe route | 24.2 | 56.69 | router_topk + moe_quant + 3 FlashInfer routing → 1 (+1 init per stage) |
| situ in route | 23.2 | 56.64 | shared expert's situ made by the route kernel |
| (main 8f3d6a4, with p-kda) | 20.9 | 54.19 | |
| residual on 14 warps | 20.9 | 53.96 | K1a/K1b/K1d 448 threads a row, bit-identical |

A MoE layer is now: `land_add_attnres_rms_bf16` → router* → lat_down* → wsh* → `moe_route` →
fc1* → fc2* → `moe_finalize_rms` → lat_up* → sh_down* → next `land_add2_attnres_rms`.
Non-GEMM launches: 4 a layer (+ `moe_route_init` once a stage). Without touching the GEMMs that is
the floor: every remaining glue kernel sits between two GEMMs.

Shares (ablate, main 8f3d6a4): moe_finalize_rms 2.03%, land_add_attnres_rms_bf16 2.02%,
land_add2_attnres_rms 1.96%, moe_route 1.52%, mla_prep 0.30%, mla_gate 0.29%, latent_gather 0.08%.
Per call at 8192 rows (bracketed): residual ~195 us (floor ~100 us at 8 TB/s), finalize 190 us
(floor ~125 us), route 136 us, mla_prep 113 us, mla_gate 113 us (floor ~75 us).

## Tooling

- `work/ops.py REPORT [op...]`: per-op in-program time per scenario (bracketed, inflated for small ops).
- `work/hb/`: harness of k3_moe_route.cu against the old top-k / quant / situ kernels and a CPU
  build of the tables (`loop/withgpu -n 1 ./hb`).
- `work/hr/`: harness of the residual kernels (K1b / K1d) of a candidate source against the
  committed one, bit-compared and timed at 10 / 354 / 8192 rows, nb 1 / 2 / 4.

## Done

### k3_moe_route.cu: top-k + quant + routing tables (+ situ) in one launch
- Persistent grid (304 blocks of 512, 64 up to 256 rows; two an SM, 64 regs) with one grid
  barrier; barrier and tables live in a workspace zeroed by `moe_route_init` where a stage starts
  (a carry is refused by `kern cut`: "writes carry buffer before the last stage").
- Top-k by threshold (16th largest lane max) + candidate ranking instead of 16 serial rounds;
  contiguous experts per lane; sigmoid recomputed for the picks. Bit-identical picks / weights.
- Token order inside an expert matters to moe_fc1: atomic order cost +40 us/layer of FC1 at 8192
  rows. Tables are built from per-expert row bitmaps + 256-row chunk counts: exact row order,
  deterministic.
- situ (and the quant) as 256-column pieces spread over every warp of the grid; at a row a warp
  it made 354-row items slower (one warp per row, latency-bound).
- Role split (half the warps top-k, half quant/situ): better at small B, worse at 8192 (top-k warps
  become the tail). Not kept.

### Residual kernels on 14 warps (bit-identical)
- 7 → 14 warps a row: 80 → 60 regs, no spills, 2 blocks/SM: 197.6 → 173.6 us (K1d, 8192, nb 2).
- Tried on top (harness, all bit-identical, none kept):
  - packed fp32 (FFMA2/FADD2: (sq, dp) chains in lockstep, mix pairs): no gain alone; with the
    first two snapshot candidates prefetched into registers it spilled (72 regs).
  - TMA bulk staging of the row's operands + 2 candidates in smem (mbarrier): 168.7 us at nb 2,
    but slower at 354 rows (11.5 vs 10.1 us) and nb 4 (231 vs 218 us).
  - 28 warps (1 vector a thread): 243 us; launch bounds 3 blocks: spills.
  - What remains is half issue (≈ 890 warp-instr a row a warp: unpacking, 7 butterflies of 2
    groups, the scoring and the mix) and half DRAM at 58% of peak.

## Out of scope (measured, kept on branch `wip/gemm-merge`, not in main)

The orchestrator ruled GEMM merges out of scope (18:13). Measured before that:
- lat_down + wsh as one GEMM (weight [lat_down; gate; up], 15872 columns): −1 launch/layer at equal
  weighted cost (10 rows −0.8%, 354 rows +0.26%); cuBLASLt picks another kernel, a few landed
  bf16 values move one ulp: end-to-end KL ≤ 2.5e-3, 2 near-tie flips.
- lat_up + sh_down as one K-concat GEMM ([routed latent normed | shared act] @ [lat_up | sh_down]^T,
  K = 9728, weight concatenated by columns in the derive program): −1 launch/layer more and −0.33%
  weighted (one bf16 partial instead of two; `land_add2_attnres_rms` with two = 0). Numerics:
  hidden = bf16(prefix2 + bf16(latup + shdown)) instead of bf16(prefix2 + bf16(latup) + bf16(shdown)).

## Next

1. moe_finalize_rms speed (190 us at 8192 vs ~125 floor).
2. latent_gather into mla_prep (−1 launch per MLA layer).
3. mla_gate: only neighbours are the prebuilt FMHA and cuBLAS o_proj: no fusion without our own FMHA.
