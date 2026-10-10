# Error budget: where main's numbers differ, and by how much

The rule: not bit-exact, but every numerical difference explained. This file
lists each source of difference in `main` (7839851; 96fb15b adds only
bit-identical D changes) for P (against the original base 3ad4ec9 and
against an f32 KDA reference) and for D (against the one-GPU oracle), says
how it arises, and gives its measured size. Each source was isolated where
possible: its commit against its parent, or a variant build that turns it
off. The tools and the variant patch are in `harness/error-budget/`.

## The floor comes first

Both checks run truncated models (P: 12 layers, D: 4 layers) with bf16
activations, and both already sit at a **bf16 noise floor**: any change in
rounding order, even one that computes the same math (another cuBLASLt
kernel for the same GEMM), moves the logits by that floor's amount. A
source's size only means something next to it:

| | floor sample | size |
|---|---|---|
| P (kern test p-l12, 6000 random tokens in 2048-row chunks + 20 decode steps, 23 logit rows) | lat_down + shared gate\|up as one GEMM (53ac505): cuBLASLt picks another kernel, a few landed values move by one ulp | KL(A‖B) per row median 1.0e-3, max 2.1e-3, 1 flip (margin 0.08) |
| P, a hypersensitive row (seed 2, step 27; see below) | the same GEMM merge | KL 6.3e-3 and a flip at that row; other sources 4.7e-3 to 4.7e-2 there |
| D (k3_step, 4 rows × 640 teacher-forced steps of real text, logits every 64th step) | the oracle itself at 8 / 16 rows vs at 4 rows: same math, rows 0-3 compared; cuBLAS picks other kernels at another M | top-1 0.923 / 0.925, relRMS median 1.89% / 1.78%, max 9.5% / 10.4%, KL median 6e-4 / 5e-4, max 1.6e-2 / 1.9e-2, 0 / 4 confident flips |

The D band `loop/d/check` judges against (main TP8 vs the oracle: top-1
0.9176, relRMS 1.77% / 8.3%, 1 confident flip) is one more sample from the
same distribution, **not** a TP8 or DCP effect: TP8 with both of its
structural roundings removed (below, v3) lands in the same place, and the
oracle against itself at 16 rows has 4 "confident" flips and relRMS max
10.4%, which would FAIL the check's band (max ≤ main + 2 points, ≤ 2
confident flips). The "confident" flips of every D comparison sit at oracle
margins 0.50-0.86 and recur at the same steps (step 71 row 0, step 225 row 0,
step 427 row 1): in a 4-layer stub's flat distributions those are floor
events, not bugs. The check's band is drawn from a single sample of the
floor; drawing it from several (oracle at 4 / 8 / 16 rows) would make it
honest.

Per pass (step-level runs over the first 16 steps, every step's logits kept;
median over steps 1-15 of the per-step median over rows):

| D comparison (per pass) | logits relRMS | KL |
|---|---|---|
| oracle 4 rows vs 8 rows (kernel choice only) | 3.1% | 1.9e-3 |
| oracle vs main TP8 | 2.4% | 1.0e-3 |
| main vs TP sums in f32 (v2) | 2.1% | 8.0e-4 |
| main vs DCP partial o at ~f32 (v1) | 0.82% | 1.2e-4 |
| main vs NCCL bf16 sums | 2.5% | 1.1e-3 |

There is no growth across steps (step 1 is already at 2-4%): the floor is
the stub's per-pass sensitivity to bf16 rounding, not an accumulation in
the KDA state. Step 0 is atypical (0.6-0.7% for every pair; v1 is
bit-identical there, as only member 0 holds a position).

## P: main vs base 3ad4ec9

End to end (kern test against base's p-l12, built and generated from base's
tree): KL median 1.28e-3, mean 1.43e-3, max 4.63e-3, 4 flips at near-ties
(base margins 0.009, 0.017, 0.040, 0.061; each token is the other's #2 or
#3). Seed 1: KL median 1.17e-3, max 4.8e-3, 1 near-tie flip. Seed 2 FAILs
kern test's limit at one row (below). With 400-row chunks (prefill 3000, so the
FMHA split runs): KL median 1.01e-3, max 9.25e-3, 0 flips.

Each source, its commit against its parent (both trees built and generated;
the check's workload unless noted). "Local" is kern test's same-input span
replay: the error the source injects where it enters.

| source | mechanism | end to end | local |
|---|---|---|---|
| f32 KDA state between tiles (22a1f9b) | FlashKDA rounded the recurrent state to bf16 after every 16-row tile; it now stays f32 (and its fma is not ftz) | KL median 1.23e-3, max 4.65e-3, 2 flips (0.038, 0.055) | gated (KDA layers) max abs 0.086 worst span, 0.016 median |
| lat_down + shared gate\|up as one GEMM (53ac505) | cuBLASLt picks another kernel at N = 15872: other k-order in a few dot products | KL median 1.0e-3, max 2.1e-3, 1 flip (0.080) | latent_q: ±1 mxfp8 step on ≤0.1% of codes; shared_act ≤1 bf16 ulp on ≤0.7% |
| lat_up + shared down as one k-concat GEMM (cd94d72) | bf16(latup + shdown) instead of bf16(latup) + bf16(shdown): one rounding fewer, plus the kernel choice | KL median 9.0e-4, max 2.3e-3, 3 flips (0.035, 0.062, 0.214) | hidden ≤0.125 abs (one ulp at the residual's magnitude) on ≤31% of elements |
| absorbed MLA for chunks ≤ 128 rows (4840069) | scores over q·W_UK in the 512 + 64 latent dims instead of the expanded 128 + 64 keys, W_UV after the softmax, other bf16 landing points; the split merge is FlashInfer's reduction arithmetic (98a2022 made the fused merge bit-identical to it) | decode rows: KL median 4.5e-4, max 2.1e-3, 1 flip (0.009); 100-row chunks: median 4.6e-4, max 2.5e-3, 1 flip (0.111) | gated ≤ 0.0078 (one ulp), median 0.002 |
| FMHA 3-piece causal split, 129-512-row chunks (894a155) | the heaviest sequence's attention in three pieces, merged in f32 from bf16-rounded piece outputs through the softmax stats | not run by the 2048-row check; 400-row chunks: KL median 3.9e-4, max 1.9e-3, 0 flips | gated ≤ 0.0156 (one ulp) on ≤22% |
| bf16 wfu output from cuBLASLt's epilogue (d0a09db) | every reader landed the f32 output to bf16 first; the epilogue does the same rounding | bit-identical at 2048- and 400-row chunks | — (another row count may pick another kernel: then floor-sized) |
| deterministic routing tables (9c2c46d) | FlashInfer's atomic order inside an expert replaced by row order | bit-identical | — |

Every other P commit since base states, and its check showed,
bit-identical outputs. The sources do not add up linearly: each alone is
at the floor, and so is their sum (main vs base).

**Seed 2, step 27** is a hypersensitive row (a decode step whose top two
tokens are 0.001-0.13 apart and whose distribution moves a lot with them):
every source moves it far past the others. KL there: f32 KDA state 4.7e-2,
k-concat GEMM 1.75e-2 (INCONCLUSIVE alone), the kernel-choice-only GEMM
merge 6.3e-3 (flips), absorbed MLA 4.7e-3; main vs base 5.2e-2 with a flip,
so main FAILs kern test's 1e-2 limit against base at that seed. At that row
main's token (39955) is the f32 KDA reference's and base's (139410) is not:
the f32 state moves it toward f32. The limit of 1e-2 sits inside the floor
of such rows; the other 30 rows of seed 2 are at KL ≤ 3.7e-3.

### Against an f32 KDA reference

The only f32 reference that exists is p-kda's build of the packed KDA that
runs every row a token at a time in f32 (branch agent/p-kda-kda2 with
K14_ALL_ROWS + K9_NO_CHUNKS; it is based on 64d3f13). To compare like with
like, the two other sides are 64d3f13 itself (bf16 state) and 64d3f13 with
22a1f9b cherry-picked (f32 state): everything but the KDA is the same in
all three.

| vs the f32 reference | bf16 state (64d3f13) | f32 state (64d3f13 + 22a1f9b) |
|---|---|---|
| local: KDA gated output, prefill spans (27) | worst max abs 0.086 | worst 0.023; closer in 12 spans, equal in 15, never farther |
| end to end, seed 0: KL median / mean / max | 1.45e-3 / 1.88e-3 / 8.95e-3 | 1.27e-3 / 1.72e-3 / 1.19e-2 |
| local, seeds 1 / 2 (225 / 279 KDA spans) | worst 0.078 / 0.086 | worst 0.031 / 0.031 |
| end to end, seed 1: KL median / mean / max | 1.57e-3 / 1.96e-3 / 5.5e-3 | 1.03e-3 / 1.84e-3 / 8.3e-3 |
| end to end, seed 2 | 1.44e-3 / 2.82e-3 / 4.3e-2 (FAIL: flip at step 27, margin 0.001) | 1.25e-3 / 2.34e-3 / 3.4e-2 (same token as the reference) |

So the f32 state is closer to f32 where it enters (every span, all three
seeds; its end-to-end median and mean are lower at every seed), and end to
end both are at the floor (the end-to-end max is one random-token row and
swings either way). What is left of main's KDA error against f32 is the
chunked algorithm's bf16 MMA operands (q, k, v, the gated products):
local ≤ 0.031 abs on the gated output, end to end at the floor. Removing it
means f32 or tf32 MMAs in the chunked kernels, i.e. a slower KDA, for an
effect the end-to-end metric cannot see.

## D: main TP8 vs the one-GPU oracle

The oracle (d-l4-tp1.json) of main is bit-identical to the base's oracle
(rerun on main, 2560/2560 tokens, logits identical): nothing in main changed
the one-GPU path's numerics. So every D difference is in how the TP8 / EP8 /
DCP8 step splits the work. The isolation variants are in a scratch tree
(`harness/error-budget/d-variants.diff`; not for main): **v1** carries each
member's partial o through the DCP exchange as bf16 hi + bf16 lo (hi + lo
≈ 16 mantissa bits, so the extra rounding is gone); **v2** sums every TP
partial in f32 (o_proj, the dense down projection, the per-rank MoE combine
and the shared expert: f32 GEMM outputs, an f32 finalize, nccl_allreduce_f32,
then one landing to bf16, exactly the one rounding the oracle has); **v3**
is both.

| TP8 run | vs oracle (640 steps) | vs main (640 steps) |
|---|---|---|
| main | top-1 0.9176, relRMS 1.77% / 8.3%, KL 5.4e-4 / 1.25e-2, 1 confident | — |
| v1: DCP partial o at ~f32 | 0.9203, 1.82% / 8.3%, 1 confident | 0.977, 0.80% / 8.0% |
| v2: TP sums in f32 | 0.9266, 1.89% / 10.3%, 3 confident | 0.924, 1.84% / 10.3% |
| v3: both | 0.9227, 1.89% / 10.3%, 3 confident | — |
| NCCL bf16 sums (no `--peer-ar`) | 0.9129, 2.33% / 9.4%, 3 confident | 0.913, 2.08% / 9.4% |
| main at 8 rows, rows 0-3 | bit-identical to main at 4 rows (the TP8 step is row-count invariant; the oracle is not) | |

Sources:

| source | mechanism | size |
|---|---|---|
| TP partial sums | each member's o_proj / dense-down / MoE-combine / shared-down partial is rounded to bf16 before the f32 rank-order sum and the final bf16 rounding; the oracle rounds once | per pass 2.1% logits relRMS (main vs v2), the floor's size; over 640 steps indistinguishable from the floor (v2 vs oracle 1.89% median) |
| DCP partial o in bf16 | a member's merged o over its positions is rounded to bf16 for the exchange, then the members' partials merge by LSE in f32 and round again (the oracle's single-split o rounds once) | per pass 0.82% (main vs v1), a third of the floor; 640 steps: v1 vs oracle as main |
| DCP LSE merge arithmetic | expf weights over 8 members' lse, the send side's ex2 / lg2 approximations (exact when a member has one split, as in the check); the oracle's DSL kernel merges online over its tiles | inside v3's residual: v3 vs oracle per pass 2.05%, the floor |
| GEMM kernel choice of the sharded GEMMs | qkvg / q_b / o_proj at N or K / 8 per member pick other cuBLAS kernels than the oracle's full-size ones | inside v3's residual, floor-sized (the oracle's own 4 → 8 rows change is 3.1% per pass) |
| W_UV · o order in the exchange (1242531) | the 512-long f32 dot as 16-column fma chains per lane + a 32-lane butterfly instead of 64-column slices summed over 8 | vs the previous main: 7 flips (none confident), relRMS median 0, max 0.6%, KL max 6e-5; vs oracle unchanged |
| vocab-parallel lm_head (9850876) | the head GEMM at N = 20480 per member runs another cuBLAS kernel | relRMS 1e-5, KL 1.4e-10, 0 flips |
| -0.0 sent as +0.0 in the exchange (c3bd3d1) | the Lamport poison is a bf16 -0.0 | bit-identical (an o of exactly -0.0 merges as +0.0) |
| ≤ 16 KV splits a row (e43336a) | a row whose plan wants more splits runs 16; the DSL merge is the same math | not exercised by the check (contexts ≤ 640: one split per member); at the bench's 64k-128k a row runs ≤ 16 splits a member, merged by the DSL's own reduction arithmetic |
| Lamport all-reduce vs NCCL | main sums the 8 bf16 partials in f32 in rank order and rounds once; NCCL's ring rounds to bf16 at each hop | main is the more precise of the two: NCCL bf16 sums are 2.33% / 3 confident vs main's 1.77% / 1 |

The D band is the floor. The two structural TP8 roundings are real but
the larger of them (the TP partial sums) is the floor's own size per pass,
and the DCP one is a third of it; neither shows over 640 steps.

## Summary

| source | P / D | size | why acceptable, or what would remove it |
|---|---|---|---|
| bf16 activation floor (rounding order of any op) | both | P: KL median 1e-3, max 2e-3; D: relRMS 1.8-1.9% median, up to 10%, 0-4 "confident" flips at margins ≤ 0.86 | the model runs bf16 activations; two equally valid orders differ by this much. The checks' bands should be drawn from several floor samples |
| f32 KDA state (22a1f9b) | P | at the floor end to end (but 4.7e-2 at seed 2's hypersensitive row, where it takes the f32 reference's token); locally closer to the f32 reference (worst span 0.078-0.086 → 0.023-0.031) | it is a precision gain |
| chunked KDA's bf16 MMA operands vs f32 recurrence | P | local ≤ 0.031 abs on the gated output; end to end at the floor | f32 / tf32 MMAs in the chunked kernels (slower) |
| merged GEMMs' kernel choice (53ac505, cd94d72) | P | at the floor (KL ≤ 2.3e-3, near-tie flips) | cuBLASLt's pick; pinning the algorithm per shape would make it bit-stable, not more precise |
| one rounding fewer in lat_up + shared down (cd94d72) | P | at the floor | a precision gain |
| absorbed MLA for short chunks (4840069) | P | KL median 4.5e-4, max 2.5e-3, near-tie flips only | the same attention in another basis (the decode step's form); removing it costs 2.3 ms per MLA layer at a 131k prefix |
| FMHA 3-piece split (894a155) | P | KL median 3.9e-4, max 1.9e-3 (129-512-row chunks only) | one bf16 ulp in the merged rows; an f32 piece output would remove it |
| bf16 wfu epilogue (d0a09db) | P | bit-identical at the measured shapes | — |
| TP partial sums in bf16 | D | per pass 2.1%, = floor; 640 steps: floor | SGLang's layout does the same; f32 partials (2x all-reduce bytes) would remove it (v2), for no visible gain |
| DCP partial o in bf16 | D | per pass 0.82%, a third of the floor | hi + lo bf16 (2x exchange bytes) would remove it (v1), for no visible gain |
| DCP LSE merge, sharded GEMMs' kernel choice | D | inside the floor (v3 residual) | inherent to splitting the work |
| W_UV · o order (1242531) | D | relRMS max 0.6% vs previous main, 7 near-tie flips | another f32 order of the same dot |
| vocab-parallel head (9850876) | D | relRMS 1e-5 | cuBLAS kernel choice at the head |
| -0.0 → +0.0 in the exchange | D | bit-identical | — |
| ≤ 16 KV splits a row | D | not exercised by the check | same merge math; only rows that wanted more splits run fewer |

## How it was measured

- P: a git worktree per commit (base, main, each source's commit and its
  parent, the f32 reference), `loop/build` + `loop/gen` in each, then
  `harness/error-budget/pt.sh A.json B.json OUT [kern test args]` (loop/p/check's
  kern test with the workload free: `--prefill / --chunk / --seed`) on one
  P GPU; `psum.py` / `pspan.py` read the reports (per-row KL median / mean /
  max and flips; per-buffer local span diffs).
- D: `dt.sh` runs a TP8 manifest as loop/d/check does and compares it with
  `dcmp.py` (top-1 over all 2560 tokens, relRMS and KL over the 40 kept
  rows); `drun.sh` runs any D manifest over the first 16 steps with every
  step's logits kept, `dsteps.py` compares them step by step; `subrows.py`
  cuts rows 0-3 out of an 8- or 16-row run.
- KL is KL(A‖B) of the softmaxed logits, A the reference; P's rows are
  random-token rows (the check's workload), D's real text.
