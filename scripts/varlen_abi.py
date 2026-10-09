"""The packed prefill's two vendored kernels over several sequences at once:
FlashKDA's varlen build and TRT-LLM gen's context FMHA at a batch of
`seqs`. Both reuse kern's captured ABIs (flash_kda_abi, trtllm_fmha_abi);
only what the batch changes is spelled here.

FlashKDA varlen (`flash_kda_d128_varlen`, the same sources built with
-DKERN_VARLEN): the template's IsVarlen flag flips and nothing else in the
parameter types, so the TiledCopy packs are the non-varlen ones. Sequence j
is rows [cu_seqlens[j], cu_seqlens[j + 1]) (int64) of q / k / v / g / beta /
out and block j of the f32 state in and out ([seqs][heads][128][128]).
Kernel 1 runs a tile per 16 rows of a sequence: a sequence adds at most
one partial tile, so `ceil(rows / 16) + seqs` bounds the grid, the
workspace and `total_tiles` (the workspace's head stride, the same in both
kernels); excess CTAs return. `_flash_kda_build_tile_prefix` fills the
tile prefix kernel 1 searches, kernel 2 runs one CTA per (sequence, head).
"""
import flash_kda_abi as kda
import trtllm_fmha_abi as fmha

KDA_MODULE = "flash_kda_d128_varlen"
BUILD_TILE_PREFIX = "_Z28_flash_kda_build_tile_prefixPKliiPi"
PREPARE = kda.PREPARE.replace("Li16ELi128ELi256ELb0EE", "Li16ELi128ELi256ELb1EE")
RECURRENCE = kda.RECURRENCE.replace("Li192ELb1ELb1ELb1ELb0EE", "Li192ELb1ELb1ELb1ELb1EE")
assert PREPARE != kda.PREPARE and RECURRENCE != kda.RECURRENCE


def tiles_max(rows_max, seqs_max):
    return -(-rows_max // kda.CHUNK) + seqs_max


def kda_workspace(hl, rows_max, seqs_max):
    """The six workspace arrays at the packed tile bound, the tile prefix and the staged states."""
    n = tiles_max(rows_max, seqs_max) * hl
    c, d = kda.CHUNK, kda.HEAD_DIM
    return {
        "span_ws_kd": {"dtype": "bf16", "shape": [n, c, d], "kind": "workspace"},
        "span_ws_qd": {"dtype": "bf16", "shape": [n, c, d], "kind": "workspace"},
        "span_ws_kr": {"dtype": "bf16", "shape": [n, c, d], "kind": "workspace"},
        "span_ws_gt": {"dtype": "f32", "shape": [n, d], "kind": "workspace"},
        "span_ws_inv": {"dtype": "bf16", "shape": [n, c, c], "kind": "workspace"},
        "span_ws_mqk": {"dtype": "bf16", "shape": [n, c, c], "kind": "workspace"},
        "span_tile_prefix": {"dtype": "i32", "shape": [seqs_max + 1], "kind": "workspace"},
        "span_state_in": {"dtype": "f32", "shape": [seqs_max * hl, d, d], "kind": "workspace"},
        "span_state_out": {"dtype": "f32", "shape": [seqs_max * hl, d, d], "kind": "workspace"},
    }


def kda_op(hl, rows_max, seqs_max, module, prefix_module, rows, seqs, scale=kda.QSCALE):
    """Interface: flash_kda_abi.op's seventeen, then cu_seqlens (i64 [seqs + 1]) | tile_prefix (i32
    [seqs + 1]) | seqs. `rows` / `seqs` are the call's row and sequence vars."""
    n = tiles_max(rows_max, seqs_max) * hl
    T, N, cu, tp = {"param": 16}, {"param": 19}, {"param": 17}, {"param": 18}
    tiles_expr = {"add": [{"ceil_div": [rows, kda.CHUNK]}, seqs]}
    tiles = {"pack": {"size": 4, "fields": [{"at": 0, "expr": tiles_expr}]}}
    tc = kda.tiled_copy
    D = kda.HEAD_DIM
    r = lambda param, box_rows: tc(param, "bf16", [D, rows_max, hl], [hl * 256, 256], [box_rows, kda.CHUNK, 1],
                                   dynamic=True)
    beta = tc(4, "bf16", [hl * rows_max], [], [32])
    dt_bias = tc(5, "f32", [D, hl], [512], [D, 1])
    ws_rows = lambda param: tc(param, "bf16", [D, kda.CHUNK, n], [256, 4096], [8, kda.CHUNK, 1])
    ws_gt = tc(13, "f32", [D, n], [512], [D, 1])
    ws_sq = lambda param: tc(param, "bf16", [kda.CHUNK, kda.CHUNK, n], [32, 512], [8, kda.CHUNK, 1])
    state = lambda param: tc(param, "f32", [D, D, hl * seqs_max], [512, 65536], [8, D, 1], swizzle=32)
    copies = ["bytes<256>"] * 11
    prefix = {
        **prefix_module, "entry": BUILD_TILE_PREFIX, "block": [32, 1, 1], "grid": [1, 1, 1],
        "params": ["in buffer<i64>", "i32", "i32", "out buffer<i32>"],
        "args": [cu, N, {"i32": kda.CHUNK}, tp],
    }
    prepare = {
        **module, "entry": PREPARE, "block": [256, 1, 1], "grid": [tiles_expr, hl, 1], "shared_mem": 21248,
        "params": copies + ["f32", "i32", "i32", "i32", "in buffer<i64>", "bytes<4>", "in buffer<f32>", "f32",
                            "in buffer<i32>"],
        "args": [r(0, D), r(1, D), beta, r(3, D), dt_bias, ws_rows(10), ws_rows(11), ws_rows(12), ws_gt, ws_sq(14),
                 ws_sq(15), {"f32": scale}, T, {"i32": hl}, N, cu, tiles, {"param": 6}, {"f32": kda.GATE_SCALE}, tp],
    }
    recurrence = {
        **module, "entry": RECURRENCE, "block": [192, 1, 1], "grid": [seqs, hl, 1], "shared_mem": 98432,
        "params": copies + ["out buffer<bf16>", "i32", "i32", "i32", "in buffer<i64>", "bytes<4>"],
        "args": [r(2, 8), beta, ws_rows(10), ws_rows(11), ws_rows(12), ws_gt, ws_sq(14), ws_sq(15),
                 state(7), state(8), r(9, 8), {"param": 9}, T, {"i32": hl}, N, cu, tiles],
    }
    return {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                   "in buffer<f32>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>", "out buffer<bf16>",
                   "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<f32>", "out buffer<bf16>",
                   "out buffer<bf16>", "i32", "in buffer<i64>", "out buffer<i32>", "i32"],
        "impl": {"launches": [prefix, prepare, recurrence]},
    }


def fmha_op(heads, q_max, kv_max, module, q, seqs):
    """trtllm_fmha_abi.op at a batch of `seqs` (a var): grid z and mBatchSize are the sequences, the
    length tables ([seqs] / [seqs + 1] / [seqs + 1]) are the caller's."""
    op = fmha.op(heads, q_max, kv_max, module, q)
    launch = op["impl"]["launches"][0]
    fields = launch["args"][0]["pack"]["fields"]
    at = {f["at"]: k for k, f in enumerate(fields)}
    fields[at[1132]] = {"at": 1132, "var": seqs}
    launch["grid"] = [launch["grid"][0], launch["grid"][1], seqs]
    return op
