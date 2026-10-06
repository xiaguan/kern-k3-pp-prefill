"""Per-snapshot and per-row breakdown of a stage's diff against SGLang.

    blocks.py <sglang dump dir l<A>-<B>> <kern out dir>
"""
import json
import pathlib
import sys

import numpy as np

from compare import H, bf16

sgl, kern = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
meta = json.load(open(sgl / "meta.json"))
t, nb = meta["tokens"], meta["out.residual"]["shape"][1]
a, b = bf16(kern / "blocks.bf16", (t, nb, H)), bf16(sgl / "out.residual", (t, nb, H))
nb_in = meta.get("in.residual", {"shape": [0, 0]})["shape"][1]
for k in range(nb):
    d = (a[:, k] - b[:, k]).astype(np.float64)
    rel = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(b[:, k], axis=1), 1e-30)
    print(f"block {k} ({'copied in' if k < nb_in else 'written'}, layer {12 * k}): rel_l2 "
          f"{np.linalg.norm(d) / np.linalg.norm(b[:, k]):.3e}  row rel p50 {np.median(rel):.3e} "
          f"p99 {np.quantile(rel, .99):.3e} max {rel.max():.3e}  ref row norm p50 "
          f"{np.median(np.linalg.norm(b[:, k], axis=1)):.3g}")
h, r = bf16(kern / "hidden.bf16", (t, H)), bf16(sgl / "out.hidden_states", (t, H))
rel = np.linalg.norm(h - r, axis=1) / np.linalg.norm(r, axis=1)
print(f"hidden row rel p50 {np.median(rel):.3e} p90 {np.quantile(rel, .9):.3e} p99 {np.quantile(rel, .99):.3e}; "
      f"rows 0..4 {np.round(rel[:5], 3).tolist()}; ref row norm p50 {np.median(np.linalg.norm(r, axis=1)):.3g} "
      f"row0 {np.linalg.norm(r[0]):.3g}")
