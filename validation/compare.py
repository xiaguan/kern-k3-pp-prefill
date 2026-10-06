"""Compare kern stage outputs with SGLang's PP proxy tensors.

    compare.py <sglang dump dir l<A>-<B>> <kern out dir>
    compare.py head <kern out dir> <sglang generate.json>

`hidden` [T, H] against SGLang's out.hidden_states, the valid bank rows
[T, ceil(B/12), H] against out.residual: max |diff|, relative L2, cosine,
and the worst row by relative L2.
"""
import json
import pathlib
import sys

import numpy as np

H = 7168


def bf16(path, shape):
    u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape)


def stats(name, a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    rows = d.reshape(-1, H)
    ref = b.reshape(-1, H).astype(np.float64)
    rel_rows = np.linalg.norm(rows, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    cos = float((a.ravel().astype(np.float64) @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"{name:10s} max|d| {np.abs(d).max():.4g}  max|ref| {np.abs(b).max():.4g}  "
          f"rel_l2 {np.linalg.norm(d) / np.linalg.norm(b):.3e}  cos {cos:.7f}  "
          f"worst row {int(rel_rows.argmax())} rel {rel_rows.max():.3e}")
    return {"max_abs": float(np.abs(d).max()), "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(b)), "cos": cos}


def main():
    sgl, kern = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    meta = json.load(open(sgl / "meta.json"))
    t, end = meta["tokens"], meta["end_layer"]
    nb = meta["out.residual"]["shape"][1]
    assert nb == -(-end // 12), (nb, end)
    out = {"hidden": stats("hidden", bf16(kern / "hidden.bf16", (t, H)), bf16(sgl / "out.hidden_states", (t, H))),
           "bank": stats("bank", bf16(kern / "blocks.bf16", (t, nb, H)), bf16(sgl / "out.residual", (t, nb, H)))}
    json.dump(out, open(kern / "compare.json", "w"), indent=1)




def head(kern, generate):
    """The last stage: kern's logits of the last row against SGLang's top-5 logprobs."""
    logits = np.fromfile(pathlib.Path(kern) / "logits.f32", dtype=np.float32).astype(np.float64)
    lp = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
    top = np.argsort(-lp)[:5]
    ref = json.load(open(generate))["meta_info"]["output_top_logprobs"][0]
    print("kern top5 ", [(int(i), round(float(lp[i]), 4)) for i in top])
    print("sgl  top5 ", [(t, round(l, 4)) for l, t, _ in ref])
    print("logprob diff on SGLang's top5:", [round(float(lp[t]) - l, 4) for l, t, _ in ref])


if __name__ == "__main__":
    head(sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else main()
