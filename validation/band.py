"""Three producers at every stage boundary: SGLang (fp8 KV), SGLang (bf16 KV), kern chained.

    band.py <data dir> [kern run, default chain]

For each boundary, the relative L2 of `hidden` and of the bank between every
pair; at the head, each producer's top-5 logprobs and the largest |Δ| over
the union of the top-5s where both have the token. kern is inside the band
when its distance to either SGLang is of the order of SGLang's distance to
itself.
"""
import json
import pathlib
import sys

import numpy as np

from compare import H, bf16

data = pathlib.Path(sys.argv[1])
run = sys.argv[2] if len(sys.argv) > 2 else "chain"
pairs = [("sgl-fp8", "sgl-bf16"), ("kern", "sgl-fp8"), ("kern", "sgl-bf16")]


def boundary(stage):
    meta = json.load(open(data / "sglang" / stage / "meta.json"))
    t, nb = meta["tokens"], meta["out.residual"]["shape"][1]
    src = {"sgl-fp8": (data / "sglang" / stage / "out.hidden_states", data / "sglang" / stage / "out.residual"),
           "sgl-bf16": (data / "sglang-bf16kv" / stage / "out.hidden_states", data / "sglang-bf16kv" / stage / "out.residual"),
           "kern": (data / "kern" / run / stage / "hidden.bf16", data / "kern" / run / stage / "blocks.bf16")}
    x = {k: (bf16(h, (t, H)), bf16(b, (t, nb, H))) for k, (h, b) in src.items()}
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return {f"{p} | {q}": {"hidden": rel(x[p][0], x[q][0]), "bank": rel(x[p][1], x[q][1])} for p, q in pairs}


def head():
    tops = {}
    for k, f in (("sgl-fp8", data / "sglang" / "generate.json"), ("sgl-bf16", data / "sglang-bf16kv" / "generate.json")):
        tops[k] = {t: l for l, t, _ in json.load(open(f))["meta_info"]["output_top_logprobs"][0]}
    logits = np.fromfile(data / "kern" / run / "l69-93" / "logits.f32", dtype=np.float32).astype(np.float64)
    lp = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
    tops["kern"] = {int(i): float(lp[i]) for i in np.argsort(-lp)[:5]}
    full = {"kern": lambda t: float(lp[t])}
    out = {k: sorted(v.items(), key=lambda kv: -kv[1]) for k, v in tops.items()}
    for p, q in pairs:
        common = set(tops[p]) & set(tops[q]) if "kern" not in (p, q) else set(tops[p if p != "kern" else q])
        get = lambda k, t: full[k](t) if k in full else tops[k].get(t)
        d = [abs(get(p, t) - get(q, t)) for t in common if get(p, t) is not None and get(q, t) is not None]
        out[f"{p} | {q} max|dlogprob|"] = max(d)
    return out


res = {s: boundary(s) for s in ("l0-23", "l23-46", "l46-69")}
res["head"] = head()
for s in ("l0-23", "l23-46", "l46-69"):
    print(s, "  ".join(f"[{k}] hidden {v['hidden']:.3f} bank {v['bank']:.3f}" for k, v in res[s].items()))
for k, v in res["head"].items():
    print("head", k, [(t, round(l, 3)) for t, l in v] if isinstance(v, list) else round(v, 3))
json.dump(res, open(data / "band.json", "w"), indent=1)
