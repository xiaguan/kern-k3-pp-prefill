"""Tokenize prose from kern's English docs to exactly N tokens; write ids (i64) and the text.

    prompt.py <tokenizer dir> <N> <out dir> <kern checkout>
"""
import sys

import numpy as np
from transformers import AutoTokenizer

model, n, out, kern = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
docs = ["docs/vllm.md", "docs/vllm-rsi-example.md", "docs/qwen38-vllm.md"]
text = "\n\n".join(open(f"{kern}/{d}").read() for d in docs)
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ids = tok.encode(text, add_special_tokens=False)
assert len(ids) >= n, len(ids)
np.array(ids[:n], dtype=np.int64).tofile(out + "/ids.i64")
open(out + "/prompt.txt", "w").write(tok.decode(ids[:n]))
print(len(ids), "->", n)
