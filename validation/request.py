"""Send the prompt once for the dump (top-5 logprobs saved), then 3 times for the warm TTFT.

    request.py <ids.i64> <out generate.json>
"""
import json
import struct
import sys
import urllib.request

ids_path, out = sys.argv[1], sys.argv[2]
raw = open(ids_path, "rb").read()
ids = list(struct.unpack(f"<{len(raw) // 8}q", raw))


def generate(body):
    req = urllib.request.Request("http://127.0.0.1:30000/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


r = generate({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
              "return_logprob": True, "top_logprobs_num": 5, "logprob_start_len": len(ids) - 1})
json.dump(r, open(out, "w"), indent=1)
print("output_ids", r["output_ids"], "top5", r["meta_info"]["output_top_logprobs"][0])
for i in range(3):
    r = generate({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
    print(i, r["output_ids"], "e2e %.3f s" % r["meta_info"]["e2e_latency"])
