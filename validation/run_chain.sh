#!/usr/bin/env bash
# kern's four stages chained, each fed the previous kern stage's files: kern/${TAG:-chain}/l<A>-<B>/.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data=${DATA:?the run root (ids.i64, sglang/ dumps)}
bin=${K3_STAGE:?the k3_stage binary (stage/)}
w=${WEIGHTS:?the pruned K3 checkpoint dir}
mkdir -p $data/kern/${TAG:-chain} $data/logs
prev=
for s in 0-23 23-46 46-69 69-93; do
  if [ -z "$prev" ]; then in=(--tokens $data/ids.i64); else
    in=(--hidden-in $prev/hidden.bf16 --blocks-in $prev/blocks.bf16); fi
  $bin --manifest $here/../manifests/${MANIFEST_PREFIX:-k3-pruned-pp4}-l$s.json --weights $w --gpu ${GPU:-0} "${in[@]}" \
    --out $data/kern/${TAG:-chain}/l$s > $data/logs/${TAG:-chain}-l$s.log 2>&1
  tail -n 2 $data/logs/${TAG:-chain}-l$s.log
  prev=$data/kern/${TAG:-chain}/l$s
done
