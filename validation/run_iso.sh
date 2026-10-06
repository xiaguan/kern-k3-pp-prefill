#!/usr/bin/env bash
# Each stage on its own GPU, fed SGLang's boundary tensors (isolation): kern/${TAG:-iso}/l<A>-<B>/.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data=${DATA:?the run root (ids.i64, sglang/ dumps)}
bin=${K3_STAGE:?kern's k3_stage example binary}
w=${WEIGHTS:?the pruned K3 checkpoint dir}
mkdir -p $data/kern/${TAG:-iso} $data/logs
g=0
for s in 0-23 23-46 46-69 69-93; do
  if [ $s = 0-23 ]; then in=(--tokens $data/ids.i64); else
    in=(--hidden-in $data/sglang/l$s/in.hidden_states --blocks-in $data/sglang/l$s/in.residual); fi
  $bin --manifest $here/../manifests/${MANIFEST_PREFIX:-k3-pruned-pp4}-l$s.json --weights $w --gpu $g "${in[@]}" \
    --out $data/kern/${TAG:-iso}/l$s --iters ${ITERS:-3} > $data/logs/${TAG:-iso}-l$s.log 2>&1 &
  g=$((g + 1))
done
wait
tail -n 3 $data/logs/${TAG:-iso}-*.log
