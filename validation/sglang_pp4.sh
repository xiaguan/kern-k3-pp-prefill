#!/usr/bin/env bash
# SGLang pruned K3, TP1 / PP4 on one tray, prefill eager, with the kcap dump plugin.
#   sglang_pp4.sh start|stop|logs ; env MODEL (the checkpoint dir), OUT (the dump root), TOKENS,
#   KV (kv cache dtype, default fp8_e4m3), TAG (dump dir suffix: $OUT/sglang$TAG)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
img=lmsysorg/sglang:nightly-dev-cu13-20261005-f70e8c68
model=${MODEL:?the pruned K3 checkpoint dir}
out=${OUT:?the dump root}
name=sgl-k3-pp4-kcap
dump="$out/sglang${TAG:-}"
case $1 in
start)
  mkdir -p "$dump"
  docker run -d --name $name --gpus all --ipc=host --network host --ulimit memlock=-1:-1 \
    -v "$model:$model" -v "$out:$out" -v "$here:$here" \
    -e PYTHONPATH="$here/kcap" -e SGLANG_PLUGINS=kcap -e KCAP_DIR="$dump" -e KCAP_TOKENS="$TOKENS" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SGLANG_PP_LAYER_PARTITION=23,23,23,24 \
    --entrypoint python3 $img -m sglang.launch_server --model-path $model \
    --trust-remote-code --tp-size 1 --pp-size 4 --chunked-prefill-size 8192 \
    --mm-feature-transport cpu --prefill-attention-backend trtllm_mla \
    --decode-attention-backend cutedsl_mla --moe-a2a-backend none \
    --moe-runner-backend flashinfer_mxfp4 --kv-cache-dtype ${KV:-fp8_e4m3} --page-size 64 \
    --mem-fraction-static 0.80 --mamba-full-memory-ratio 0.5 --max-mamba-cache-size 64 \
    --context-length 32768 --max-running-requests 8 --disable-radix-cache \
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
    --watchdog-timeout 3600 --host 127.0.0.1 --port 30000 ;;
stop) docker stop $name && docker rm $name ;;
logs) docker logs --tail ${N:-40} $name 2>&1 ;;
esac
