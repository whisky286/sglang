#!/usr/bin/env bash
#
# Healthy-state launcher for understanding the effect of
# SGLANG_SCHEDULER_SKIP_ALL_GATHER.
#
# Run this file with `bash`, not `source`. Start baseline and skip sequentially
# because both runs use the same four NPUs.
#
#   PORT=31210 bash test/manual/ascend/run_scheduler_skip_understanding_server.sh baseline
#   PORT=31211 bash test/manual/ascend/run_scheduler_skip_understanding_server.sh skip
#
# This is deliberately not a fault-tolerance experiment:
# - all ranks remain healthy;
# - no process is killed;
# - --enable-fault-tolerance is not enabled.

MODE="${1:-baseline}"
PORT="${PORT:-31210}"
MODEL_PATH="${MODEL_PATH:-/home/l00893053/Qwen3-30B-A3B}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/npu_eep/scheduler_skip_understanding}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"

if [[ "${MODE}" == "skip" ]]; then
  export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
else
  MODE="baseline"
  unset SGLANG_SCHEDULER_SKIP_ALL_GATHER
fi

# Keep the user's external proxy, but make every localhost request bypass it.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export ASCEND_RT_VISIBLE_DEVICES

mkdir -p "${ARTIFACT_ROOT}/${MODE}"
LOG_FILE="${ARTIFACT_ROOT}/${MODE}/server.log"

echo "mode=${MODE}"
echo "port=${PORT}"
echo "model_path=${MODEL_PATH}"
echo "visible_npus=${ASCEND_RT_VISIBLE_DEVICES}"
echo "skip_all_gather=${SGLANG_SCHEDULER_SKIP_ALL_GATHER:-0}"
echo "server_log=${LOG_FILE}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --device npu \
  --attention-backend ascend \
  --dtype bfloat16 \
  --tp-size 4 \
  --dp-size 4 \
  --ep-size 4 \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --disable-cuda-graph \
  --disable-radix-cache \
  --mem-fraction-static 0.7 \
  --max-running-requests 8 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "${PORT}" \
  2>&1 | tee "${LOG_FILE}"
