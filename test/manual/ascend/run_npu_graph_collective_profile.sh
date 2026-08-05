#!/usr/bin/env bash
#
# A7 Stage 1: collect graph-capture and healthy graph-replay profiles for the
# fixed 4 x Atlas A3 masked-original-topology target.
#
# Required:
#   MODEL_PATH=/absolute/model/path \
#   ARTIFACT_ROOT=/absolute/artifact/root \
#   bash test/manual/ascend/run_npu_graph_collective_profile.sh
#
# The script intentionally keeps NPU graph enabled. It profiles a healthy run;
# it does not inject a rank failure or modify any recovery behavior.

set -euo pipefail

FIXED_BASELINE="2f14d6c6f23fc06d59249ce4663b94f8a92ad02e"
PORT="${PORT:-31220}"
HOST="${HOST:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/home/l00893053/Qwen3-30B-A3B}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/tmp/sglang-npu-graph-profile}"
CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-1 2 4 8}"
PROFILE_STEPS="${PROFILE_STEPS:-5}"
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
RUN_GSM8K_EVAL="${RUN_GSM8K_EVAL:-0}"
EPLB_REBALANCE_NUM_ITERATIONS="${EPLB_REBALANCE_NUM_ITERATIONS:-1000}"
EPLB_REBALANCE_LAYERS_PER_CHUNK="${EPLB_REBALANCE_LAYERS_PER_CHUNK:-}"
REQUIRE_FOCUSED_VALIDATION="${REQUIRE_FOCUSED_VALIDATION:-0}"
REQUIRE_COMPLETE_COMMUNICATION_MATRIX="${REQUIRE_COMPLETE_COMMUNICATION_MATRIX:-0}"
SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE="${SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE:-1}"
SGLANG_BIG_TP_COLLECTIVE_TRACE="${SGLANG_BIG_TP_COLLECTIVE_TRACE:-0}"
SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS="${SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS:-0}"
SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS_PER_SOURCE="${SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS_PER_SOURCE:-0}"
SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-0}"
SGLANG_PROFILE_WITH_STACK="${SGLANG_PROFILE_WITH_STACK:-true}"
SGLANG_PROFILE_RECORD_SHAPES="${SGLANG_PROFILE_RECORD_SHAPES:-true}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This experiment requires a Linux Atlas A3 host." >&2
  exit 2
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
for command_name in git python curl grep setsid; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command_name}" >&2
    exit 2
  fi
done
if ! command -v npu-smi >/dev/null 2>&1; then
  echo "npu-smi is required to fingerprint the Atlas environment." >&2
  exit 2
fi

IFS=',' read -r -a VISIBLE_DEVICE_LIST <<<"${ASCEND_RT_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_DEVICE_LIST[@]}" -ne 4 ]]; then
  echo "Exactly four visible NPUs are required; got ${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 2
fi
read -r -a GRAPH_BUCKETS <<<"${CUDA_GRAPH_BS_DECODE}"
if [[ "${#GRAPH_BUCKETS[@]}" -eq 0 ]]; then
  echo "CUDA_GRAPH_BS_DECODE must contain at least one batch size." >&2
  exit 2
fi
for marker in GRAPH_WARMUP GRAPH_CAPTURE GRAPH_REPLAY; do
  if ! grep -q -- "${marker}" \
    python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py \
    python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py; then
    echo "Required profiler marker is missing from the source: ${marker}" >&2
    exit 2
  fi
done

OUTPUT_DIR="${ARTIFACT_ROOT%/}/${RUN_ID}"
CAPTURE_DIR="${OUTPUT_DIR}/stage-1a-capture"
REPLAY_DIR="${OUTPUT_DIR}/stage-1b-healthy-replay"
MANIFEST_DIR="${OUTPUT_DIR}/stage-0-manifest"
SERVER_LOG="${OUTPUT_DIR}/server.log"
CLIENT_LOG="${OUTPUT_DIR}/healthy-replay-client.log"
EVAL_LOG="${OUTPUT_DIR}/gsm8k-eval.log"
mkdir -p "${CAPTURE_DIR}" "${REPLAY_DIR}" "${MANIFEST_DIR}"

export ASCEND_RT_VISIBLE_DEVICES
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export SGLANG_TORCH_PROFILER_DIR="${CAPTURE_DIR}"
export SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE
export SGLANG_BIG_TP_COLLECTIVE_TRACE
export SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS
export SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS_PER_SOURCE
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH
export SGLANG_PROFILE_WITH_STACK
export SGLANG_PROFILE_RECORD_SHAPES

SERVER_ARGS=(
  --model-path "${MODEL_PATH}"
  --device npu
  --attention-backend ascend
  --dtype bfloat16
  --tp-size 4
  --dp-size 4
  --ep-size 4
  --enable-dp-attention
  --enable-dp-lm-head
  --moe-dense-tp-size 1
  --moe-dp-size 1
  --moe-a2a-backend deepep
  --deepep-mode low_latency
  --enable-eplb
  --eplb-algorithm elasticity_aware
  --ep-dispatch-algorithm dynamic
  --ep-num-redundant-experts 32
  --eplb-rebalance-num-iterations "${EPLB_REBALANCE_NUM_ITERATIONS}"
  --enable-profile-cuda-graph
  --cuda-graph-bs-decode "${GRAPH_BUCKETS[@]}"
  --disable-radix-cache
  --mem-fraction-static 0.7
  --max-running-requests 8
  --nnodes 1
  --node-rank 0
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
)
if [[ -n "${EPLB_REBALANCE_LAYERS_PER_CHUNK}" ]]; then
  SERVER_ARGS+=(
    --eplb-rebalance-layers-per-chunk "${EPLB_REBALANCE_LAYERS_PER_CHUNK}"
  )
fi
for argument in "${SERVER_ARGS[@]}"; do
  if [[ "${argument}" == "--disable-cuda-graph" ]]; then
    echo "Refusing to run with NPU graph disabled." >&2
    exit 2
  fi
done

{
  echo "fixed_sglang_baseline=${FIXED_BASELINE}"
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  echo "origin_main=$(git rev-parse --verify origin/main 2>/dev/null || echo unavailable)"
  echo "model_path=${MODEL_PATH}"
  echo "visible_npus=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "graph_buckets_decode=${CUDA_GRAPH_BS_DECODE}"
  echo "profile_steps=${PROFILE_STEPS}"
  echo "profile_batch_size=${PROFILE_BATCH_SIZE}"
  echo "gsm8k_eval=${RUN_GSM8K_EVAL}"
  echo "eplb=enabled"
  echo "eplb_rebalance_num_iterations=${EPLB_REBALANCE_NUM_ITERATIONS}"
  echo "eplb_rebalance_layers_per_chunk=${EPLB_REBALANCE_LAYERS_PER_CHUNK:-all}"
  echo "eplb_p2p_batch_chunk_size=${SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE}"
  echo "mlp_sync_device_all_gather=${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH}"
  echo "collective_trace=${SGLANG_BIG_TP_COLLECTIVE_TRACE}"
  echo "collective_trace_max_records=${SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS}"
  echo "collective_trace_max_records_per_source=${SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS_PER_SOURCE}"
  echo "moe_backend=deepep"
  echo "deepep_mode=low_latency"
  echo "deepep_num_max_dispatch_tokens_per_rank=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-unset}"
  echo "moe_tp=1"
  echo "moe_dp=1"
  echo "artifact_dir=${OUTPUT_DIR}"
} >"${MANIFEST_DIR}/experiment.txt"
git status --short --branch >"${MANIFEST_DIR}/git-status.txt"
git log --oneline "${FIXED_BASELINE}..HEAD" >"${MANIFEST_DIR}/applied-commits.txt"
git diff --no-ext-diff --binary >"${MANIFEST_DIR}/working-tree.patch"
git diff --cached --no-ext-diff --binary >"${MANIFEST_DIR}/index.patch"
git ls-files --others --exclude-standard >"${MANIFEST_DIR}/untracked-files.txt"
printf '%q ' python -m sglang.launch_server "${SERVER_ARGS[@]}" \
  >"${MANIFEST_DIR}/server-command.txt"
printf '\n' >>"${MANIFEST_DIR}/server-command.txt"
for rank in 0 1 2 3; do
  echo "original_rank_${rank}=visible_device_${VISIBLE_DEVICE_LIST[${rank}]}"
done >"${MANIFEST_DIR}/rank-device-map.txt"
env |
  grep -E '^(ASCEND|CANN|HCCL|SGLANG|PYTHON|LD_LIBRARY_PATH|PATH)=' |
  sort >"${MANIFEST_DIR}/runtime-environment.txt"
npu-smi info >"${MANIFEST_DIR}/npu-smi-info.txt" 2>&1 || true
for device_id in "${VISIBLE_DEVICE_LIST[@]}"; do
  npu-smi info -t board -i "${device_id}" \
    >>"${MANIFEST_DIR}/npu-board-info.txt" 2>&1 || true
done
for version_file in \
  "${ASCEND_HOME_PATH:-}/version.cfg" \
  "${ASCEND_HOME_PATH:-}/compiler/version.info" \
  /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
  /usr/local/Ascend/ascend-toolkit/latest/compiler/version.info; do
  if [[ -f "${version_file}" ]]; then
    echo "===== ${version_file} ====="
    sed -n '1,120p' "${version_file}"
  fi
done >"${MANIFEST_DIR}/cann-version-files.txt"
python -m pip freeze >"${MANIFEST_DIR}/pip-freeze.txt"
python - <<'PY' >"${MANIFEST_DIR}/software-versions.txt"
import importlib.metadata
import platform

import deep_ep
import sgl_kernel_npu
import sglang
import torch
import torch_npu

print(f"platform={platform.platform()}")
print(f"python={platform.python_version()}")
print(f"sglang_file={sglang.__file__}")
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"deep_ep_path={list(deep_ep.__path__)}")
print(f"sgl_kernel_npu_path={list(sgl_kernel_npu.__path__)}")
for distribution in ("sgl-kernel-npu", "torch-npu", "torch"):
    try:
        print(f"{distribution}={importlib.metadata.version(distribution)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{distribution}=not-installed-as-distribution")
PY

SERVER_PID=""
cleanup() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi

  # The launcher and all scheduler workers run in an isolated process group.
  # Clean the group even if its leader was already OOM-killed, otherwise orphan
  # workers can retain NPU memory and make the next run fail immediately.
  if kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

setsid python -m sglang.launch_server "${SERVER_ARGS[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" >"${OUTPUT_DIR}/server.pid"
echo "${SERVER_PID}" >"${OUTPUT_DIR}/server.pgid"

READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited before readiness; see ${SERVER_LOG}" >&2
    exit 1
  fi
  if curl --silent --show-error --fail \
    "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 5
done
if [[ "${READY}" -ne 1 ]]; then
  echo "Server did not become ready; see ${SERVER_LOG}" >&2
  exit 1
fi

curl --silent --show-error --fail "http://${HOST}:${PORT}/server_info" \
  >"${MANIFEST_DIR}/resolved-server-info.json"

if [[ "${RUN_GSM8K_EVAL}" == "1" ]]; then
  python -m sglang.test.run_eval \
    --host "${HOST}" \
    --port "${PORT}" \
    --eval-name gsm8k \
    --num-examples 20 >"${EVAL_LOG}" 2>&1
  python - "${EVAL_LOG}" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
scores = re.findall(r"^Score:\s+([0-9.]+)\s*$", text, flags=re.MULTILINE)
if not scores:
    raise SystemExit("GSM8K sanity check did not report a score")
score = float(scores[-1])
if score <= 0.8:
    raise SystemExit(f"GSM8K sanity score must be > 0.8 before profiling; got {score}")
PY
fi

SGLANG_TORCH_PROFILER_DIR="${REPLAY_DIR}" \
  python -m sglang.test.send_one \
  --host "${HOST}" \
  --port "${PORT}" \
  --profile \
  --profile-steps "${PROFILE_STEPS}" \
  --profile-by-stage \
  --profile-prefix healthy-replay \
  --batch-size "${PROFILE_BATCH_SIZE}" \
  --different-prompts \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --seed 0 \
  --temperature 0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" >"${CLIENT_LOG}" 2>&1

cleanup
SERVER_PID=""
trap - EXIT INT TERM

python - "${CAPTURE_DIR}" "${REPLAY_DIR}" \
  >"${OUTPUT_DIR}/marker-summary.json" <<'PY'
import gzip
import json
import pathlib
import sys

markers = ("GRAPH_WARMUP", "GRAPH_CAPTURE", "GRAPH_REPLAY")


def read_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def count_markers(path):
    counts = {marker: 0 for marker in markers}
    carry = ""
    try:
        with read_text(path) as file:
            while chunk := file.read(1024 * 1024):
                for marker in markers:
                    counts[marker] += chunk.count(marker)
                    counts[marker] += sum(
                        carry.endswith(marker[:split])
                        and chunk.startswith(marker[split:])
                        for split in range(1, len(marker))
                    )
                carry = (carry + chunk)[-(max(map(len, markers)) - 1) :]
    except (OSError, UnicodeError):
        return None
    return counts


summary = {}
for label, root_arg in zip(("capture", "healthy_replay"), sys.argv[1:]):
    root = pathlib.Path(root_arg)
    counts = {marker: 0 for marker in markers}
    scanned = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".gz"}:
            continue
        file_counts = count_markers(path)
        if file_counts is None:
            continue
        scanned.append(str(path))
        for marker in markers:
            counts[marker] += file_counts[marker]
    summary[label] = {"marker_counts": counts, "scanned_files": scanned}

print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["capture"]["marker_counts"]["GRAPH_CAPTURE"] == 0:
    raise SystemExit("GRAPH_CAPTURE marker not found in capture profile")
if summary["healthy_replay"]["marker_counts"]["GRAPH_REPLAY"] == 0:
    raise SystemExit("GRAPH_REPLAY marker not found in healthy replay profile")
PY

if [[ "${REQUIRE_COMPLETE_COMMUNICATION_MATRIX}" == "1" ]]; then
  set +e
  python test/manual/ascend/analyze_npu_communication_matrix.py \
    "${OUTPUT_DIR}" --require-complete \
    >"${OUTPUT_DIR}/communication-matrix-summary.json"
  COMMUNICATION_MATRIX_STATUS=$?
  set -e
  cat "${OUTPUT_DIR}/communication-matrix-summary.json"
  if [[ "${COMMUNICATION_MATRIX_STATUS}" -ne 0 ]]; then
    echo "Complete communication inventory still has missing or unclassified entries." >&2
    exit "${COMMUNICATION_MATRIX_STATUS}"
  fi
else
  FOCUSED_VALIDATION_ARGS=()
  if [[ "${REQUIRE_FOCUSED_VALIDATION}" == "1" ]]; then
    FOCUSED_VALIDATION_ARGS+=(--require-complete)
  fi
  set +e
  python test/manual/ascend/analyze_npu_lm_head_eplb_validation.py \
    "${OUTPUT_DIR}" "${FOCUSED_VALIDATION_ARGS[@]}" \
    >"${OUTPUT_DIR}/graph-membership-summary.json"
  FOCUSED_VALIDATION_STATUS=$?
  set -e
  cat "${OUTPUT_DIR}/graph-membership-summary.json"
  if [[ "${FOCUSED_VALIDATION_STATUS}" -ne 0 ]]; then
    echo "Focused MLP-sync/EPLB graph-membership validation did not meet all required conditions." >&2
    exit "${FOCUSED_VALIDATION_STATUS}"
  fi
fi

printf '%s\n' \
  "ID,Phase,Source symbol,Backend/op,Group members,Shape/dtype,Stream,Buffer address,Contacts all original ranks?,Initial recovery classification" \
  >"${OUTPUT_DIR}/communication-matrix.csv"

echo "A7 Stage 1 artifacts: ${OUTPUT_DIR}"
