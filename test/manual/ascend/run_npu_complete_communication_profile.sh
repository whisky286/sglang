#!/usr/bin/env bash
#
# One comprehensive run for the fixed 4 x Atlas A3 topology. The run profiles
# all four ranks, exercises prefill/decode, forces one healthy EPLB rebalance,
# and fails if any observed communication cannot be source-attributed and
# classified as graph-internal or graph-external.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/npu_eep}" \
RUN_ID="${RUN_ID:-complete-$(date -u +%Y%m%dT%H%M%SZ)}" \
RUN_GSM8K_EVAL=0 \
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-4}" \
PROFILE_STEPS="${PROFILE_STEPS:-12}" \
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}" \
EPLB_REBALANCE_NUM_ITERATIONS="${EPLB_REBALANCE_NUM_ITERATIONS:-16}" \
REQUIRE_FOCUSED_VALIDATION=0 \
REQUIRE_COMPLETE_COMMUNICATION_MATRIX=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}" \
SGLANG_BIG_TP_COLLECTIVE_TRACE=1 \
SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS="${SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS:-20000}" \
SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS_PER_SOURCE="${SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS_PER_SOURCE:-64}" \
SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1 \
SGLANG_PROFILE_WITH_STACK=false \
SGLANG_PROFILE_RECORD_SHAPES=false \
bash "${SCRIPT_DIR}/run_npu_graph_collective_profile.sh"
