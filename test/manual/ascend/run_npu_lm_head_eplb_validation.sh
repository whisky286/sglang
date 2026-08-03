#!/usr/bin/env bash
#
# A7 focused validation:
#   1. prove whether the fixed target executes an LM-head vocab gather;
#   2. capture one healthy EPLB rebalance and its weight P2P movement.
#   3. prove that MLP scheduler sync uses device HCCL all-gather on NPU.
#
# No rank is killed and no raw artifact leaves the NPU host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_GSM8K_EVAL="${RUN_GSM8K_EVAL:-0}" \
PROFILE_STEPS="${PROFILE_STEPS:-12}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}" \
EPLB_REBALANCE_NUM_ITERATIONS="${EPLB_REBALANCE_NUM_ITERATIONS:-16}" \
REQUIRE_FOCUSED_VALIDATION="${REQUIRE_FOCUSED_VALIDATION:-1}" \
SGLANG_BIG_TP_COLLECTIVE_TRACE="${SGLANG_BIG_TP_COLLECTIVE_TRACE:-1}" \
SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS="${SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS:-10000}" \
SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-1}" \
SGLANG_PROFILE_WITH_STACK="${SGLANG_PROFILE_WITH_STACK:-false}" \
SGLANG_PROFILE_RECORD_SHAPES="${SGLANG_PROFILE_RECORD_SHAPES:-false}" \
bash "${SCRIPT_DIR}/run_npu_graph_collective_profile.sh"
