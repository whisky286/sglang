#!/usr/bin/env bash
# Single-node, disposable TP=DP=EP=4 fault/replay acceptance. Shell/curl/jq only.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
# shellcheck source=test/manual/ep/ft_request_recovery_lib.sh
source "$SCRIPT_DIR/ft_request_recovery_lib.sh"

usage() {
  cat <<'EOF'
Usage: bash test/manual/ep/run_ft_request_recovery.sh [options]
  --device auto|npu|cuda   default auto; NPU uses MC2, CUDA uses Mooncake
  --model PATH            default MODEL_PATH or /mnt/share/weights/DeepSeek-V2-Lite-Chat
  --case all|nonstream|stream|before-first|cancel|repeat
  --port N                default 31000; must be unused
  --tokens N              default 512, at least 128
  --output DIR            new artifact directory; must not exist
Environment:
  FT_READY_TIMEOUT=900 FT_STATUS_TIMEOUT=180 FT_CONTROL_TIMEOUT=600
  FT_REQUEST_TIMEOUT=1800 FT_RECOVERY_TIMEOUT=600
  MOONCAKE_IB_DEVICE=...   optional CUDA RDMA device list (otherwise auto-detect)
  MODEL_PATH=...          preinstalled compatible MoE checkpoint
Requires Linux, four devices, a working SGLang/backend installation, curl, jq,
flock and ss. No installation or SSH. PYTHONPATH selects this checkout.
The script launches its own server per case, injects SIGKILL only into its own
identified Scheduler, and keeps the server/artifacts on failure. It never
stops an existing service. A successful case shuts down its own server.
EOF
}

DEVICE=auto MODEL_PATH=${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}
CASE=all PORT=31000 TOKENS=512 OUTPUT=
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --device|--model|--case|--port|--tokens|--output)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      case "$1" in
        --device) DEVICE=$2 ;; --model) MODEL_PATH=$2 ;; --case) CASE=$2 ;;
        --port) PORT=$2 ;; --tokens) TOKENS=$2 ;; --output) OUTPUT=$2 ;;
      esac
      shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
fail() { log "FAIL: $*" >&2; return 1; }
READY_TIMEOUT=${FT_READY_TIMEOUT:-900}
STATUS_TIMEOUT=${FT_STATUS_TIMEOUT:-180}
CONTROL_TIMEOUT=${FT_CONTROL_TIMEOUT:-600}
REQUEST_TIMEOUT=${FT_REQUEST_TIMEOUT:-1800}
RECOVERY_TIMEOUT=${FT_RECOVERY_TIMEOUT:-600}
for value in "$PORT" "$TOKENS" "$READY_TIMEOUT" "$STATUS_TIMEOUT" "$CONTROL_TIMEOUT" "$REQUEST_TIMEOUT" "$RECOVERY_TIMEOUT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
done
[ "$PORT" -le 65535 ] && [ "$TOKENS" -ge 128 ] && [ "$TOKENS" -le 1536 ] || exit 2
case "$CASE" in all|nonstream|stream|before-first|cancel|repeat) ;; *) usage; exit 2 ;; esac
[ "$(uname -s)" = Linux ] || { fail "requires Linux procfs on the accelerator server"; exit 2; }
for command in curl jq flock ss ps awk sed git python; do command -v "$command" >/dev/null || { fail "missing $command"; exit 2; }; done
if [ "$DEVICE" = auto ]; then
  if command -v npu-smi >/dev/null; then DEVICE=npu
  elif command -v nvidia-smi >/dev/null; then DEVICE=cuda
  else fail "no accelerator detected; choose --device"; exit 2; fi
fi
case "$DEVICE" in npu|cuda) ;; *) usage; exit 2 ;; esac
[ -d "$MODEL_PATH" ] || { fail "checkpoint directory not found: $MODEL_PATH"; exit 2; }
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p artifacts/ft_request_recovery
exec 9> "artifacts/ft_request_recovery/port-$PORT.lock"
flock -n 9 || { fail "another validation owns port $PORT"; exit 2; }
[ -z "$(ss -ltnH "sport = :$PORT")" ] || { fail "port $PORT is already in use"; exit 2; }
OUTPUT=${OUTPUT:-artifacts/ft_request_recovery/run-$(date +%Y%m%d-%H%M%S)-$$}
mkdir "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
BASE_URL="http://127.0.0.1:$PORT"
git rev-parse HEAD > "$OUTPUT/commit.txt"
git diff HEAD > "$OUTPUT/working-tree.diff"
python -m pip freeze > "$OUTPUT/requirements.txt" 2>&1
uname -a > "$OUTPUT/system.txt"
if [ "$DEVICE" = npu ]; then npu-smi info > "$OUTPUT/devices.txt" 2>&1 || true
else nvidia-smi > "$OUTPUT/devices.txt" 2>&1 || true; fi
printenv | LC_ALL=C sort | awk '/^(ASCEND|CANN|HCCL|ACL_|LD_LIBRARY_PATH=|PYTHONPATH=|DEEP_|SGLANG_)/ &&
  tolower($0) !~ /^[^=]*(secret|password|credential|api_key|auth_token|access_token|private_key)[^=]*=/' > "$OUTPUT/runtime-env.txt"
PROMPT='请用中文和 English，分条解释分布式推理、故障恢复及缓存的作用。可以使用🙂，请详细展开。'
SERVER_PID='' SERVER_TICKS='' CASE_DIR='' CLIENT_PIDS=()

on_exit() {
  local code=$?
  trap - EXIT INT TERM
  if [ "$code" -ne 0 ]; then
    for job in "${CLIENT_PIDS[@]}"; do
      if jobs -pr | grep -Fxq "$job"; then kill -TERM "$job" 2>/dev/null || true; fi
    done
    [ -z "$CASE_DIR" ] || ps -eo pid,ppid,lstart,args > "$CASE_DIR/processes-at-failure.txt"
    [ -z "$CASE_DIR" ] || tail -n 200 "$CASE_DIR/server.log" >&2 || true
    jq -n --argjson exit_code "$code" --arg case "$CASE" --arg server_pid "$SERVER_PID" \
      '{success:false,exit_code:$exit_code,case:$case,server_pid:$server_pid}' > "$OUTPUT/result.json"
    log "Artifacts: $OUTPUT; own server PID=$SERVER_PID retained for diagnosis."
  fi
  exit "$code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

status() {
  curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "$BASE_URL/fault_tolerance/status" -o "$CASE_DIR/status.tmp" &&
    mv "$CASE_DIR/status.tmp" "$CASE_DIR/status.json"
}

wait_status() {
  local condition=$1 limit=$2 started=$SECONDS
  while [ $((SECONDS - started)) -lt "$limit" ]; do
    ft_same_process "$SERVER_PID" "$SERVER_TICKS" || { fail "server exited while waiting"; return 1; }
    if status && jq -e "$condition" "$CASE_DIR/status.json" >/dev/null; then return 0; fi
    sleep 0.2
  done
  fail "status timeout: $condition"
}

post() {
  local endpoint=$1 body=$2 output=$3 timeout=${4:-30} http
  http=$(curl --noproxy '*' --connect-timeout 5 --max-time "$timeout" -sS \
    "$BASE_URL/$endpoint" -H 'Content-Type: application/json' -d "$body" -o "$output" -w '%{http_code}') || return 1
  printf '%s\n' "$http" > "$output.http"
  [ "$http" = 200 ] || { fail "$endpoint returned HTTP $http"; return 1; }
}

start_server() {
  local name=$1
  CASE_DIR="$OUTPUT/$name"
  mkdir "$CASE_DIR"
  CLIENT_PIDS=()
  CANCEL_ID=
  local args=(
    --model-path "$MODEL_PATH" --device "$DEVICE" --dtype bfloat16
    --tp-size 4 --dp-size 4 --ep-size 4 --attn-cp-size 1
    --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 --moe-dp-size 1
    --enable-eplb --eplb-algorithm elasticity_aware --ep-dispatch-algorithm dynamic
    --ep-num-redundant-experts 128 --deepep-mode low_latency
    --enable-fault-tolerance --enable-ft-request-recovery --fault-tolerance-on-error-strategy pause
    --fault-tolerance-timeout "$CONTROL_TIMEOUT" --fault-tolerance-pause-timeout "$REQUEST_TIMEOUT"
    --ft-request-timeout "$REQUEST_TIMEOUT" --ft-request-recovery-timeout "$RECOVERY_TIMEOUT"
    --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache
    --incremental-streaming-output --load-balance-method round_robin
    --mem-fraction-static 0.5 --max-running-requests 8 --max-total-tokens 4096
    --context-length 2048 --watchdog-timeout 300 --nnodes 1 --node-rank 0
    --trust-remote-code --host 127.0.0.1 --port "$PORT"
  )
  if [ "$DEVICE" = npu ]; then
    args+=(--attention-backend ascend --elastic-ep-backend mc2 --moe-a2a-backend deepep)
  else
    args+=(--elastic-ep-backend mooncake --moe-a2a-backend mooncake --disable-custom-all-reduce)
    [ -z "${MOONCAKE_IB_DEVICE:-}" ] || args+=(--mooncake-ib-device "$MOONCAKE_IB_DEVICE")
  fi
  printf '%q ' python -m sglang.launch_server "${args[@]}" > "$CASE_DIR/launch-command.txt"
  DEEP_USE_MODE=ops SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0 SGLANG_FORCE_STREAM_INTERVAL=1 \
    python -m sglang.launch_server "${args[@]}" > "$CASE_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  SERVER_TICKS=$(ft_proc_ticks "$SERVER_PID")
  printf '%s %s\n' "$SERVER_PID" "$SERVER_TICKS" > "$CASE_DIR/server.pid"
  log "$name: starting $DEVICE server PID=$SERVER_PID"
  wait_status '(.ranks|length)==4 and all(.ranks[];.state=="healthy") and .request_recovery.active_requests==0' "$READY_TIMEOUT"
  local ready_started=$SECONDS
  until grep -q 'ready to roll' "$CASE_DIR/server.log"; do
    ft_same_process "$SERVER_PID" "$SERVER_TICKS" || { fail "server exited during warmup"; return 1; }
    [ $((SECONDS-ready_started)) -lt "$READY_TIMEOUT" ] || { fail "startup warmup timeout"; return 1; }
    sleep 0.2
  done
  ps -eo pid,ppid,lstart,args > "$CASE_DIR/processes-before.txt"
  local rank pid
  for rank in 0 1 2 3; do
    pid=$(ft_rank_pid "$rank" "$SERVER_PID")
    printf '%s %s %s\n' "$rank" "$pid" "$(ft_proc_ticks "$pid")" >> "$CASE_DIR/scheduler-identities.txt"
  done
  local smoke
  smoke=$(jq -n --arg text "$PROMPT" '{rid:"ft-smoke",text:$text,sampling_params:{temperature:0,max_new_tokens:16}}')
  post generate "$smoke" "$CASE_DIR/smoke.json"
  jq -e '.meta_info.id=="ft-smoke" and .meta_info.finish_reason.type!="abort"' "$CASE_DIR/smoke.json" >/dev/null
  PROMPT_TOKENS=$(jq -r '.meta_info.prompt_tokens' "$CASE_DIR/smoke.json")
  local delay=0.02
  [ "$name" != before-first ] || delay=5
  post slow_down "$(jq -n --argjson delay "$delay" '{forward_sleep_time:$delay}')" "$CASE_DIR/slow-down.json"
}

stop_server() {
  local started=$SECONDS
  ft_same_process "$SERVER_PID" "$SERVER_TICKS" || { fail "own server exited unexpectedly"; return 1; }
  kill -TERM "$SERVER_PID"
  while ft_same_process "$SERVER_PID" "$SERVER_TICKS"; do
    [ $((SECONDS - started)) -lt 45 ] || { fail "own server did not stop"; return 1; }
    sleep 0.2
  done
  wait "$SERVER_PID" || true
  [ -z "$(ss -ltnH "sport = :$PORT")" ] || { fail "port still listening after shutdown"; return 1; }
  SERVER_PID='' SERVER_TICKS=''
}

launch_clients() {
  local stream=$1 i body file
  for i in 0 1 2 3; do
    body=$(jq -n --arg id "ft-$CASE-$i" --arg text "$PROMPT" --argjson stream "$stream" --argjson limit "$TOKENS" \
      '{rid:$id,text:$text,stream:$stream,sampling_params:{temperature:0,max_new_tokens:$limit,ignore_eos:true}}')
    file="$CASE_DIR/client-$i"
    (
      set +e
      CURL_PID=
      trap '[ -z "$CURL_PID" ] || kill -TERM "$CURL_PID" 2>/dev/null; exit 143' TERM INT
      curl --noproxy '*' --connect-timeout 5 --max-time "$REQUEST_TIMEOUT" -N -sS \
        "$BASE_URL/generate" -H 'Content-Type: application/json' -d "$body" \
        -o "$file.body" -w '%{http_code}' > "$file.http" &
      CURL_PID=$!
      wait "$CURL_PID"
      code=$?
      CURL_PID=
      printf '%s\n' "$code" > "$file.exit"
      exit "$code"
    ) &
    CLIENT_PIDS+=("$!")
  done
}

save_prefixes() {
  local round=$1 i
  for i in 0 1 2 3; do
    ft_sse_json "$CASE_DIR/client-$i.body" > "$CASE_DIR/prefix-$round-$i.json"
  done
}

inject_and_shrink() {
  local rank=$1 round=$2 pid ticks body
  pid=$(ft_rank_pid "$rank" "$SERVER_PID") || { fail "cannot uniquely identify own rank $rank"; return 1; }
  ticks=$(ft_proc_ticks "$pid")
  printf '%s %s\n' "$pid" "$ticks" > "$CASE_DIR/victim-$round.pid"
  cp "$CASE_DIR/status.json" "$CASE_DIR/before-fault-$round.json"
  ps -eo pid,ppid,lstart,args > "$CASE_DIR/processes-fault-$round.txt"
  if [ "$CASE" = before-first ]; then
    status
    jq -e '.request_recovery.active_requests==4 and all(.request_recovery.active[];.output_tokens==0)' "$CASE_DIR/status.json" >/dev/null || {
      fail "missed the before-first-token injection window"; return 1;
    }
    cp "$CASE_DIR/status.json" "$CASE_DIR/before-fault-$round.json"
  fi
  log "$CASE: kill own rank $rank PID=$pid, round=$round"
  ft_signal_rank "$pid" "$ticks" "$rank" "$SERVER_PID"
  local condition
  condition="all(.ranks[]; if .rank==$rank then .state==\"dead\" else .state!=\"healthy\" end) and .request_recovery.paused_requests==4"
  wait_status "$condition" "$STATUS_TIMEOUT"
  cp "$CASE_DIR/status.json" "$CASE_DIR/paused-$round.json"
  if [ "$CASE" = cancel ]; then
    CANCEL_ID=$(jq -r --argjson rank "$rank" '.request_recovery.active[] | select(.dp_rank==$rank) | .id' "$CASE_DIR/before-fault-$round.json" | head -n 1)
    [ -n "$CANCEL_ID" ] || { fail "no request proven on victim"; return 1; }
    post abort_request "$(jq -n --arg id "$CANCEL_ID" '{rid:$id}')" "$CASE_DIR/cancel.json"
    wait_status '.request_recovery.active_requests==3' "$STATUS_TIMEOUT"
  fi
  body=$(jq -n --argjson rank "$rank" --argjson timeout "$CONTROL_TIMEOUT" \
    '{instruction:"scale_down",params:{ranks:[$rank],timeout:$timeout}}')
  # A mutation is sent exactly once. An HTTP timeout is retained as a failure.
  post fault_tolerance/apply "$body" "$CASE_DIR/scale-down-$round.json" "$((CONTROL_TIMEOUT * 3 + 30))"
  jq -e '.success==true' "$CASE_DIR/scale-down-$round.json" >/dev/null
  wait_status "([.ranks[] | select(.state==\"healthy\")]|length)==$((4-round))" "$STATUS_TIMEOUT"
  if [ "$CASE" = before-first ]; then
    post slow_down '{"forward_sleep_time":0.02}' "$CASE_DIR/slow-down-after-recovery.json"
  fi
  # Surviving Scheduler identities must be unchanged.
  local survivor expected actual
  for survivor in 0 1 2 3; do
    [ "$survivor" = 1 ] && continue
    [ "$round" -eq 1 ] || { [ "$survivor" != 2 ] || continue; }
    expected=$(awk -v rank="$survivor" '$1==rank {print $2}' "$CASE_DIR/scheduler-identities.txt")
    actual=$(ft_rank_pid "$survivor" "$SERVER_PID")
    [ "$actual" = "$expected" ] || { fail "survivor $survivor changed PID"; return 1; }
    ft_same_process "$actual" "$(awk -v rank="$survivor" '$1==rank {print $3}' "$CASE_DIR/scheduler-identities.txt")" || { fail "survivor $survivor restarted"; return 1; }
  done
}

run_case() {
  CASE=$1
  start_server "$CASE"
  local stream=true
  [ "$CASE" != nonstream ] || stream=false
  launch_clients "$stream"
  if [ "$CASE" = before-first ]; then
    wait_status '.request_recovery.active_requests==4 and all(.request_recovery.active[];.output_tokens==0)' "$STATUS_TIMEOUT"
  else
    wait_status '.request_recovery.active_requests==4 and all(.request_recovery.active[];.output_tokens>=8) and any(.request_recovery.active[];.dp_rank==1)' "$STATUS_TIMEOUT"
  fi
  [ "$stream" = false ] || save_prefixes 1
  inject_and_shrink 1 1
  local attempts=1
  if [ "$CASE" = repeat ]; then
    wait_status '.request_recovery.active_requests==4 and all(.request_recovery.active[];.attempt==1 and .dp_rank!=null) and any(.request_recovery.active[];.dp_rank==2)' "$STATUS_TIMEOUT"
    save_prefixes 2
    inject_and_shrink 2 2
    attempts=2
  fi
  local job i terminal rid
  for job in "${CLIENT_PIDS[@]}"; do wait "$job" || { fail "client failed"; return 1; }; done
  for i in 0 1 2 3; do
    [ "$(cat "$CASE_DIR/client-$i.exit")" = 0 ] && [ "$(cat "$CASE_DIR/client-$i.http")" = 200 ] || { fail "client $i transport error"; return 1; }
    rid="ft-$CASE-$i"
    if [ "$stream" = true ]; then
      ft_sse_json "$CASE_DIR/client-$i.body" > "$CASE_DIR/client-$i.json"
      terminal=length
      [ "$rid" != "${CANCEL_ID:-}" ] || terminal=abort
      ft_check_stream "$CASE_DIR/client-$i.json" "$CASE_DIR/prefix-1-$i.json" "$rid" "$TOKENS" "$attempts" "$terminal"
      if [ "$CASE" = repeat ]; then
        ft_check_stream "$CASE_DIR/client-$i.json" "$CASE_DIR/prefix-2-$i.json" "$rid" "$TOKENS" "$attempts"
      fi
      if [ "$terminal" = length ]; then
        jq -e --argjson n "$PROMPT_TOKENS" '.[-1].meta_info.prompt_tokens==$n' "$CASE_DIR/client-$i.json" >/dev/null
      fi
    else
      jq -e --arg id "$rid" --argjson n "$TOKENS" --argjson p "$PROMPT_TOKENS" '
        .meta_info.id==$id and .meta_info.finish_reason.type=="length" and
        .meta_info.completion_tokens==$n and (.output_ids|length)==$n and
        .meta_info.prompt_tokens==$p and .meta_info.recovery_attempts==1 and .meta_info.dp_rank!=1
      ' "$CASE_DIR/client-$i.body" >/dev/null
    fi
  done
  wait_status '.request_recovery.active_requests==0 and .request_recovery.reserved_tokens==0' "$STATUS_TIMEOUT"
  cp "$CASE_DIR/status.json" "$CASE_DIR/final-status.json"
  local expected=$((attempts * 4))
  [ "$CASE" != cancel ] || expected=3
  jq -e --argjson n "$expected" '.request_recovery.resumed_requests==$n' "$CASE_DIR/final-status.json" >/dev/null
  jq -n --arg case "$CASE" '{success:true,case:$case}' > "$CASE_DIR/result.json"
  log "$CASE: PASS (original clients finished; recovery records released)"
  stop_server
}

if [ "$CASE" = all ]; then
  for selected in nonstream stream before-first cancel repeat; do run_case "$selected"; done
else
  run_case "$CASE"
fi
jq -s '{success:all(.[];.success),cases:.}' "$OUTPUT"/*/result.json > "$OUTPUT/result.json"
log "PASS: $OUTPUT/result.json"
