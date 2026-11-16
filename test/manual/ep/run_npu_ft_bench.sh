#!/usr/bin/env bash
# Disposable, single-node TP=DP=EP=4 experiments. Run from the NPU checkout.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
# shellcheck source=npu_ft_bench_lib.sh
source "$SCRIPT_DIR/npu_ft_bench_lib.sh"

usage() {
  cat <<'EOF'
Usage: bash test/manual/ep/run_npu_ft_bench.sh MODE [options]
Modes:
  cycle     One fault -> wait for dead/unhealthy -> shrink -> rejoin check
  recovery  Paired whole-server restart / fault-scale-down comparison
  capacity  Staircase load at four-before / three / four-rejoined
  drain     Paired long stream without / with rejoin; separate drain timeout
  all       Run recovery, capacity, drain in that order
Options:
  --rounds N        independent repetitions (default 1; use 5 for formal runs)
  --output DIR      new output directory (must not exist)
  --rates CSV       capacity rate grid (default 0.1,0.2,0.4,0.8,1.6,3.2)
  --window SEC      post-fault observation window (default 600)
  --step SEC        minimum capacity window (default 120; max 600)
  --min-samples N   desired normal samples per capacity point (default 200)
Environment:
  MODEL_PATH        default /mnt/share/weights/DeepSeek-V2-Lite-Chat
  FT_PORT           default 31000 (must be unused; existing servers untouched)
  FT_LOAD_RATE      common request rate for recovery arms (default 0.2)
  FT_FAULT_TOKENS   trigger on this many output tokens (default 1)
  FT_DRAIN_TOKENS   rejoin after this many output tokens (default 256)
  FT_TTFT_LIMIT / FT_TPOT_LIMIT  capacity gates in seconds; set both or neither
  FT_REQUEST_TIMEOUT=120 FT_STATUS_TIMEOUT=120 FT_CONTROL_TIMEOUT=600
  FT_MAX_SEND_LAG=0.25 FT_MAX_CLIENTS=256
  FT_MAX_RUNNING_REQUESTS=8 FT_MAX_TOTAL_TOKENS=4096
The script starts its own server and automatically resolves rank-1's PID.
No SSH, package installation, global pkill, automatic mutation retries or Python
helper is used. On failure, keep the server and all artifacts for diagnosis.
EOF
}

MODE=${1:-help}; [ "$#" -eq 0 ] || shift
ROUNDS=1; OUTPUT=; RATES=0.1,0.2,0.4,0.8,1.6,3.2
WINDOW=600; STEP=120; MIN_SAMPLES=200
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rounds|--output|--rates|--window|--step|--min-samples)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      case "$1" in
        --rounds) ROUNDS=$2 ;; --output) OUTPUT=$2 ;; --rates) RATES=$2 ;;
        --window) WINDOW=$2 ;; --step) STEP=$2 ;; --min-samples) MIN_SAMPLES=$2 ;;
      esac; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) ft_fail "unknown option: $1"; exit 2 ;;
  esac
done
case "$MODE" in help|-h|--help) usage; exit 0 ;; cycle|recovery|capacity|drain|all) ;; *) usage; exit 2 ;; esac

MODEL_PATH=${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}
FT_PORT=${FT_PORT:-31000}; BASE_URL="http://127.0.0.1:$FT_PORT"
FAULT_RANK=1; FAULT_AFTER_TOKENS=${FT_FAULT_TOKENS:-1}; DRAIN_AFTER_TOKENS=${FT_DRAIN_TOKENS:-256}
REQUEST_TIMEOUT=${FT_REQUEST_TIMEOUT:-120}; STATUS_TIMEOUT=${FT_STATUS_TIMEOUT:-120}
CONTROL_TIMEOUT=${FT_CONTROL_TIMEOUT:-600}; LOAD_RATE=${FT_LOAD_RATE:-0.2}
MAX_SEND_LAG=${FT_MAX_SEND_LAG:-0.25}; MAX_CLIENTS=${FT_MAX_CLIENTS:-256}
MAX_RUNNING=${FT_MAX_RUNNING_REQUESTS:-8}; MAX_TOTAL=${FT_MAX_TOTAL_TOKENS:-4096}
TTFT_LIMIT=${FT_TTFT_LIMIT:-}; TPOT_LIMIT=${FT_TPOT_LIMIT:-}
SERVER_PID=; SERVER_TICKS=; SERVER_DIR=; CASE_DIR=; LOAD_JOB=; TRIGGER_JOB=; PROBE_JOB=; LOG_JOB=
for value in "$ROUNDS" "$WINDOW" "$STEP" "$MIN_SAMPLES" "$REQUEST_TIMEOUT" "$STATUS_TIMEOUT" \
  "$CONTROL_TIMEOUT" "$FAULT_AFTER_TOKENS" "$DRAIN_AFTER_TOKENS" "$MAX_CLIENTS" "$MAX_RUNNING" "$MAX_TOTAL" "$FT_PORT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { ft_fail "expected positive integer: $value"; exit 2; }
done
[ "$FAULT_AFTER_TOKENS" -lt 1536 ] && [ "$DRAIN_AFTER_TOKENS" -lt 1536 ] && [ "$STEP" -le 600 ] || exit 2
[ "$FT_PORT" -le 65535 ] || exit 2
if [ -n "$TTFT_LIMIT" ] || [ -n "$TPOT_LIMIT" ]; then
  jq -en --arg a "$TTFT_LIMIT" --arg b "$TPOT_LIMIT" \
    '($a|tonumber)>0 and ($b|tonumber)>0' >/dev/null
fi
jq -en --arg rates "$RATES" --arg rate "$LOAD_RATE" --arg lag "$MAX_SEND_LAG" '
  ($rates|split(",")|map(tonumber)) as $r |
  ($r|length)>0 and all($r[];.>0) and ($r|unique|sort)==$r and
  ($rate|tonumber)>0 and ($lag|tonumber)>0' >/dev/null
[ "$(uname -s)" = Linux ] && [ -r /proc/uptime ] || { ft_fail "requires Linux procfs on the NPU host"; exit 2; }
for command in curl jq awk ps flock ss sha256sum python git; do command -v "$command" >/dev/null || exit 2; done
[ -d "$MODEL_PATH" ] || { ft_fail "model not found: $MODEL_PATH"; exit 2; }
cd "$REPO_DIR"
mkdir -p artifacts/npu_ft
exec 9> "artifacts/npu_ft/bench-port-$FT_PORT.lock"
flock -n 9 || { ft_fail "another benchmark owns port $FT_PORT"; exit 2; }
OUTPUT=${OUTPUT:-artifacts/npu_ft/bench-$(date +%Y%m%d-%H%M%S)-$$}
mkdir "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
INPUT_IDS="$OUTPUT/input-ids.json"
git rev-parse HEAD > "$OUTPUT/commit.txt"
git diff HEAD > "$OUTPUT/working-tree.diff"
python -m pip freeze > "$OUTPUT/requirements.txt" 2>&1
uname -a > "$OUTPUT/system.txt"
if command -v npu-smi >/dev/null; then npu-smi info > "$OUTPUT/npu-smi.txt" 2>&1 || true; fi
printenv | LC_ALL=C sort | awk '/^(ASCEND|CANN|HCCL|ACL_|LD_LIBRARY_PATH=|PYTHONPATH=|DEEP_|SGLANG_)/ &&
  tolower($0) !~ /^[^=]*(secret|password|credential|api_key|auth_token|access_token|private_key)[^=]*=/' > "$OUTPUT/runtime-env.txt"
jq -n --arg mode "$MODE" --arg model "$MODEL_PATH" --arg base_url "$BASE_URL" \
  --argjson rounds "$ROUNDS" --arg rates "$RATES" --argjson window "$WINDOW" \
  --argjson step "$STEP" --argjson samples "$MIN_SAMPLES" --arg rate "$LOAD_RATE" \
  --argjson fault_tokens "$FAULT_AFTER_TOKENS" --argjson drain_tokens "$DRAIN_AFTER_TOKENS" \
  --argjson max_running "$MAX_RUNNING" --argjson max_total "$MAX_TOTAL" \
  --argjson request_timeout "$REQUEST_TIMEOUT" --argjson status_timeout "$STATUS_TIMEOUT" \
  --argjson control_timeout "$CONTROL_TIMEOUT" --argjson max_lag "$MAX_SEND_LAG" --argjson clients "$MAX_CLIENTS" \
  '{mode:$mode,model:$model,base_url:$base_url,rounds:$rounds,rates:($rates|split(",")|map(tonumber)),
    window:$window,step:$step,min_samples:$samples,load_rate:($rate|tonumber),
    fault_after_tokens:$fault_tokens,drain_after_tokens:$drain_tokens,
    max_running_requests:$max_running,max_total_tokens:$max_total,
    request_timeout_seconds:$request_timeout,status_timeout_seconds:$status_timeout,
    control_timeout_seconds:$control_timeout,max_send_lag_seconds:$max_lag,max_clients:$clients,
    clock:"Linux /proc/uptime",clock_resolution_seconds:0.01,
    fault_type:"scheduler process SIGKILL",input_tokens:256,output_tokens:256,
    temperature:0,ignore_eos:true,client_retries:false}' > "$OUTPUT/manifest.json"

ft_stop_job() {
  # Take identity snapshots, then stop children before their shell parent.
  local job=$1 pid ppid stat args ticks rows
  ticks=$(cat "$OUTPUT/client-job-$job.start" 2>/dev/null) || return 0
  ft_identity_matches "$job" "$ticks" || return 0
  rows=$(ps -e -o pid= -o ppid= -o stat= -o args= | ft_descendants "$job")
  while read -r pid ppid stat args; do
    [ -n "$pid" ] && [ "$pid" != "$job" ] || continue
    ticks=$(ft_start_ticks "$pid" 2>/dev/null) || continue
    if ft_identity_matches "$pid" "$ticks"; then kill -TERM "$pid" 2>/dev/null || true; fi
  done <<< "$rows"
  ticks=$(ft_start_ticks "$job" 2>/dev/null) || ticks=
  if [ -n "$ticks" ] && ft_identity_matches "$job" "$ticks"; then kill -TERM "$job" 2>/dev/null || true; fi
  wait "$job" 2>/dev/null || true
}
stop_clients() {
  local job
  for job in "$LOAD_JOB" "$TRIGGER_JOB" "$PROBE_JOB" "$LOG_JOB"; do
    [ -z "$job" ] || ft_stop_job "$job"
  done
  LOAD_JOB=; TRIGGER_JOB=; PROBE_JOB=; LOG_JOB=
}
on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  stop_clients
  if [ "$rc" -ne 0 ]; then
    if [ -n "$CASE_DIR" ]; then
      jq -n --argjson exit_code "$rc" --argjson at "$(ft_now)" \
        '{success:false,exit_code:$exit_code,observed_at:$at}' > "$CASE_DIR/failure.json"
      curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS "$BASE_URL/fault_tolerance/status" \
        > "$CASE_DIR/status-at-failure.json" 2> "$CASE_DIR/status-at-failure.stderr" || true
      [ -z "$SERVER_DIR" ] || tail -n 200 "$SERVER_DIR/server.log" > "$CASE_DIR/server-tail-at-failure.log" || true
    fi
    printf 'FAILED: artifacts retained at %s\nServer retained: PID=%s log=%s/server.log\n' \
      "$OUTPUT" "$SERVER_PID" "$SERVER_DIR" >&2
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_stop() {
  local deadline pid ppid stat args ticks
  [ -n "$SERVER_PID" ] || return 0
  ft_identity_matches "$SERVER_PID" "$SERVER_TICKS" || \
    { ft_fail "server identity lost; refusing automatic process cleanup"; return 1; }
  # Capture children before TERM; identity checks prevent PID-reuse kills.
  ps -e -o pid= -o ppid= -o stat= -o args= | ft_descendants "$SERVER_PID" > "$SERVER_DIR/stop-tree.txt"
  : > "$SERVER_DIR/stop-identities.tsv"
  while read -r pid ppid stat args; do
    ticks=$(ft_start_ticks "$pid" 2>/dev/null) || continue
    printf '%s\t%s\n' "$pid" "$ticks" >> "$SERVER_DIR/stop-identities.tsv"
  done < "$SERVER_DIR/stop-tree.txt"
  if ft_identity_matches "$SERVER_PID" "$SERVER_TICKS"; then kill -TERM "$SERVER_PID"; fi
  deadline=$((SECONDS + 30))
  while ft_identity_matches "$SERVER_PID" "$SERVER_TICKS" && [ "$SECONDS" -lt "$deadline" ]; do sleep 0.2; done
  while read -r pid ticks; do
    if ft_identity_matches "$pid" "$ticks"; then kill -KILL "$pid" 2>/dev/null || true; fi
  done < "$SERVER_DIR/stop-identities.tsv"
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=; SERVER_TICKS=
}
server_start() {
  local label=$1 deadline code
  [ -z "$SERVER_PID" ] || server_stop
  if ss -H -ltn "sport = :$FT_PORT" | awk 'END{exit !(NR>0)}'; then
    ft_fail "port $FT_PORT already in use; existing service was not stopped"; return 1
  fi
  SERVER_DIR="$CASE_DIR/$label"; mkdir -p "$SERVER_DIR"
  local launch=(python -m sglang.launch_server --model-path "$MODEL_PATH"
    --device npu --attention-backend ascend --dtype bfloat16
    --tp-size 4 --dp-size 4 --ep-size 4 --attn-cp-size 1
    --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 --moe-dp-size 1
    --moe-a2a-backend deepep --deepep-mode low_latency
    --enable-eplb --eplb-algorithm elasticity_aware --ep-dispatch-algorithm dynamic
    --ep-num-redundant-experts 128 --elastic-ep-backend mc2
    --enable-fault-tolerance --enable-npu-ft-recovery --fault-tolerance-on-error-strategy pause
    --fault-tolerance-timeout "$CONTROL_TIMEOUT" --fault-tolerance-pause-timeout 600
    --cuda-graph-bs-decode 1 2 4 8 --disable-cuda-graph --disable-overlap-schedule
    --disable-radix-cache --mem-fraction-static 0.5 --max-running-requests "$MAX_RUNNING"
    --max-total-tokens "$MAX_TOTAL" --context-length 2048 --watchdog-timeout 300
    --nnodes 1 --node-rank 0 --trust-remote-code --host 127.0.0.1 --port "$FT_PORT")
  printf '%q ' env DEEP_USE_MODE=ops SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0 "PYTHONPATH=$REPO_DIR/python${PYTHONPATH:+:$PYTHONPATH}" \
    "${launch[@]}" > "$SERVER_DIR/launch-command.txt"
  printf '\n' >> "$SERVER_DIR/launch-command.txt"
  env DEEP_USE_MODE=ops SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0 \
    "PYTHONPATH=$REPO_DIR/python${PYTHONPATH:+:$PYTHONPATH}" "${launch[@]}" > "$SERVER_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  SERVER_TICKS=$(ft_start_ticks "$SERVER_PID")
  printf '%s\n' "$SERVER_PID" > "$SERVER_DIR/server.pid"
  ft_event server_started "$(jq -nc --argjson pid "$SERVER_PID" --arg ticks "$SERVER_TICKS" '{pid:$pid,start_ticks:$ticks}')"
  deadline=$((SECONDS + 1800))
  while [ "$SECONDS" -lt "$deadline" ]; do
    ft_identity_matches "$SERVER_PID" "$SERVER_TICKS" || { ft_fail "server exited; see $SERVER_DIR/server.log"; return 1; }
    if ft_status "$SERVER_DIR/startup-status.json" 2> "$SERVER_DIR/startup-curl.stderr" &&
       jq -e -L "$SCRIPT_DIR" 'include "npu_ft_bench_metrics"; status_matches("four";1)' \
         "$SERVER_DIR/startup-status.json" >/dev/null; then break; fi
    sleep 1
  done
  [ "$SECONDS" -lt "$deadline" ] || { ft_fail "startup timed out"; return 1; }
  ft_schedulers "$SERVER_DIR/schedulers.json"
  jq -e 'length==4' "$SERVER_DIR/schedulers.json" >/dev/null
  if [ ! -f "$INPUT_IDS" ]; then
    jq -nc '{prompt:("Explain how distributed inference handles requests and communicates between devices. " * 160),
      add_special_tokens:false}' > "$OUTPUT/tokenize-request.json"
    code=$(curl --noproxy '*' --connect-timeout 2 --max-time 30 -sS "$BASE_URL/v1/tokenize" \
      -H 'Content-Type: application/json' --data-binary "@$OUTPUT/tokenize-request.json" \
      -o "$OUTPUT/tokenized.json" -w '%{http_code}')
    [ "$code" = 200 ] && jq -e '.tokens|length>=256' "$OUTPUT/tokenized.json" >/dev/null
    jq '.tokens[:256]' "$OUTPUT/tokenized.json" > "$INPUT_IDS"
    sha256sum "$INPUT_IDS" > "$OUTPUT/input-ids.sha256"
  fi
  curl --noproxy '*' --connect-timeout 2 --max-time 10 -sS "$BASE_URL/get_server_info" \
    > "$SERVER_DIR/server-info.json" 2> "$SERVER_DIR/server-info.stderr" || true
  ft_event server_ready
}
new_case() {
  CASE_DIR="$OUTPUT/$1"; mkdir -p "$CASE_DIR"
  printf '%s\n' "Running $1 -> $CASE_DIR"
}
idle_before_next_phase() {
  local deadline code
  deadline=$((SECONDS + REQUEST_TIMEOUT + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    code=$(curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS -X POST "$BASE_URL/flush_cache" \
      -o "$CASE_DIR/idle-check.json" -w '%{http_code}') || code=000
    [ "$code" = 200 ] && return 0
    sleep 1
  done
  ft_fail "previous phase did not drain; refusing to contaminate next measurement"
}
load_loop() {
  local dir=$1 rate=$2 start=$3 initial_end=$4 index=0 planned deadline now
  local jobs_count
  mkdir -p "$dir/requests"
  while :; do
    deadline=$initial_end
    [ ! -f "$dir/end" ] || deadline=$(cat "$dir/end")
    planned=$(awk -v s="$start" -v i="$index" -v r="$rate" 'BEGIN{printf "%.6f",s+i/r}')
    awk -v p="$planned" -v d="$deadline" 'BEGIN{exit !(p<d)}' || break
    while :; do
      now=$(ft_now)
      awk -v n="$now" -v p="$planned" 'BEGIN{exit !(n<p)}' || break
      sleep 0.02
    done
    jobs_count=$(jobs -pr | wc -l)
    if [ "$jobs_count" -ge "$MAX_CLIENTS" ]; then
      touch "$dir/client-limited"; break
    fi
    ft_request "$dir/requests/request-$index" 256 -1 "$planned" "$REQUEST_TIMEOUT" &
    index=$((index + 1))
  done
  wait
  printf '%s\n' "$index" > "$dir/scheduled-count"
}
start_load() {
  local dir=$1 rate=$2 start=$3 end=$4
  load_loop "$dir" "$rate" "$start" "$end" & LOAD_JOB=$!
  ft_track_job "$LOAD_JOB"
}
finish_load() {
  wait "$LOAD_JOB"; LOAD_JOB=
}
window_summary() {
  local dir=$1 start=$2 end=$3 count got
  local files=("$dir"/requests/*/result.json)
  [ -f "${files[0]}" ] || { ft_fail "no request results"; return 1; }
  count=$(cat "$dir/scheduled-count"); got=${#files[@]}
  [ "$count" -eq "$got" ] || { ft_fail "request collector lost results: scheduled=$count got=$got"; return 1; }
  [ ! -e "$dir/client-limited" ] || { ft_fail "load client ran out of concurrency"; return 1; }
  jq -s . "${files[@]}" > "$dir/requests.json"
  jq -L "$SCRIPT_DIR" --argjson start "$start" --argjson end "$end" \
    --argjson ttft "${TTFT_LIMIT:-1000000}" --argjson tpot "${TPOT_LIMIT:-1000000}" \
    --argjson samples "$MIN_SAMPLES" --argjson lag "$MAX_SEND_LAG" '
    include "npu_ft_bench_metrics";
    summarize_window($start;$end;$ttft;$tpot;$samples) |
    .client_timing_valid=(.max_send_lag<=$lag) | .qualified=(.qualified and .client_timing_valid)' \
    "$dir/requests.json" > "$dir/summary.json"
}
cycle() {
  new_case "cycle-$1"; server_start server
  ft_generation warmup 0
  ft_inject_fault "$CASE_DIR/fault-stream"
  ft_apply scale_down scale-down
  ft_wait_status three
  ft_generation survivor 0
  ft_finish_fault_stream
  ft_recover
  ft_event cycle_passed
  jq -n --argjson fault_at "$FAULT_AT" '{success:true,fault_at:$fault_at}' > "$CASE_DIR/summary.json"
  server_stop
}
recovery_arm() {
  local round=$1 arm=$2 load_start end service_at
  new_case "recovery-$round-$arm"; server_start server
  ft_generation warmup 0
  load_start=$(ft_now)
  start_load "$CASE_DIR/load" "$LOAD_RATE" "$load_start" "$(awk -v n="$load_start" -v w="$WINDOW" 'BEGIN{print n+w+1800}')"
  ft_inject_fault "$CASE_DIR/fault-stream"
  end=$(awk -v n="$FAULT_AT" -v w="$WINDOW" 'BEGIN{printf "%.6f",n+w}')
  printf '%s\n' "$end" > "$CASE_DIR/load/end"
  if [ "$arm" = restart ]; then
    ft_event restart_started; server_stop; server_start restarted-server
  else
    ft_apply scale_down scale-down; ft_wait_status three
  fi
  ft_generation first-recovered-request 0
  service_at=$(jq -r .ended "$CASE_DIR/first-recovered-request/result.json")
  ft_event first_recovered_generation "$(jq -nc --argjson time "$service_at" '{completed_at:$time}')"
  finish_load
  window_summary "$CASE_DIR/load" "$FAULT_AT" "$end"
  jq -n --arg arm "$arm" --argjson fault "$FAULT_AT" --argjson service "$service_at" \
    --argjson window "$WINDOW" --slurpfile summary "$CASE_DIR/load/summary.json" \
    '{arm:$arm,fault_at:$fault,service_at:$service,recovery_seconds:($service-$fault),
      within_observation_window:(($service-$fault)<=$window),load:($summary[0]|del(.qualified))}' > "$CASE_DIR/summary.json"
  ft_finish_fault_stream
  server_stop
}
capacity_phase() {
  local phase=$1 rate duration start end label old_case=$CASE_DIR
  IFS=',' read -r -a rate_grid <<< "$RATES"
  for rate in "${rate_grid[@]}"; do
    idle_before_next_phase
    ft_generation "warmup-$phase-$rate" 0
    duration=$(awk -v minimum="$STEP" -v samples="$MIN_SAMPLES" -v rate="$rate" \
      'BEGIN{n=int(samples/0.99/rate+0.999999); if(n<minimum)n=minimum; if(n>600)n=600; print n}')
    label="$old_case/$phase/rate-$rate"; mkdir -p "$label"
    start=$(ft_now); end=$(awk -v s="$start" -v d="$duration" 'BEGIN{print s+d}')
    start_load "$label" "$rate" "$start" "$end"; finish_load
    window_summary "$label" "$start" "$end"
    jq --argjson rate "$rate" --arg phase "$phase" '.+{rate:$rate,phase:$phase}' \
      "$label/summary.json" > "$label/point.json"
  done
}
capacity() {
  local round=$1 index ttft tpot
  new_case "capacity-$round"; server_start server
  ft_generation initial-warmup 0
  if [ -z "$TTFT_LIMIT" ]; then
    mkdir -p "$CASE_DIR/reference"
    for index in $(seq 1 20); do
      ft_request "$CASE_DIR/reference/request-$index" 256 0 "$(ft_now)" "$REQUEST_TIMEOUT"
      jq -e '.outcome=="normal"' "$CASE_DIR/reference/request-$index/result.json" >/dev/null
    done
    TTFT_LIMIT=$(jq -s -L "$SCRIPT_DIR" 'include "npu_ft_bench_metrics"; map(.ttft)|percentile(0.95)*2' "$CASE_DIR"/reference/*/result.json)
    TPOT_LIMIT=$(jq -s -L "$SCRIPT_DIR" 'include "npu_ft_bench_metrics"; map(.tpot)|percentile(0.95)*2' "$CASE_DIR"/reference/*/result.json)
  fi
  jq -n --argjson ttft "$TTFT_LIMIT" --argjson tpot "$TPOT_LIMIT" \
    '{p95_ttft_limit_seconds:$ttft,p95_tpot_limit_seconds:$tpot}' > "$CASE_DIR/gates.json"
  capacity_phase four-before
  ft_inject_fault "$CASE_DIR/fault-stream"
  ft_apply scale_down scale-down; ft_wait_status three
  ft_finish_fault_stream
  capacity_phase three
  idle_before_next_phase; ft_recover
  capacity_phase four-rejoined
  jq -s '
    def cap($phase): [.[]|select(.phase==$phase and .qualified)|.rate]|max;
    . as $points | cap("four-before") as $before | cap("three") as $three | cap("four-rejoined") as $after |
    {points:$points,max_verified_request_rates:{four_before:$before,three:$three,four_rejoined:$after},
      capacity_retention:(if $before!=null and $three!=null then $three/$before else null end),
      note:"maximum verified grid point; not a proven upper limit if all grid points pass"}' \
    "$CASE_DIR"/four-before/*/point.json "$CASE_DIR"/three/*/point.json "$CASE_DIR"/four-rejoined/*/point.json \
    > "$CASE_DIR/summary.json"
  server_stop
}
admission_probes() {
  local dir=$1 index=0
  mkdir -p "$dir"
  while [ ! -e "$dir/stop" ]; do
    ft_request "$dir/probe-$index" 1 0 "$(ft_now)" 10 &
    index=$((index + 1)); sleep 1
  done
  wait
}
drain_arm() {
  local round=$1 arm=$2 old_log_lines
  new_case "drain-$round-$arm"; server_start server
  ft_inject_fault "$CASE_DIR/fault-stream"; ft_apply scale_down scale-down; ft_wait_status three
  ft_finish_fault_stream
  ft_generation survivor 0
  if [ "$arm" = timeout ]; then
    ft_schedulers "$CASE_DIR/drain-pids-before.json"
    ft_request "$CASE_DIR/long" 1536 0 "$(ft_now)" 900 & TRIGGER_JOB=$!
    ft_track_job "$TRIGGER_JOB"
    ft_wait_progress "$CASE_DIR/long" 1
    ft_apply recover drain-timeout 1
    [ ! -e "$CASE_DIR/long/done" ] && [ ! -e "$CASE_DIR/long/terminal" ]
    ft_wait_status three
    ft_schedulers "$CASE_DIR/drain-pids-after.json"
    cmp "$CASE_DIR/drain-pids-before.json" "$CASE_DIR/drain-pids-after.json"
    ft_generation admission-after-timeout 0
    wait "$TRIGGER_JOB"; TRIGGER_JOB=
    jq -e '.outcome=="normal"' "$CASE_DIR/long/result.json" >/dev/null
  else
    ft_request "$CASE_DIR/long" 1536 0 "$(ft_now)" 900 & TRIGGER_JOB=$!
    ft_track_job "$TRIGGER_JOB"
    ft_wait_progress "$CASE_DIR/long" "$DRAIN_AFTER_TOKENS"
    if [ "$arm" = recover ]; then
      ft_event recover_trigger_observed "$(jq -nc --arg tokens "$(cat "$CASE_DIR/long/progress")" \
        '{observed_output_tokens:($tokens|tonumber)}')"
      old_log_lines=$(wc -l < "$SERVER_DIR/server.log")
      admission_probes "$CASE_DIR/admission" & PROBE_JOB=$!
      ft_track_job "$PROBE_JOB"
      ft_recover
      ft_request "$CASE_DIR/admission/probe-after-recovery" 1 0 "$(ft_now)" 10
      touch "$CASE_DIR/admission/stop"; wait "$PROBE_JOB"; PROBE_JOB=
      tail -n "+$((old_log_lines + 1))" "$SERVER_DIR/server.log" > "$CASE_DIR/recovery-server.log"
      grep -q 'drain started; admission closed; pending_requests=True' "$CASE_DIR/recovery-server.log"
      grep -q 'drain complete; starting membership recovery' "$CASE_DIR/recovery-server.log"
      jq -s '[.[]|select(.outcome=="rejected")]|length>0' "$CASE_DIR"/admission/*/result.json | grep -qx true
    fi
    wait "$TRIGGER_JOB"; TRIGGER_JOB=
    jq -e '.outcome=="normal"' "$CASE_DIR/long/result.json" >/dev/null
  fi
  jq -n --arg arm "$arm" --slurpfile long "$CASE_DIR/long/result.json" \
    '{arm:$arm,long_request:$long[0]}' > "$CASE_DIR/summary.json"
  if [ "$arm" = recover ]; then
    jq -s 'sort_by(.sent) | . as $all |
      ([.[]|select(.outcome=="rejected")|.sent]|min) as $first |
      ([.[]|select(.outcome=="rejected")|.sent]|max) as $last |
      {rejected_requests:([.[]|select(.outcome=="rejected")]|length),
       first_rejection:$first,last_rejection:$last,
       first_acceptance_after_rejections:([$all[]|select(.sent>$last and .outcome=="normal")|.sent]|min),
       sampling_interval_seconds:1}' "$CASE_DIR"/admission/*/result.json > "$CASE_DIR/admission-summary.json"
    jq -s '([.[]|select(.event=="drain_started")]|first) as $start |
      ([.[]|select(.event=="drain_completed")]|first) as $end |
      {drain_wait_observed_seconds:($end.at-$start.at),
       observation_delay_note:"log tail polls at 50 ms; timestamps are client observation times"}' \
      "$CASE_DIR/recovery-log-events.jsonl" > "$CASE_DIR/phases-summary.json"
  fi
  server_stop
}

for round in $(seq 1 "$ROUNDS"); do
  case "$MODE" in cycle) cycle "$round" ;; esac
  if [ "$MODE" = recovery ] || [ "$MODE" = all ]; then
    if [ "$((round % 2))" -eq 1 ]; then arms=(restart shrink); else arms=(shrink restart); fi
    for arm in "${arms[@]}"; do recovery_arm "$round" "$arm"; done
  fi
  if [ "$MODE" = capacity ] || [ "$MODE" = all ]; then capacity "$round"; fi
  if [ "$MODE" = drain ] || [ "$MODE" = all ]; then
    if [ "$((round % 2))" -eq 1 ]; then arms=(baseline recover); else arms=(recover baseline); fi
    for arm in "${arms[@]}"; do drain_arm "$round" "$arm"; done
    drain_arm "$round" timeout
  fi
done
jq -n '{cases:[]}' > "$OUTPUT/summary.json"
summary_files=("$OUTPUT"/*/summary.json)
if [ -f "${summary_files[0]}" ]; then
  : > "$OUTPUT/cases.jsonl"
  for file in "${summary_files[@]}"; do
    name=${file%/summary.json}; name=${name##*/}
    jq --arg case "$name" '.+{case:$case}' "$file" >> "$OUTPUT/cases.jsonl"
  done
  jq -s '
    def stats: sort | if length==0 then null else
      {n:length,median:(if length%2==1 then .[length/2|floor]
       else (.[length/2-1]+.[length/2])/2 end),min:.[0],max:.[-1]} end;
    . as $cases | {cases:$cases,
      recovery_seconds:{restart:([$cases[]|select(.arm?=="restart")|.recovery_seconds]|stats),
                        shrink:([$cases[]|select(.arm?=="shrink")|.recovery_seconds]|stats)},
      capacity_retention:([$cases[]|select(.capacity_retention?!=null)|.capacity_retention]|stats),
      failed_rounds_note:"Any failed run stops immediately and retains failure.json; it is not silently retried."}' \
      "$OUTPUT/cases.jsonl" > "$OUTPUT/summary.json"
fi
printf 'Completed. Raw streams, events, PID identities and summaries: %s\n' "$OUTPUT"
