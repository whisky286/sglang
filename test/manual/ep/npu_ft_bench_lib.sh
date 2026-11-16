#!/usr/bin/env bash
# Runtime helpers use only Bash, curl, awk, jq, ps and Linux procfs.

ft_now() { local value rest; read -r value rest < "${FT_CLOCK_FILE:-/proc/uptime}"; printf '%s\n' "$value"; }
ft_fail() { printf 'STOP: %s\n' "$*" >&2; return 1; }
ft_event() {
  local name=$1 details=${2:-'{}'}
  jq -nc --arg event "$name" --argjson at "$(ft_now)" --argjson details "$details" \
    '{event:$event,at:$at,details:$details}' >> "$CASE_DIR/events.jsonl"
}
ft_start_ticks() {
  local stat rest
  IFS= read -r stat < "${FT_PROC_ROOT:-/proc}/$1/stat" || return 1
  rest=${stat##*) }
  # After comm, field 3 is first; starttime is field 22 (index 20 here).
  set -- $rest
  [ "$#" -ge 20 ] || return 1
  [ "$1" != Z ] && [ "$1" != X ] || return 1
  shift 19
  printf '%s\n' "$1"
}
ft_identity_matches() { [ "$(ft_start_ticks "$1" 2>/dev/null)" = "$2" ]; }
ft_track_job() {
  local ticks
  [ -n "${OUTPUT:-}" ] || return 0
  ticks=$(ft_start_ticks "$1" 2>/dev/null) || return 0
  printf '%s\n' "$ticks" > "$OUTPUT/client-job-$1.start"
}

ft_descendants() {
  # Read all rows before walking the parent links; ps ordering is immaterial.
  awk -v root="$1" '
    { p[$1]=$2; row[$1]=$0 }
    END { for (id in row) {
      at=id; delete seen
      while (at in p && !seen[at]++) {
        if (at == root) {print row[id]; break}
        at=p[at]
      }
    }}'
}
ft_schedulers() {
  local target=$1 pid ppid state title extra ticks rank
  ft_identity_matches "$SERVER_PID" "$SERVER_TICKS" || ft_fail "server identity changed" || return
  ps -e -o pid= -o ppid= -o stat= -o args= | ft_descendants "$SERVER_PID" > "$target.tree"
  : > "$target.jsonl"
  while read -r pid ppid state title extra; do
    case "$state" in Z*) continue ;; esac
    case "$title" in sglang::scheduler_DP*) ;; *) continue ;; esac
    rank=${title#sglang::scheduler_DP}; rank=${rank%%_*}
    case "$rank" in 0|1|2|3) ;; *) return 1 ;; esac
    [ "$title" = "sglang::scheduler_DP${rank}_TP${rank}_EP${rank}" ] || \
      { ft_fail "requires TP=DP=EP=4, one scheduler per DP: $title"; return 1; }
    ticks=$(ft_start_ticks "$pid") || return
    jq -nc --argjson rank "$rank" --argjson pid "$pid" --arg ticks "$ticks" \
      '{rank:$rank,pid:$pid,start_ticks:$ticks}' >> "$target.jsonl"
  done < "$target.tree"
  jq -s 'sort_by(.rank)' "$target.jsonl" > "$target"
  jq -e '([.[].rank] | unique | length) == length' "$target" >/dev/null
}
ft_status() {
  local output=$1 code
  code=$(curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
    "$BASE_URL/fault_tolerance/status" -o "$output" -w '%{http_code}') || return 1
  [ "$code" = 200 ] || return 1
  jq -e -L "$SCRIPT_DIR" 'include "npu_ft_bench_metrics"; valid_status' "$output" >/dev/null
}
ft_wait_status() {
  local mode=$1 deadline file sequence=0
  FT_STATUS_SERIAL=$((${FT_STATUS_SERIAL:-0} + 1))
  deadline=$(awk -v a="$(ft_now)" -v b="$STATUS_TIMEOUT" 'BEGIN{print a+b}')
  while awk -v n="$(ft_now)" -v d="$deadline" 'BEGIN{exit !(n<d)}'; do
    sequence=$((sequence + 1)); file="$CASE_DIR/status-${mode}-${FT_STATUS_SERIAL}-${sequence}.json"
    if ft_status "$file"; then
      jq -nc --arg mode "$mode" --argjson at "$(ft_now)" --slurpfile status "$file" \
        '{mode:$mode,at:$at,status:$status[0]}' >> "$CASE_DIR/status-observations.jsonl"
      if [ "$mode" = incident ] && [ ! -e "$CASE_DIR/first-incident.json" ] &&
         jq -e 'any(.ranks[]; .state!="healthy")' "$file" >/dev/null; then
        cp "$file" "$CASE_DIR/first-incident.json"
        ft_event incident_observed "$(cat "$file")"
      fi
      if jq -e -L "$SCRIPT_DIR" --arg mode "$mode" --argjson rank "$FAULT_RANK" \
        'include "npu_ft_bench_metrics"; status_matches($mode;$rank)' "$file" >/dev/null; then
        ft_event "status_$mode" "$(cat "$file")"
        return 0
      fi
    fi
    sleep 0.1
  done
  ft_fail "timeout waiting for $mode; no control instruction sent"
}
ft_request() {
  # One logical request, exactly one curl attempt. Results remain on any failure.
  local dir=$1 tokens=$2 rank=$3 planned=$4 limit=$5
  local sent ended codes http curl_rc collector_rc
  mkdir -p "$dir"
  jq --argjson tokens "$tokens" --argjson rank "$rank" \
    '{input_ids:.,stream:true,sampling_params:{temperature:0,max_new_tokens:$tokens,ignore_eos:true}} |
     if $rank >= 0 then .routed_dp_rank=$rank else . end' "$INPUT_IDS" > "$dir/payload.json"
  sent=$(ft_now)
  set +e
  curl --noproxy '*' --connect-timeout 2 --max-time "$limit" -N -sS \
    "$BASE_URL/generate" -H 'Content-Type: application/json' \
    --data-binary "@$dir/payload.json" -w '\n__HTTP_STATUS__:%{http_code}\n' 2> "$dir/curl.stderr" |
    awk -v dir="$dir" -v clock_file="${FT_CLOCK_FILE:-/proc/uptime}" -f "$SCRIPT_DIR/npu_ft_bench_stream.awk" |
    jq --unbuffered -Rnr '
      inputs | split("\t") | .[0] as $at |
      (.[1:] | join("\t")) | select(. != "[DONE]") | fromjson |
      select(type=="object") |
      [$at, (.meta_info.completion_tokens // 0),
       (.meta_info.finish_reason | if type=="object" then .type elif type=="string" then . else "-" end),
       (.meta_info.prompt_tokens // -1), (has("error"))] | @tsv' 2> "$dir/collector.stderr" |
    awk -v dir="$dir" 'BEGIN{FS="\t"} {
      print $0 >> (dir "/tokens.tsv"); fflush()
      print $2 > (dir "/progress"); close(dir "/progress")
      if ($3 != "-" || $5 == "true") {print "terminal" > (dir "/terminal"); close(dir "/terminal")}
    }'
  codes=("${PIPESTATUS[@]}")
  curl_rc=${codes[0]}; collector_rc=$((codes[1] + codes[2] + codes[3]))
  set -e
  ended=$(ft_now)
  http=$(cat "$dir/http.txt" 2>/dev/null || printf '0')
  # curl prints 000 on a failed connection; normalize it as JSON number 0.
  http=$((10#${http:-0}))
  touch "$dir/tokens.tsv"
  jq -Rn '[inputs | split("\t") |
    {at:(.[0]|tonumber),count:(.[1]|tonumber),finish:.[2],prompt:(.[3]|tonumber),error:(.[4]=="true")}]' \
    "$dir/tokens.tsv" > "$dir/token-events.json"
  jq -n -L "$SCRIPT_DIR" --arg id "${dir##*/}" --argjson sent "$sent" --argjson ended "$ended" \
    --argjson planned "$planned" --argjson expected "$tokens" --argjson http "$http" \
    --argjson curl_exit "$curl_rc" --argjson collector_exit "$collector_rc" \
    --argjson input_tokens "$(jq length "$INPUT_IDS")" --slurpfile events "$dir/token-events.json" '
      include "npu_ft_bench_metrics";
      {id:$id,sent:$sent,ended:$ended,planned:$planned,expected:$expected,http:$http,
       curl_exit:$curl_exit,collector_exit:$collector_exit,input_tokens:$input_tokens,events:$events[0]} |
      summarize_request' > "$dir/result.json.tmp"
  mv "$dir/result.json.tmp" "$dir/result.json"
  touch "$dir/done"
}
ft_wait_progress() {
  local dir=$1 threshold=$2 deadline progress
  deadline=$(awk -v a="$(ft_now)" -v b="$REQUEST_TIMEOUT" 'BEGIN{print a+b}')
  while awk -v n="$(ft_now)" -v d="$deadline" 'BEGIN{exit !(n<d)}'; do
    if [ -e "$dir/terminal" ] || [ -e "$dir/done" ]; then
      ft_fail "trigger request already ended; fault was NOT injected"; return 1
    fi
    progress=$(cat "$dir/progress" 2>/dev/null || true)
    if [ -n "$progress" ] && [ "$progress" -ge "$threshold" ]; then return 0; fi
    sleep 0.02
  done
  ft_fail "trigger request produced no usable token; fault was NOT injected"
}
ft_inject_fault() {
  local dir=$1 pid ticks current
  ft_schedulers "$CASE_DIR/pids-before.json" || return
  jq -e 'length == 4' "$CASE_DIR/pids-before.json" >/dev/null || return 1
  pid=$(jq -r --argjson rank "$FAULT_RANK" '.[] | select(.rank==$rank) | .pid' "$CASE_DIR/pids-before.json")
  ticks=$(ft_start_ticks "$pid") || return
  ft_request "$dir" 1536 "$FAULT_RANK" "$(ft_now)" 900 &
  TRIGGER_JOB=$!
  ft_track_job "$TRIGGER_JOB"
  ft_wait_progress "$dir" "$FAULT_AFTER_TOKENS" || return
  # Repeat ancestry, title and birth-time checks immediately before the signal.
  ft_schedulers "$CASE_DIR/pids-at-kill.json" || return
  current=$(jq -r --argjson rank "$FAULT_RANK" '.[] | select(.rank==$rank) | .pid' "$CASE_DIR/pids-at-kill.json")
  [ "$pid" = "$current" ] && ft_identity_matches "$pid" "$ticks" || \
    { ft_fail "rank PID or process identity changed before kill"; return 1; }
  [ ! -e "$dir/terminal" ] && [ ! -e "$dir/done" ] || \
    { ft_fail "trigger completed before kill; refusing idle fault injection"; return 1; }
  FAULT_AT=$(ft_now)
  kill -KILL "$pid"
  ft_event fault_injected "$(jq -nc --argjson pid "$pid" --arg ticks "$ticks" --argjson rank "$FAULT_RANK" \
    --argjson at "$FAULT_AT" --arg tokens "$(cat "$dir/progress")" \
    '{pid:$pid,start_ticks:$ticks,rank:$rank,signal_at:$at,observed_output_tokens:($tokens|tonumber)}')"
  ft_wait_status incident
}
ft_finish_fault_stream() {
  wait "$TRIGGER_JOB"; TRIGGER_JOB=
  # A completed long request means the intended in-flight fault was missed.
  # A timeout or collector error is not evidence of reliable terminal cleanup.
  jq -e '.outcome=="aborted" or .outcome=="transport_error" or .outcome=="missing_terminal"' \
    "$CASE_DIR/fault-stream/result.json" >/dev/null
}
ft_apply() {
  local instruction=$1 label=$2 drain=${3:-600} code rc
  if [ "$instruction" = scale_down ]; then
    ft_wait_status incident || return
  fi
  jq -nc --arg instruction "$instruction" --argjson rank "$FAULT_RANK" \
    --argjson timeout "$CONTROL_TIMEOUT" --argjson drain "$drain" \
    '{instruction:$instruction,params:{ranks:[$rank],timeout:$timeout}} |
     if $instruction=="recover" then .params.drain_timeout=$drain else . end' > "$CASE_DIR/$label-request.json"
  ft_event "${label}_started"
  # Never retry this POST: a timed-out operation may already have been accepted.
  rc=0
  code=$(curl --noproxy '*' --connect-timeout 2 --max-time "$((5 * CONTROL_TIMEOUT + drain + 30))" \
    -sS "$BASE_URL/fault_tolerance/apply" -H 'Content-Type: application/json' \
    --data-binary "@$CASE_DIR/$label-request.json" -o "$CASE_DIR/$label.json" \
    -w '%{http_code}' 2> "$CASE_DIR/$label.stderr") || rc=$?
  printf '%s\n' "$code" > "$CASE_DIR/$label.http"
  ft_event "${label}_returned" "$(jq -nc --arg code "$code" --argjson rc "$rc" '{http:$code,curl_exit:$rc}')"
  [ "$rc" -eq 0 ] || { ft_fail "$label response unknown; operation will NOT be resent"; return 1; }
  if [ "$label" = "drain-timeout" ]; then
    [ "$code" = 408 ] && jq -e '.success==false and .message=="recover_drain_timeout"' "$CASE_DIR/$label.json" >/dev/null
  else
    [ "$code" = 200 ] && jq -e --arg expected "fault_tolerance_${instruction}_committed" \
      '.success==true and .message==$expected' "$CASE_DIR/$label.json" >/dev/null
  fi
}
ft_generation() {
  local label=$1 rank=$2
  ft_request "$CASE_DIR/$label" 16 "$rank" "$(ft_now)" "$REQUEST_TIMEOUT"
  jq -e '.outcome=="normal"' "$CASE_DIR/$label/result.json" >/dev/null
}
ft_log_events() {
  local file=$1 offset=$2
  # Each log phase is timestamped on receipt, on the same monotonic clock as
  # requests. The polling delay (<=50 ms, plus scheduling) is documented.
  tail -n "+$((offset + 1))" -f --sleep-interval=0.05 "$file" |
    awk -v clock_file="${FT_CLOCK_FILE:-/proc/uptime}" '
      /NPU FT recovery drain started|NPU FT recovery drain complete|NPU FT recovery drain stopped/ {
        getline clock < clock_file; close(clock_file); split(clock,t," ")
        if (/drain started/) event="drain_started"
        else if (/drain complete/) event="drain_completed"
        else event="drain_stopped"
        printf "{\"event\":\"%s\",\"at\":%s,\"pending_requests\":%s}\n", \
          event,t[1],($0 ~ /pending_requests=True/ ? "true" : "false")
        fflush()
      }' >> "$CASE_DIR/recovery-log-events.jsonl"
}
ft_recover() {
  if [ -n "${SERVER_DIR:-}" ] && [ -f "$SERVER_DIR/server.log" ]; then
    ft_log_events "$SERVER_DIR/server.log" "$(wc -l < "$SERVER_DIR/server.log")" & LOG_JOB=$!
    ft_track_job "$LOG_JOB"
  fi
  ft_apply recover recover
  ft_wait_status four
  ft_schedulers "$CASE_DIR/pids-after.json"
  jq -e -n --argjson target "$FAULT_RANK" --slurpfile before "$CASE_DIR/pids-before.json" \
    --slurpfile after "$CASE_DIR/pids-after.json" '
    ($before[0]|length)==4 and ($after[0]|length)==4 and
    all($before[0][]; . as $b | ($after[0][]|select(.rank==$b.rank)) as $a |
      if .rank==$target then ($a.pid!=$b.pid or $a.start_ticks!=$b.start_ticks)
      else ($a.pid==$b.pid and $a.start_ticks==$b.start_ticks) end)' >/dev/null
  ft_generation rejoined "$FAULT_RANK"
  ft_event recovered_rank_generated
  if [ -n "${LOG_JOB:-}" ]; then
    ft_stop_job "$LOG_JOB"; LOG_JOB=
  fi
}
