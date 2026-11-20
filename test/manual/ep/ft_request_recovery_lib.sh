#!/usr/bin/env bash
# Shared parsing/checks; sourcing this file performs no server operations.

ft_sse_json() {
  if [ ! -f "$1" ]; then printf '[]\n'; return 0; fi
  sed -n 's/^data: \({.*}\)$/\1/p' "$1" | jq -s .
}

ft_check_stream() {
  local data=$1 prefix=$2 rid=$3 tokens=$4 attempts=$5 terminal=${6:-length}
  jq -e --arg rid "$rid" --arg terminal "$terminal" --argjson tokens "$tokens" \
    --argjson attempts "$attempts" --slurpfile before "$prefix" '
    length > 0 and all(.[]; .meta_info.id == $rid) and
    (([.[].text] | join("")) | startswith(([$before[0][].text] | join("")))) and
    ([.[] | select(.meta_info.finish_reason != null)] | length) == 1 and
    .[-1].meta_info.finish_reason.type == $terminal and
    (if $terminal == "length" then
      .[-1].meta_info.completion_tokens == $tokens and
      .[-1].meta_info.recovery_attempts == $attempts
     else true end) and
    ([.[].output_ids[]] | length) == .[-1].meta_info.completion_tokens and
    ([.[].output_ids[]][0:([$before[0][].output_ids[]] | length)] ==
      [$before[0][].output_ids[]]) and
    (reduce .[] as $chunk ({n:0, ok:true};
      .n += ($chunk.output_ids | length) |
      .ok = (.ok and .n == $chunk.meta_info.completion_tokens)) | .ok)
  ' "$data" >/dev/null
}

ft_proc_ticks() {
  [ -r "/proc/$1/stat" ] || return 1
  awk '{sub(/^.*\) /,""); print $20}' "/proc/$1/stat"
}

ft_same_process() {
  local actual
  actual=$(ft_proc_ticks "$1") || return 1
  [ "$actual" = "$2" ]
}

ft_descends_from() {
  local pid=$1 ancestor=$2 depth=0
  while [ "$pid" -gt 1 ] && [ "$depth" -lt 32 ]; do
    [ "$pid" != "$ancestor" ] || return 0
    pid=$(awk '/^PPid:/ {print $2}' "/proc/$pid/status") || return 1
    [ -n "$pid" ] || return 1
    depth=$((depth + 1))
  done
  return 1
}

ft_rank_pid() {
  local rank=$1 ancestor=$2 pid name matches=
  name="sglang::scheduler_DP${rank}_TP${rank}_EP${rank}"
  while read -r pid; do
    if ft_descends_from "$pid" "$ancestor"; then
      [ -z "$matches" ] || return 1
      matches=$pid
    fi
  done < <(ps -eo pid=,args= | awk -v name="$name" 'index($0,name) {print $1}')
  [ -n "$matches" ] || return 1
  printf '%s\n' "$matches"
}

ft_signal_rank() {
  local pid=$1 ticks=$2 rank=$3 ancestor=$4
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$ancestor" =~ ^[1-9][0-9]*$ && "$ticks" =~ ^[0-9]+$ ]] || return 1
  ft_same_process "$pid" "$ticks" && ft_descends_from "$pid" "$ancestor" || return 1
  ps -p "$pid" -o args= | grep -Fq "sglang::scheduler_DP${rank}_TP${rank}_EP${rank}" || return 1
  kill -KILL "$pid"
}
