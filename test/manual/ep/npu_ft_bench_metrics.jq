# The collector timestamps SSE events using Linux /proc/uptime (10 ms resolution).
def percentile($p):
  sort | if length == 0 then null else .[(([length * $p | ceil, 1] | max) - 1)] end;

def summarize_request:
  . as $r |
  (reduce .events[] as $e
    ({previous: 0, first: null, last: null, max_gap: 0, increments: [], invalid: false};
     if $e.count < .previous then .invalid = true
     elif $e.count > .previous then
       .increments += [{at: $e.at, count: ($e.count - .previous)}] |
       .max_gap = ([.max_gap, (if .last == null then 0 else $e.at - .last end)] | max) |
       .first = (.first // $e.at) | .last = $e.at | .previous = $e.count
     else . end)) as $s |
  ($r.events | map(select(.finish != "-")) | last) as $terminal |
  ($r.events | any(.error)) as $error |
  (if $r.http >= 400 then (if $r.http == 503 then "rejected" else "http_error" end)
   elif $terminal.finish == "abort" or $error then "aborted"
   elif $r.curl_exit == 28 then "timeout"
   elif $r.curl_exit != 0 then "transport_error"
   elif $r.collector_exit != 0 or $s.invalid then "collector_error"
   elif $r.http != 200 or $terminal == null then "missing_terminal"
   elif $terminal.finish != "length" or $terminal.count != $r.expected then "wrong_length"
   elif $terminal.prompt != $r.input_tokens then "wrong_input_length"
   else "normal" end) as $outcome |
  $r + {outcome: $outcome, output_tokens: $s.previous,
    ttft: (if $s.first == null then null else $s.first - $r.sent end),
    tpot: (if $s.previous > 1 then ($s.last - $s.first) / ($s.previous - 1) else null end),
    max_output_gap: $s.max_gap, duration: ($r.ended - $r.sent),
    send_lag: ($r.sent - $r.planned), increments: $s.increments,
    terminal: $terminal} | del(.events);

def summarize_window($start; $end; $ttft; $tpot; $samples):
  map(select(.sent >= $start and .sent < $end)) as $all |
  ($all | map(select(.outcome == "normal"))) as $ok |
  # Include token arrivals in this window from *all* normally completed requests,
  # even if their send time preceded the window. Request-rate denominators above
  # instead use the cohort sent in the window.
  ([.[] | select(.outcome == "normal") | .increments[] |
      select(.at >= $start and .at < $end) | .count] | add // 0) as $tokens |
  ($ok | map(.ttft) | percentile(0.95)) as $p95_ttft |
  ($ok | map(.tpot | select(. != null)) | percentile(0.95)) as $p95_tpot |
  (if ($all | length) == 0 then 0 else ($ok | length) / ($all | length) end) as $rate |
  {start: $start, end: $end, duration: ($end - $start),
   sent: ($all | length), normal: ($ok | length), success_rate: $rate,
   failures: ($all | map(select(.outcome != "normal")) | length),
   outcomes: ($all | sort_by(.outcome) | group_by(.outcome) |
     map({key: .[0].outcome, value: length}) | from_entries),
   output_tokens_per_second: ($tokens / ($end - $start)),
   p95_ttft: $p95_ttft, p95_tpot: $p95_tpot,
   max_send_lag: ($all | map(.send_lag) | max // 0),
   enough_samples: (($ok | length) >= $samples),
   qualified: (($ok | length) >= $samples and $rate >= 0.99 and
      $p95_ttft != null and $p95_ttft <= $ttft and
      $p95_tpot != null and $p95_tpot <= $tpot)};

def valid_status:
  (.ranks | type) == "array" and (.ranks | length) == 4 and
  ([.ranks[].rank] | sort) == [0,1,2,3] and
  all(.ranks[]; (.state == "healthy" or .state == "unhealthy" or .state == "dead"));

def status_matches($mode; $rank):
  valid_status and
  (if $mode == "four" then all(.ranks[]; .state == "healthy")
   elif $mode == "incident" then
     all(.ranks[]; if .rank == $rank then .state == "dead" else .state == "unhealthy" end)
   elif $mode == "three" then
     all(.ranks[]; if .rank == $rank then .state == "dead" else .state == "healthy" end)
   else false end);
