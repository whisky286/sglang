# 第 2 步｜保存请求状态与隔离旧执行（手动）

本页固定代码 `976f7f8afb0b73712170ac1ac13995ea42160499`。开关启用后，服务为单条普通文本请求保留 CPU 上的输入与生成进度，并用内部执行编号区分同一个公开请求编号的不同执行。批量预热和健康探测继续走原路径。本页验证记录创建、正常结束/取消时清理，以及一次故障缩容、补卡和编号复用。故障请求在本批仍明确结束，自动重试在下一批。

2026-11-11 已按本页全流程在单机四卡执行完毕，三个小节全部通过：journal 开启与计数归零、同编号两次生成、主动取消收尾、故障流明确 abort、缩容与补卡提交、新 rank 1 复用编号生成，PID 对照仅 rank 1 更换进程。证据目录 `artifacts/npu_ft/request-journal-20261111-222125`，逐条结论见[连续性方案的实机记录](PLAN_npu_request_continuity.md)。

在 NPU 仓库根目录、原 sglang-ft 环境中执行 Bash 命令，保留已有 CANN/自定义算子设置。开始本页时需启动一次新版本服务，加载默认关闭的 `--enable-npu-ft-request-journal`。

## 1. 启动并核对版本

终端 1。先拉取代码并确认提交可用。

```bash
git fetch https://github.com/whisky286/sglang.git codex/npu-ft-recover &&
git cat-file -t 976f7f8afb0b73712170ac1ac13995ea42160499
```

看到 commit 后，等已有请求结束，按保存的主进程 PID 停止旧实验服务，保留旧日志。Ctrl+C 可先退出正在跟随的 tail。若没有 current-recovery-dir.txt，直接做端口检查。

```bash
if [ -f artifacts/npu_ft/current-recovery-dir.txt ]; then
  OLD_RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
  if [ -f "$OLD_RUN_DIR/server.pid" ]; then
    OLD_SERVER_PID=$(cat "$OLD_RUN_DIR/server.pid")
    if ps -p "$OLD_SERVER_PID" -o args= | grep -q '[s]glang.launch_server' &&
       ss -ltnp 'sport = :31000' | grep -Fq "pid=$OLD_SERVER_PID,"; then
      kill -TERM "$OLD_SERVER_PID"
    fi
  fi
fi
```

单独检查：31000 没有监听后再继续；仍监听时先处理旧服务，不启动第二份。

```bash
ss -ltnp 'sport = :31000'
```

终端 1。固定到本页提交。Git 报本地修改冲突时保留修改，停止后续启动。

```bash
git switch --detach 976f7f8afb0b73712170ac1ac13995ea42160499 &&
git rev-parse HEAD
```

更新成功后再启动：

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}"
RUN_DIR="artifacts/npu_ft/request-journal-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > artifacts/npu_ft/current-recovery-dir.txt
git rev-parse HEAD > "$RUN_DIR/commit.txt"
git diff > "$RUN_DIR/working-tree.diff"
python --version > "$RUN_DIR/python-version.txt" 2>&1
python -m pip freeze > "$RUN_DIR/requirements.txt"

DEEP_USE_MODE=ops \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0 \
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --device npu --attention-backend ascend --dtype bfloat16 \
  --tp-size 4 --dp-size 4 --ep-size 4 --attn-cp-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-dense-tp-size 1 --moe-dp-size 1 \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --enable-eplb --eplb-algorithm elasticity_aware \
  --ep-dispatch-algorithm dynamic --ep-num-redundant-experts 128 \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance --enable-npu-ft-recovery --enable-npu-ft-request-journal \
  --fault-tolerance-on-error-strategy pause \
  --fault-tolerance-timeout 600 --fault-tolerance-pause-timeout 600 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --disable-cuda-graph --disable-overlap-schedule \
  --disable-radix-cache --mem-fraction-static 0.5 \
  --max-running-requests 8 --max-total-tokens 4096 \
  --context-length 2048 --watchdog-timeout 300 \
  --nnodes 1 --node-rank 0 --trust-remote-code \
  --host 127.0.0.1 --port 31000 \
  > "$RUN_DIR/server.log" 2>&1 &
printf '%s\n' "$!" > "$RUN_DIR/server.pid"
tail -f "$RUN_DIR/server.log"
```

终端 3。等终端 1 出现 ready to roll 后执行。预期四卡 healthy，启动日志确认 journal 已开启，计数均为 0。后面的检查通过才进入故障注入。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
grep -E 'NPU FT request journal enabled|ready to roll' "$RUN_DIR/server.log" | tail -n 5
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
  http://127.0.0.1:31000/fault_tolerance/status \
  | tee "$RUN_DIR/status-ready.json" \
  | jq -e '(.ranks | length) == 4 and all(.ranks[]; .state == "healthy") and (.request_journal == {active_requests:0,reserved_tokens:0,stale_messages_dropped:0})'
```

## 2. 验证普通请求和主动取消

终端 3。连续两次用同一个公开请求编号，每次都应返回 true。状态里的 active_requests 与 reserved_tokens 在两个请求结束后都应回到 0。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
for N in 1 2; do
  curl --noproxy '*' --connect-timeout 5 --max-time 600 -fsS \
    http://127.0.0.1:31000/generate \
    -H 'Content-Type: application/json' \
    -d '{"rid":"journal-reuse-check","text":"The capital of France is","routed_dp_rank":0,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
    | tee "$RUN_DIR/reuse-$N.json" \
    | jq -e '.meta_info.id == "journal-reuse-check" and .meta_info.completion_tokens > 0 and (.meta_info.finish_reason.type != "abort")' || break
done
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
  http://127.0.0.1:31000/fault_tolerance/status \
  | tee "$RUN_DIR/status-after-reuse.json" | jq '.request_journal'
```

终端 2。启动长流式请求。看到持续的 data 行后切到终端 3。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
curl --noproxy '*' --connect-timeout 5 --max-time 900 -N -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"rid":"journal-cancel-check","text":"Write a detailed numbered explanation of distributed inference.","stream":true,"sampling_params":{"temperature":0,"max_new_tokens":1536,"ignore_eos":true}}' \
  | tee "$RUN_DIR/cancel-stream.sse"
```

终端 3。确认当前实验请求还在运行再取消。预期 active_requests 为 1，reserved_tokens 大于 0，取消接口返回 HTTP 200。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
     http://127.0.0.1:31000/fault_tolerance/status -o "$RUN_DIR/status-before-cancel.json" &&
   jq -e '.request_journal | .active_requests == 1 and .reserved_tokens > 0' "$RUN_DIR/status-before-cancel.json"; then
  curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
    http://127.0.0.1:31000/abort_request \
    -H 'Content-Type: application/json' \
    -d '{"rid":"journal-cancel-check","abort_all":false}' \
    -o "$RUN_DIR/cancel-response.txt" -w 'HTTP=%{http_code}\n'
fi
```

终端 2 的流结束后，终端 3 核对终态。预期同一公开编号、abort、计数归零。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
sed -n 's/^data: \({.*}\)$/\1/p' "$RUN_DIR/cancel-stream.sse" \
  | jq -s 'map(select(.meta_info.finish_reason != null)) | last | {id:.meta_info.id,finish_reason:.meta_info.finish_reason}' \
  | tee "$RUN_DIR/cancel-terminal.json"
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
  http://127.0.0.1:31000/fault_tolerance/status \
  | tee "$RUN_DIR/status-after-cancel.json" | jq '.request_journal'
```

## 3. 验证故障收尾、缩容和补卡

终端 3：创建本轮目录，保存本次真实 PID。根据输出核对 rank 1，不复用旧实验的 PID。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR="$RUN_DIR/cycle-1"
mkdir -p "$ROUND_DIR"
printf '%s\n' "$ROUND_DIR" > "$RUN_DIR/current-round.txt"
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
ps -eo pid=,ppid=,lstart=,args= | grep '[s]glang::scheduler_' | tee "$ROUND_DIR/pids-before.txt"
```

终端 2：启动长流式请求，看到持续输出后回到终端 3。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 900 -N -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"rid":"journal-fault-check","text":"Write a detailed numbered explanation of distributed inference.","stream":true,"sampling_params":{"temperature":0,"max_new_tokens":1536,"ignore_eos":true}}' \
  | tee "$ROUND_DIR/fault-stream.sse"
printf 'curl exit=%s\n' "${PIPESTATUS[0]}" | tee "$ROUND_DIR/fault-client-exit.txt"
```

终端 3：输入刚核对的本轮 rank 1 PID，然后回车。输入空白/非数字时不会执行 kill；仍需人工确认属于本轮服务。

```bash
printf '输入本轮 rank 1 PID：'
read -r RANK1_PID
case "$RANK1_PID" in
  ''|*[!0-9]*) printf '%s\n' 'STOP: PID 不是数字，未发送 kill。' ;;
  *)
    RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
    ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
    CURRENT_PID_LINE=$(ps -p "$RANK1_PID" -o pid=,ppid=,lstart=,args= | awk '{$1=$1;print}')
    if printf '%s\n' "$CURRENT_PID_LINE" | grep -q 'sglang::scheduler_DP1_TP1_EP1' &&
       awk '{$1=$1;print}' "$ROUND_DIR/pids-before.txt" | grep -Fxq "$CURRENT_PID_LINE"; then
      kill -9 "$RANK1_PID"
    else
      printf '%s\n' 'STOP: PID 不是 rank 1 Scheduler，未发送 kill。'
    fi
    ;;
esac
```

查询状态，等 rank 1 dead、0/2/3 unhealthy 后再缩容。未达到时只重复这条查询，不重复 kill。

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
  http://127.0.0.1:31000/fault_tolerance/status | jq .
```

终端 3：缩容，再验证三卡生成。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 2430 -sS \
  http://127.0.0.1:31000/fault_tolerance/apply \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"scale_down","params":{"ranks":[1],"timeout":600}}' \
  | tee "$ROUND_DIR/scale-down.json" | jq .
```

仅 success=true 时继续：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","routed_dp_rank":0,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$ROUND_DIR/survivor.json" | jq .
```

终端 2 的故障长请求应以 abort/503 或错误事件结束。终端 3 在三卡生成通过后确认流已结束且记录释放，再补卡。若流仍挂住，保留日志与当前现场。

终端 3。先保存故障流末尾与三卡状态。预期故障流已有终态、幸存 rank 可生成、active_requests=0 且 reserved_tokens=0。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
tail -n 8 "$ROUND_DIR/fault-stream.sse"
cat "$ROUND_DIR/fault-client-exit.txt"
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
  http://127.0.0.1:31000/fault_tolerance/status \
  | tee "$ROUND_DIR/status-after-shrink.json" | jq .
```

终端 3：只发一次 recover，不重试可能已接受的事务。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
if jq -e '.success == true' "$ROUND_DIR/scale-down.json" >/dev/null &&
   jq -e '.meta_info.completion_tokens > 0 and (.text | length > 0) and ((.meta_info.finish_reason.type? // "") != "abort")' "$ROUND_DIR/survivor.json" >/dev/null &&
   test -f "$ROUND_DIR/fault-client-exit.txt" &&
   jq -e '.request_journal | .active_requests == 0 and .reserved_tokens == 0' \
     "$ROUND_DIR/status-after-shrink.json" >/dev/null; then
  curl --noproxy '*' --connect-timeout 5 --max-time 2430 -sS \
    http://127.0.0.1:31000/fault_tolerance/apply \
    -H 'Content-Type: application/json' \
    -d '{"instruction":"recover","params":{"ranks":[1],"timeout":600}}' \
    | tee "$ROUND_DIR/recover.json" | jq .
else
  printf '%s\n' 'STOP: 前置检查失败，未发送 recover。'
fi
```

恢复 success=true 后，终端 3 独立检查四卡 healthy，人工对照 PID：只有 rank 1 换进程，其余三个 PID/创建时间不变。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
  http://127.0.0.1:31000/fault_tolerance/status | tee "$ROUND_DIR/status-after.json" | jq .
cat "$ROUND_DIR/pids-before.txt"
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$ROUND_DIR/pids-after.txt"
```

终端 3。四卡 healthy 后复用故障请求的公开编号，指定新 rank 1 生成。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"rid":"journal-fault-check","text":"The capital of France is","routed_dp_rank":1,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$ROUND_DIR/recovered.json" | jq .
```

终端 3。预期生成通过，公开编号仍为 journal-fault-check，active_requests 和 reserved_tokens 回到 0。stale_messages_dropped 有无增长取决于现场是否收到迟到消息。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
jq -e '.meta_info.id == "journal-fault-check" and .meta_info.completion_tokens > 0 and (.meta_info.finish_reason.type != "abort")' \
  "$ROUND_DIR/recovered.json"
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
  http://127.0.0.1:31000/fault_tolerance/status \
  | tee "$ROUND_DIR/status-final.json" | jq .
```

回传 commit.txt、status-ready.json、两次普通请求的检查结果、status-before-cancel.json、cancel-terminal.json、status-after-cancel.json、故障流尾部、scale-down.json、recover.json、恢复后生成检查、status-final.json 和 PID 对照结论。旧消息的定向注入已在本机单测覆盖，实机用正常 API 验证整个请求路径。

## 失败时回传

任一步失败后保留目录，不重复 kill/recover。逐条执行，只读：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
cat "$RUN_DIR/commit.txt"
cat "$RUN_DIR/current-round.txt"
tail -n 250 "$RUN_DIR/server.log"
```

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_'
curl --noproxy '*' --connect-timeout 2 --max-time 5 -i \
  http://127.0.0.1:31000/fault_tolerance/status
```
