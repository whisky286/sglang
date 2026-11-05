# 第 0 步：成功基线连续两轮恢复（手动）

固定代码 3060ee795527680b98752fb2cb719ab59dcf2fe5。用户已确认单轮通过；本页新增同一服务双轮验证。此页不是新增实机通过结果。

在 NPU 仓库根目录、原 sglang-ft 环境执行。保留已有 CANN/自定义算子设置，不升级包。下面全部是手动 Shell/curl/ps/jq 操作。

终端 1：先停止本轮旧后台服务，保存文件保留。若没有 current-recovery-dir.txt，跳过停止块。Ctrl+C 退出 tail 不会停止后台服务。

```bash
if [ -f artifacts/npu_ft/current-recovery-dir.txt ]; then
  OLD_RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
  if [ -f "$OLD_RUN_DIR/server.pid" ]; then
    OLD_SERVER_PID=$(cat "$OLD_RUN_DIR/server.pid")
    if ps -p "$OLD_SERVER_PID" -o args= | grep -q '[s]glang.launch_server'; then
      kill -TERM "$OLD_SERVER_PID"
    fi
  fi
fi
```

单独检查：31000 没有监听后再继续；仍监听时先处理旧服务，不启动第二份。

```bash
ss -ltnp 'sport = :31000'
```

终端 1：拉取并固定到本页提交；Git 报本地修改冲突时保留修改，停止后续启动，不强制覆盖。

```bash
git fetch https://github.com/whisky286/sglang.git codex/npu-ft-recover &&
git switch --detach 3060ee795527680b98752fb2cb719ab59dcf2fe5 &&
git rev-parse HEAD
```

更新成功后再启动：

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}"
RUN_DIR="artifacts/npu_ft/request-baseline-$(date +%Y%m%d-%H%M%S)"
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
  --enable-fault-tolerance --enable-npu-ft-recovery \
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

服务就绪后，终端 3 确认四卡 healthy：

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
  http://127.0.0.1:31000/fault_tolerance/status | jq .
```

## 第 1 轮：同一服务，不重启

终端 3：创建本轮目录，保存本次真实 PID。根据输出核对 rank 1，不复用旧实验的 PID。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR="$RUN_DIR/cycle-1"
mkdir -p "$ROUND_DIR"
printf '%s\n' "$ROUND_DIR" > "$RUN_DIR/current-round.txt"
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$ROUND_DIR/pids-before.txt"
```

终端 2：启动长流式请求，看到持续输出后回到终端 3。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 900 -N -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a detailed numbered explanation of distributed inference.","stream":true,"sampling_params":{"temperature":0,"max_new_tokens":1536,"ignore_eos":true}}' \
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
    if ps -p "$RANK1_PID" -o args= | grep -q 'sglang::scheduler_DP1_TP1_EP1'; then
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

基线代码中旧长连接可能不自动结束。三卡新请求通过后，可以在终端 2 Ctrl+C 结束旧 curl；这不算自动收尾通过。继续前先确认没有旧客户端仍在发新请求。

终端 3：只发一次 recover，不重试可能已接受的事务。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
if jq -e '.success == true' "$ROUND_DIR/scale-down.json" >/dev/null &&
   jq -e '.meta_info.completion_tokens > 0 and (.text | length > 0) and ((.meta_info.finish_reason.type? // "") != "abort")' "$ROUND_DIR/survivor.json" >/dev/null; then
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

四卡 healthy 后指定新 rank 1 生成：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","routed_dp_rank":1,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$ROUND_DIR/recovered.json" | jq .
```

只有生成有效 token、finish_reason 非 abort、状态与进程身份均符合，才记录本轮通过。

## 第 2 轮：同一服务，不重启

终端 3：创建本轮目录，保存本次真实 PID。根据输出核对 rank 1，不复用旧实验的 PID。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR="$RUN_DIR/cycle-2"
mkdir -p "$ROUND_DIR"
printf '%s\n' "$ROUND_DIR" > "$RUN_DIR/current-round.txt"
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$ROUND_DIR/pids-before.txt"
```

终端 2：启动长流式请求，看到持续输出后回到终端 3。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 900 -N -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a detailed numbered explanation of distributed inference.","stream":true,"sampling_params":{"temperature":0,"max_new_tokens":1536,"ignore_eos":true}}' \
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
    if ps -p "$RANK1_PID" -o args= | grep -q 'sglang::scheduler_DP1_TP1_EP1'; then
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

基线代码中旧长连接可能不自动结束。三卡新请求通过后，可以在终端 2 Ctrl+C 结束旧 curl；这不算自动收尾通过。继续前先确认没有旧客户端仍在发新请求。

终端 3：只发一次 recover，不重试可能已接受的事务。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
if jq -e '.success == true' "$ROUND_DIR/scale-down.json" >/dev/null &&
   jq -e '.meta_info.completion_tokens > 0 and (.text | length > 0) and ((.meta_info.finish_reason.type? // "") != "abort")' "$ROUND_DIR/survivor.json" >/dev/null; then
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

四卡 healthy 后指定新 rank 1 生成：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","routed_dp_rank":1,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$ROUND_DIR/recovered.json" | jq .
```

只有生成有效 token、finish_reason 非 abort、状态与进程身份均符合，才记录本轮通过。

两轮均通过后再执行第 1 步。第二轮前重新查询 PID，不用第一轮被杀的 PID。

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
