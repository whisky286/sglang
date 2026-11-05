# 第 1 步：请求可靠收尾与主动补卡排空（手动）

新增实现提交为 63f9fbb35de68ea126338369573788ebfd8d1705。如果 stage0 已在 pull 后启动的新代码上完成两轮，当前服务包含本提交且四卡健康，可以沿用，无需重启；从 A 入口继续。只有旧服务尚未加载新实现时，才走 B 入口更新、重启。本机检查通过；用户已回传长请求正常完成、排空完成、recover 提交成功和四卡 healthy，核心排空路径符合预期，其他终验项仍按本文检查。本批不实现故障请求自动续推：故障请求明确结束；健康三卡主动补卡时已有请求正常完成。

在 NPU 仓库根目录、原 sglang-ft 环境执行。保留已有 CANN/自定义算子设置，不升级包。下面全部是手动 Shell/curl/ps/jq 操作。

## A. 沿用已经运行新实现的四卡服务

适用于 pull 后启动服务，并在该服务上完成 stage0 两轮的情况。只 pull 不会更新旧 Python 进程；不要用当前 git HEAD 代替启动身份。以下检查当前主进程、四卡状态和本轮日志里的新增排空路径。检查失败只停在这里，不会停止服务，也不会自动重启；缺少日志或身份不清时先回传信息。

终端 3：整块执行。成功后创建独立 stage1 目录，原服务继续运行、日志仍写到原文件；stage0 的 cycle-1/cycle-2 完整保留。

```bash
ACTIVE_RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
STATUS_FILE="$ACTIVE_RUN_DIR/stage1-reuse-check-$(date +%Y%m%d-%H%M%S).json"
cat "$ACTIVE_RUN_DIR/commit.txt"
ps -p "$(cat "$ACTIVE_RUN_DIR/server.pid")" -o pid,ppid,stat,args
grep 'NPU FT recovery drain' "$ACTIVE_RUN_DIR/server.log" | tail -n 12
if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
     http://127.0.0.1:31000/fault_tolerance/status -o "$STATUS_FILE" &&
   jq -e '(.ranks | length) == 4 and all(.ranks[]; .state == "healthy")' "$STATUS_FILE" >/dev/null &&
   ps -p "$(cat "$ACTIVE_RUN_DIR/server.pid")" -o args= | grep -q '[s]glang.launch_server' &&
   grep -q 'NPU FT recovery drain complete' "$ACTIVE_RUN_DIR/server.log" &&
   test -f "$ACTIVE_RUN_DIR/commit.txt" &&
   git merge-base --is-ancestor 63f9fbb35de68ea126338369573788ebfd8d1705 "$(cat "$ACTIVE_RUN_DIR/commit.txt")"; then
  RUN_DIR="artifacts/npu_ft/request-drain-reuse-$(date +%Y%m%d-%H%M%S)"
  REAL_SERVER_LOG=$(readlink -f "$ACTIVE_RUN_DIR/server.log")
  if mkdir "$RUN_DIR" &&
     cp "$ACTIVE_RUN_DIR/server.pid" "$RUN_DIR/server.pid" &&
     cp "$ACTIVE_RUN_DIR/commit.txt" "$RUN_DIR/commit.txt" &&
     ln -s "$REAL_SERVER_LOG" "$RUN_DIR/server.log"; then
    printf '%s\n' "$ACTIVE_RUN_DIR" > "$RUN_DIR/service-source-dir.txt"
    printf '%s\n' "$RUN_DIR" > artifacts/npu_ft/current-recovery-dir.txt
    printf '%s\n' 'READY：沿用当前服务；跳过 B，从“故障请求自动结束与缩容”开始。'
  else
    printf '%s\n' 'STOP：目录准备失败，未切换本轮目录；服务未停止。'
  fi
else
  printf '%s\n' 'STOP：未确认可沿用，请回传本块输出；服务未停止，不继续故障注入。'
fi
```

仅看到 READY 时继续本文“故障请求自动结束与缩容”部分。不要执行下面 B 的停止/启动块；后续终端都会从 current-recovery-dir.txt 读取新验证目录。

## B. 从头更新并启动服务（已有服务包含新实现时跳过）

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
git switch --detach 63f9fbb35de68ea126338369573788ebfd8d1705 &&
git rev-parse HEAD
```

更新成功后再启动：

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}"
RUN_DIR="artifacts/npu_ft/request-drain-$(date +%Y%m%d-%H%M%S)"
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

## 故障请求自动结束与缩容

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

新版本中终端 2 的故障长请求应明确结束，出现 abort/503 或错误事件，无需手动 Ctrl+C。终端 3 在三卡生成通过后确认这一点；如仍挂住，先停止验收并回传日志。

## 验证排空超时和正常排空

此时停在健康三卡，先不 recover。终端 2：启动新的三卡长请求；结束后终端自动记录时间与 curl 退出码。ignore_eos 用于避免请求过早自然结束，不作为输出质量评测。

从左花括号到右花括号整块复制执行，让完成时间与退出码随请求结束一起记录。分组粘贴时 Shell 显示 > 是在等末尾右花括号，完整粘贴后才开始执行。如果旧轮缺少 drain-client-end.txt，保留缺失事实，不用当前时间补写，也不需要为补计时重跑已完成请求。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
{
  date -Ins > "$ROUND_DIR/drain-client-start.txt"
  curl --noproxy '*' --connect-timeout 5 --max-time 900 -N -sS \
    http://127.0.0.1:31000/generate \
    -H 'Content-Type: application/json' \
    -d '{"text":"Explain distributed inference in detail.","routed_dp_rank":0,"stream":true,"sampling_params":{"temperature":0,"max_new_tokens":1536,"ignore_eos":true}}' \
    | tee "$ROUND_DIR/drain-stream.sse"
  DRAIN_CURL_EXIT=${PIPESTATUS[0]}
  date -Ins > "$ROUND_DIR/drain-client-end.txt"
  printf 'curl exit=%s\n' "$DRAIN_CURL_EXIT" | tee "$ROUND_DIR/drain-client-exit.txt"
}
```

看到长请求持续输出时，终端 3：仅这次使用 1 秒排空期限。预期 HTTP 408、success=false、recover_drain_timeout；不拉新进程、不切组。它是预期的排空失败测试，确认结果后才发下一条正常 recover。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 5 --max-time 15 -sS \
  http://127.0.0.1:31000/fault_tolerance/apply \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"recover","params":{"ranks":[1],"timeout":600,"drain_timeout":1}}' \
  -o "$ROUND_DIR/drain-timeout.json" -w 'HTTP=%{http_code}\n'
cat "$ROUND_DIR/drain-timeout.json" | jq .
curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
  http://127.0.0.1:31000/fault_tolerance/status | jq .
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' \
  | tee "$ROUND_DIR/pids-after-drain-timeout.txt"
```

此时 0/2/3 应仍 healthy、1 dead，原长请求继续输出，只有三张卡的原 PID。若返回成功/空响应/其他错误，停止并回传，不继续。

确认上述预期后，趁同一个长请求仍在输出，终端 3 发正常 recover：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
date -Ins > "$ROUND_DIR/recover-start.txt"
curl --noproxy '*' --connect-timeout 5 --max-time 3030 -sS \
  http://127.0.0.1:31000/fault_tolerance/apply \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"recover","params":{"ranks":[1],"timeout":600,"drain_timeout":600}}' \
  | tee "$ROUND_DIR/recover.json" | jq .
date -Ins > "$ROUND_DIR/recover-end.txt"
```

这条等待期间，终端 4：确认排空开始时 pending_requests=True。若为 False，本轮只覆盖空闲恢复，不算长请求排空通过。可以等回归四卡后再组织下一轮，不能在事务进行中再发 recover。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
grep 'NPU FT recovery drain' "$RUN_DIR/server.log" | tail -n 12
```

仍处于 drain started 且未 complete 时，终端 4 发一个新请求，预期 HTTP 503，表示新入口已关闭；这不会打断终端 2 已接受的长请求：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
curl --noproxy '*' --connect-timeout 2 --max-time 8 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","sampling_params":{"temperature":0,"max_new_tokens":1}}' \
  -o "$ROUND_DIR/admission-during-drain.json" -w 'HTTP=%{http_code}\n'
cat "$ROUND_DIR/admission-during-drain.json" | jq .
```

预期顺序：终端 2 长请求自然完成（不用 Ctrl+C）→ drain complete → 新 rank 加载 → recover 成功。请求暂停接入会持续到成员恢复完成。

结束后，终端 2/4 查看原连接最终事件，completion_tokens 应为 1536、finish_reason 为 length，不能为 abort。这个检查忽略 SSE 的非 JSON 行：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ROUND_DIR=$(cat "$RUN_DIR/current-round.txt")
sed -n 's/^data: \({.*}\)$/\1/p' "$ROUND_DIR/drain-stream.sse" \
  | jq -s 'map(select(.meta_info.finish_reason != null)) | last | {finish_reason:.meta_info.finish_reason,completion_tokens:.meta_info.completion_tokens}'
cat "$ROUND_DIR/drain-client-exit.txt"
cat "$ROUND_DIR/drain-client-start.txt" "$ROUND_DIR/drain-client-end.txt" \
  "$ROUND_DIR/recover-start.txt" "$ROUND_DIR/recover-end.txt"
```

recover-end.txt 只表示 curl 命令结束，不单独证明恢复成功。还需 recover.json 的 success=true、服务日志在本次 pending_requests=True 之后出现 drain complete，以及下面的四卡状态/PID/新 rank 生成检查；不要把前几轮 pending_requests=False 的 drain complete 当成本轮结果。
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

本页通过后才推进请求日志/尝试隔离、自动重试与流式续推。当前不得把“故障时明确结束”记录成“故障后自动续推”。

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
