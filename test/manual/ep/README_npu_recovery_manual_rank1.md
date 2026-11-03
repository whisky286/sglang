# NPU ops/dynamic 手动长请求故障与原 rank 1 恢复验收

用户已回传 ops/dynamic、128 冗余、关闭执行图与关闭调度重叠下原缩容正常；完整 4→3→4 尚未实机通过。此轮在这个组合上启用恢复开关，并满足恢复所需的标准通信组设置。保持长请求中杀 rank 1、等待幸存者 unhealthy 后再缩容的步骤，不使用默认脚本的空闲杀 rank 2 流程。

在 NPU 服务器原 sglang-ft 环境、仓库根目录执行。先在旧服务终端按 Ctrl+C 停止旧服务；下面的终端 1、2、3 都使用同一仓库和环境。沿用已设置的 MODEL_PATH、CANN 与自定义算子库路径，不升级包。

## 1. 更新代码并启动带恢复开关的新四卡服务（终端 1）

先执行更新；若 Git 提示本地修改冲突，保留修改，不强制覆盖。确认更新成功再执行启动块。

```bash
git fetch https://github.com/whisky286/sglang.git codex/npu-ft-recover
git switch --detach FETCH_HEAD
git rev-parse HEAD
```

以下命令将服务放到后台，保存的是 Python 主进程 PID；tail 用于看日志，之后 Ctrl+C 退出 tail 不会停止后台服务。每轮使用新目录，后续终端从 current-recovery-dir.txt 读取该目录。

```bash
RUN_DIR="artifacts/npu_ft/recover-rank1-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > artifacts/npu_ft/current-recovery-dir.txt
git rev-parse HEAD > "$RUN_DIR/commit.txt"

DEEP_USE_MODE=ops \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0 \
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --device npu \
  --attention-backend ascend \
  --dtype bfloat16 \
  --tp-size 4 \
  --dp-size 4 \
  --ep-size 4 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-dense-tp-size 1 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-eplb \
  --eplb-algorithm elasticity_aware \
  --ep-dispatch-algorithm dynamic \
  --ep-num-redundant-experts 128 \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance \
  --enable-npu-ft-recovery \
  --fault-tolerance-on-error-strategy pause \
  --fault-tolerance-timeout 600 \
  --fault-tolerance-pause-timeout 600 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --mem-fraction-static 0.5 \
  --max-running-requests 8 \
  --max-total-tokens 4096 \
  --context-length 2048 \
  --watchdog-timeout 300 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 31000 \
  > "$RUN_DIR/server.log" 2>&1 &
printf '%s\n' "$!" > "$RUN_DIR/server.pid"
tail -f "$RUN_DIR/server.log"
```

两个进程组环境变量是恢复实现所需设置：禁用旧消息队列广播，统一使用可重建的标准组；不表示其与恢复开关的影响已经独立实测。若启动失败，不继续故障注入，保留 server.log 和 commit.txt。

## 2. 启动原来的长流式请求（终端 2）

服务就绪后执行，看到持续输出时在终端 3 执行下一步。

```bash
curl --noproxy '*' -N http://127.0.0.1:31000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Write a detailed numbered explanation of how distributed inference works. Keep generating until you reach the token limit."}],
    "max_tokens": 1536,
    "temperature": 0,
    "stream": true
  }'
```

## 3. 手动核对四卡 PID，再结束本次服务的 rank 1（终端 3）

先看本轮主进程和 Scheduler 列表，保存用于恢复后人工对照。根据本轮启动日志与父进程关系确认 rank 1 的 PID，不复用旧实验的数字；同机其他服务不属于本轮。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$RUN_DIR/pids-before.txt"
```

看到长请求持续输出时，输入刚确认的本轮 rank 1 PID 后按回车：

```bash
printf '输入本轮 rank 1 的 PID：'
read -r RANK1_PID
kill -9 "$RANK1_PID"
```

查询状态；等 rank 1 dead 且 0/2/3 unhealthy 后，再执行缩容块。状态未达到时可重复这条只读查询，不重复杀进程。

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
  http://127.0.0.1:31000/fault_tolerance/status | jq .
```

## 4. 缩容并验证三卡真实生成（终端 3）

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
curl --noproxy '*' --connect-timeout 5 --max-time 2430 -sS \
  http://127.0.0.1:31000/fault_tolerance/apply \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"scale_down","params":{"ranks":[1],"timeout":600}}' \
  | tee "$RUN_DIR/scale-down.json" | jq .
```

缩容返回 success=true 后，执行指定幸存 rank 0 的新请求：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","routed_dp_rank":0,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$RUN_DIR/survivor-generation.json" | jq .
```

若终端 2 的旧连接仍挂住，可 Ctrl+C 结束该 curl；这不作为旧请求自动终止或自动续推通过的证据。新请求失败时不继续 recover。

## 5. 补回 rank 1（终端 3）

下列块先检查缩容成功与幸存 rank 生成有效 token，之后才发送 recover，不重试已提交的恢复操作。恢复可能需几分钟，等待这一条请求返回，不重复发送。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
if jq -e '.success == true' "$RUN_DIR/scale-down.json" >/dev/null && \
   jq -e '.meta_info.completion_tokens > 0 and (.text | length > 0) and ((.meta_info.finish_reason.type? // "") != "abort")' \
     "$RUN_DIR/survivor-generation.json" >/dev/null; then
  curl --noproxy '*' --connect-timeout 5 --max-time 2430 -sS \
    http://127.0.0.1:31000/fault_tolerance/apply \
    -H 'Content-Type: application/json' \
    -d '{"instruction":"recover","params":{"ranks":[1],"timeout":600}}' \
    | tee "$RUN_DIR/recover.json" | jq .
else
  printf '%s\n' 'STOP: scale-down or survivor generation did not pass; recover was not sent.' >&2
fi
```

## 6. 核对只有 rank 1 换进程，并指定它生成（终端 3）

先确认恢复返回 success=true、四 rank 均 healthy，再人工对照两个 PID 列表：本轮 rank 0/2/3 不变，只有 rank 1 换进程。此处不运行 Python 自动校验。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
cat "$RUN_DIR/recover.json" | jq .
cat "$RUN_DIR/pids-before.txt"
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$RUN_DIR/pids-after.txt"
```

然后明确把新请求交给 rank 1：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","routed_dp_rank":1,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$RUN_DIR/recovered-rank-generation.json" | jq .
```

人工确认 text 非空、completion_tokens 大于 0、finish_reason 不是 abort，才将这轮标为通过。若未保存 pids-before.txt，不补造进程未重启的证据。

成功只覆盖一轮手动故障与补回原 rank，非完整精度评测，也未覆盖第二次故障、图恢复或旧请求自动收尾。失败后保留本轮目录，不重发 recover；回传 server.log 最早完整 traceback、scale-down.json、survivor-generation.json、recover.json 与 commit.txt。若无法读取恢复响应，应先检查状态和服务日志，不把客户端超时当作恢复未执行。

本机只对以上 Shell 与 JSON 请求体做语法检查，未执行这些故障或恢复命令。当前文档不是 NPU 通过结果。此前嵌入 Python 的辅助校验已按用户偏好改为手动命令；这不会改变服务端的恢复实现。

## recover 返回 curl (52) Empty reply from server

2026-11-03 用户在第 5 步回传 curl (52)，未收到恢复 HTTP 正文。恢复进度、主进程是否仍存活、实际提交与底层原因等待日志确认。当前 FT manager 在 apply 异常时会记录错误并结束服务进程树，这是候选解释，不是本轮已证实根因。不要重发 recover 或重启后覆盖现场；先逐组执行只读检查：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
tail -n 250 "$RUN_DIR/server.log"
```

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -i \
  http://127.0.0.1:31000/fault_tolerance/status
```

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
cat "$RUN_DIR/commit.txt"
```
