# NPU 原 rank 回归：真实模型服务验收

这轮验证单机 eager 服务的 `4→3→4`：原 rank 2 退出后，三卡先恢复生成，再由控制器拉起同编号新进程、重建通信和 DeepEP buffer、加载当前专家布局，最后重新开放四卡请求路由。需要重新启动带 `--enable-npu-ft-recovery` 的服务。

**状态（2026-10-30）：用户确认当前代码沿用原 `ops + dynamic + 执行图` 命令时缩容后可继续处理请求；此前 `default + static + eager` 验收在三卡首请求失败。完整 recover 尚未实机验证。** 两次实验的故障时机和 victim rank 也不同，尚不能锁定单一差异。[上一阶段通信探针](README_npu_rank_recovery.md) 的 HCCL/新 buffer 结果只覆盖其声明的范围。

## 当前优先实验：保留成功路径，只关闭执行图

用户最新回传：在**当前代码**使用原命令，`DEEP_USE_MODE=ops`、dynamic 路由、128 冗余专家、decode 图 batch size 为 1/2/4/8。在长流式请求执行中杀 rank 1，等 rank 0/2/3 unhealthy、rank 1 dead 后调用 scale_down，随后服务可正常生成。该报告没有附精确 SHA、完整服务端日志和本轮软件元组，按用户回传结论记录，不补造这些字段。

这说明不必再把“旧版原命令 vs 新版原命令”作为首要实验。当前应保留成功的 ops/dynamic/冗余专家、原环境和**长请求中杀 rank 1、等幸存者 unhealthy 后缩容**的步骤，独立重启四卡服务，在原启动命令中只新增：

```bash
--disable-cuda-graph
```

原 `--cuda-graph-bs-decode 1 2 4 8` 在全禁用模式下不再用于捕图。该轮不要同时新增 disable-overlap、recovery 开关或同步定位变量，不切换 default/static；服务日志另存，记录实际 SHA 和启动环境。若这一轮失败，差异定位到关闭图后的路径及其交互；若通过，再单独验证关闭 overlap，之后启用 recovery 开关及其要求的标准进程组配置。各阶段尚未实机执行。

自动验收脚本此前在短请求结束后杀 rank 2，只等待 dead 即提交缩容，与用户手动实验不同。不能将两次输出当作只改了策略的严格对照；当前优先重复手动注入方式。若在这一步改变 fault-rank、是否有在途请求或等待条件，应作为独立实验记录。

恢复配置已允许 DeepEP `ops/default` 和 `dynamic/static`，不再强制用户离开已跑通的 ops/dynamic 路径；eager、无 overlap 等恢复限制继续保留。该改动是解除无充分依据的配置限制，**不是已完成 ops 全流程实机恢复验证**。探针不传 elastic_info，服务会传；探针的无 elastic_info/EP=4 ops 失败不能推广为所有 EP=4 服务均不支持 ops。

用户还报告原流式 curl 需 Ctrl+C。当前能确认服务恢复后可处理请求，不能称原长请求自动续上。Ctrl+C 关闭客户端连接，并不能单独证明服务端请求状态已清理。受故障影响的旧请求应明确以错误结束，当前长连接未及时收尾属于待定位问题；请求实际落在哪个 DP rank、本轮 AbortReq/连接日志尚未知，不假设它一定归故障 rank 所有。

## 1. 更新代码并保留已跑通的环境

在 NPU 服务器的仓库根目录操作。下面切到远端分支的确定快照；Git 若提示本地修改冲突，先保留修改，勿强制覆盖。

```bash
conda activate sglang-ft
git fetch https://github.com/whisky286/sglang.git codex/npu-ft-recover
git switch --detach FETCH_HEAD
git rev-parse HEAD
git diff --stat
```

沿用上一轮通过的 CANN/Python/Torch/torch_npu/DeepEP 环境，不需要为本次改动升级它们。确认已在当前 shell 加载相同 CANN 环境，再导出之前修正过的自定义算子路径；这些变量必须在启动 Python **之前**设置。

```bash
DEEPEP_DIR=$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("deep_ep").origin).parent)')
DEEPEP_OPP="$DEEPEP_DIR/vendors/hwcomputing"
export LD_LIBRARY_PATH="$DEEPEP_OPP/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_CUSTOM_OPP_PATH="$DEEPEP_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export DEEP_USE_MODE=ops
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

这里保留用户完整服务已跑通的 `ops` 策略；下面的独立探针仍通过 `--mc2-strategy default` 选择它已验证的新 buffer 路径，两者范围不同。关闭消息队列广播，是为了让回归进程使用新建的 Gloo 控制组；旧广播队列没有重新入组协议。当前顶部的单变量缩容实验先沿用原环境，不要提前混入本节恢复专用变量。

可以先再次运行原通信探针，确认服务器环境仍正常；已经确认环境没有变化时可直接进入模型验收：

```bash
python test/manual/ep/probe_npu_rank_recovery.py \
  --backend hccl --world-size 4 --fault-rank 2 \
  --mc2 --mc2-strategy default --timeout 120
```

## 2. 启动一套专用测试服务

以下是后续 eager **恢复候选**命令，先完成顶部的单变量缩容实验再使用。保留已跑通的 ops/dynamic 及 128 冗余专家；使用之前能启动的非量化模型目录。测试会自动结束指定 Scheduler 进程。

```bash
MODEL_PATH=/absolute/path/to/Qwen3-30B-A3B
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --device npu --attention-backend ascend --dtype bfloat16 \
  --tp-size 4 --dp-size 4 --ep-size 4 \
  --moe-dense-tp-size 1 --moe-dp-size 1 --attn-cp-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --enable-eplb --eplb-algorithm elasticity_aware \
  --ep-dispatch-algorithm dynamic --ep-num-redundant-experts 128 \
  --elastic-ep-backend mc2 --enable-fault-tolerance \
  --enable-npu-ft-recovery \
  --fault-tolerance-timeout 600 --fault-tolerance-pause-timeout 900 \
  --disable-cuda-graph --disable-overlap-schedule \
  --mem-fraction-static 0.5 --max-total-tokens 4096 \
  --max-running-requests 8 --context-length 2048 \
  --host 127.0.0.1 --port 30000 \
  > /tmp/npu-recovery-server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > /tmp/npu-recovery-server.pid
```

Qwen3-30B-A3B 有 128 个逻辑专家；本轮保留用户验证的 128 个冗余槽位，总共 256 个物理槽位，每卡 64 个，三卡有 192 个，可以覆盖全部逻辑专家。此前 44 冗余槽位的示例是容量下限附近的另一配置，不用于替换本轮已通过参数。其他模型需按实际专家数核对容量。

查看启动日志：

```bash
tail -f /tmp/npu-recovery-server.log
```

按 Ctrl+C 只退出 `tail`，后台服务继续运行。也可在启动后直接运行下一步，脚本会等待健康接口就绪。

## 3. 自动完成两轮故障和回归

在同一环境、仓库根目录执行：

```bash
SERVER_PID=$(cat /tmp/npu-recovery-server.pid)
python test/manual/ep/verify_npu_recovery_serving.py \
  --server-pid "$SERVER_PID" --base-url http://127.0.0.1:30000 \
  --fault-rank 2 --cycles 2 --timeout 600 \
  --expected-text Paris \
  --output /tmp/npu-recovery-serving.json \
  > /tmp/npu-recovery-test.log 2>&1
TEST_EXIT=$?
cat /tmp/npu-recovery-test.log
echo "test exit code: $TEST_EXIT"
```

脚本只会结束明确指定的服务器 PID 下、匹配 rank 的 Scheduler 子进程。HTTP 地址必须是本机地址。若服务配置了 API key，在运行前导出 `SGLANG_API_KEY`。

每轮都会检查：

1. 初始四卡中，指定 rank 2 的请求能生成文本。
2. 结束 rank 2，等控制器确认进程退出，然后调用 `scale_down`。
3. 缩容完成后，指定幸存 rank 生成文本。
4. 调用 `recover`，等待重新加载权重、新通信组和新 buffer 的全员模型试跑完成。
5. 只有 rank 2 的 PID 改变，其他三卡 PID 不变；控制器报告四卡 healthy。
6. 再将请求指定给 rank 2，确认实际生成文本。
7. 对同一个 rank 再执行一轮，检查重建后的故障监测仍有效。

`--expected-text Paris` 给默认提示词 `The capital of France is` 增加一个简单语义检查。更换提示词时应同时调整预期文本；这不是完整的模型精度评测。

成功输出最后一行类似：

```json
{"success": true, "report": "/tmp/npu-recovery-serving.json"}
```

报告记录实际 Git SHA、软件包版本、每阶段的响应及 PID。测试进程退出码必须为 0；异常时 `success` 为 false，并保存已完成阶段和错误信息。失败后脚本保留服务现场；恢复事务一旦进入通信切换阶段失败，服务会按现有 FT 策略整体退出，不尝试伪回滚。

### 启动验收后没有输出，或状态接口正常但 `/server_info` 超时

2026-10-30 用户在 `12b479feb8` 对应流程报告验收没有输出：`/fault_tolerance/status` 返回 HTTP 200、四个 rank healthy，但单独请求 `/server_info` 在 8 秒内收到 0 字节并超时。精确实机 SHA 未另行回报。

本机代码核对发现：Scheduler 在开启 FT 后已改用各 DP 内的控制消息广播，而 DPC 仍只给第一个 DP leader 发送普通控制消息；`/server_info` 等待所有 Scheduler 的内部状态回复，因而可能一直等不到其余 rank。修正使 DPC 对 FT、scale joiner 与显式 local-control 配置都给每个活跃 DP leader 发送控制消息。修复后的 NPU 结果仍待重测。

旧验收脚本的第一条日志在 `/health` 和 `/server_info` 之后，且状态查询错误继承了完整恢复事务的等待预算（`--timeout 600` 时为 2430 秒）。新版在加载测试依赖前打印 `test_started`，为每次请求打印 `http_request_started/finished/failed`；只读状态查询使用 30 秒读取超时，生成使用 `--timeout`，FT apply 保留多阶段预算。JSON 报告随事件原子更新，异常或中断也保留阶段；不会自动重试可能已经接受的恢复操作。

遇到此类现象时，保留现场并在另一终端逐条执行下面命令。每个代码块是一条独立命令，末尾不要加续行反斜杠，也不要直接打印完整 `/server_info` 配置：

```bash
tail -n 30 artifacts/npu_ft/recovery-test.log
```

```bash
tail -n 100 artifacts/npu_ft/sglang-npu-ft.log
```

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -i http://127.0.0.1:31000/fault_tolerance/status
```

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 8 -sS -o /tmp/npu-server-info-check.json -w 'server_info HTTP=%{http_code} elapsed=%{time_total}s\n' http://127.0.0.1:31000/server_info
```

端口 31000、`artifacts/npu_ft` 对应这次用户实际启动命令；若采用本页最初的示例，应使用 30000 和对应 `/tmp` 日志路径。`status` 的 healthy 仅代表控制器当前记录，不代表所有 Scheduler 都响应控制请求。若验收已经出现 `scheduler_killed`，先检查服务/报告当前成员状态，再决定如何继续，避免重复故障注入。

### 四卡生成成功、缩容返回成功，但三卡首次生成返回 503

2026-10-30 19:35（Asia/Shanghai）的用户报告明确记录实机 SHA 为 `aa435b4ee06d31cdc370e994bb87f78a5039e487`：server_info 在 0.019 秒返回，rank 2 基线生成 16 tokens；结束 rank 2 后 scale_down 在 4.392 秒返回 success；随后的 rank 0 生成在 10.110 秒后返回 HTTP 503。这证明上一节控制消息分发的问题在本次运行中不再阻塞验收，**没有证明三卡服务或 recover 已通过**，报告尚未执行 recover 请求。

用户服务日志显示：三卡 Gloo 组和 EPLB 完成；提交后约 5 秒 rank 1/3 再次 MLP-sync 失败；约 10 秒 rank 0 报 NPU LocalScalarDense 的 copy_stream 同步超时（507046、EE1002），随后状态为 unhealthy/unhealthy/dead/unhealthy。现有日志只保留异常字符串，没有完整 Python 栈，不能仅凭最终超时判定具体模型算子或宣布最终根因。

后续代码使 NPU FT 的小量 MLP 同步元数据从构造、Gloo 汇总到活跃 mask/标量读取都留在 CPU，避免绕回 NPU 流；5 秒控制超时与 NPU 故障超时不变。Scheduler 的 NPU FT 异常新增完整 Python 调用栈；验收脚本在 HTTP 失败事件中保留 status_code、最多 4096 字符的响应正文及截断标记，不自动重试。

本机回归覆盖 NPU device 调用参数下仍不分配 NPU 元数据、单个 prefill rank 与 idle 幸存 rank 混合时的原 rank 槽位、503 具体原因保留和长度限制。这消除了一个已发现的控制面设备依赖；**后续实机复测仍失败，详见下节**。内部缩容健康门只跑 dummy decode，不能代替真实请求的 prefill/decode 验收。

### 22:42 复测：前向后的成员检查触发设备超时

用户确认本轮 Git SHA 为 `9c7c52cb8bddb84f991be7e957e3befb8d47f676`，包版本为 Torch 2.10.0、torch-npu 2.10.0.post2、sgl-kernel-npu 2026.6.1、DeepEP 1.0.0+7a396def.cann.9.0.0.b237。四卡生成成功，scale_down 在 4.159 秒返回 success；三卡首请求在 10.672 秒后返回 503，尚未调用 recover。

完整服务端栈显示 rank 0/1/3 均在 `ModelRunner.forward → _maybe_rebalance_after_rank_fault` 中，将 NPU 成员 mask 归约结果转成 Python bool 时，触发 `AclrtSynchronizeStreamWithTimeout(copy_stream)` 超时。此处位于 `_forward_raw` 之后；NPU 工作异步下发，所以该栈确认的是**暴露设备等待的位置**，不能单凭此栈认定成员检查就是底层根因。不能仅跳过 bool 读取、延长超时或重试请求就宣称修复。

用户另行读取实机已安装的 `DefaultLowLatencyCommStrategy` 源码，确认 dispatch 和 combine 都将可选 `elastic_info` 传给 `self.runtime`。因此排除 Python 策略层漏传；该输出没有展示 C++、ACLNN 或实际加载的自定义库，不能由此证明整个算子链正确。当前缩容仍保留原 MC2 数据域和旧 buffer，恢复探针的三卡阶段则使用全新候选组和 buffer，两者范围不同。

下一轮先保持上述代码、模型、冗余专家数和通信参数不变，在**启动新服务的终端、Python 启动前**设置：

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

这是 [TorchNPU 官方的同步定位开关](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/ASCEND_LAUNCH_BLOCKING.md)，用于尽量将异步错误定位到实际算子调用。它会改变执行时序并降低性能，只用于诊断；如果该模式通过，也不能据此宣布普通异步模式已修复。当前服务已经出现设备异常，先保存旧 server/test 日志和 JSON，再按原停止方式结束专用测试服务、重新启动；仅给客户端设置此变量不生效。

新服务仍使用端口 31000 时，在另一终端运行一轮并使用新文件名：

```bash
python -u test/manual/ep/verify_npu_recovery_serving.py \
  --server-pid "$(cat artifacts/npu_ft/server.pid)" \
  --base-url http://127.0.0.1:31000 \
  --fault-rank 2 --cycles 1 --timeout 600 \
  --output artifacts/npu_ft/recovery-report-sync.json \
  2>&1 | tee artifacts/npu_ft/recovery-test-sync.log
```

`server.pid` 必须记录这次重新启动的 Python 服务主进程；服务日志另存为 `artifacts/npu_ft/sglang-npu-ft-sync.log`。回传最早的完整 Python traceback、对应 rank/时间和客户端 JSON；若四卡基线也失败，保留该阶段，不能将新错误直接当作缩容根因。诊断后重新启动普通异步服务前执行 `unset ASCEND_LAUNCH_BLOCKING`。

### 对照实习结束时已验证的缩容基线

用户随后明确确认：实习于 2026-09-11 结束前，`npu-ft-0908` 的 **`46147ef9bc`** 已验证缩容后继续推理正常。该信息是用户提供的历史实机结果，应作为已知成功基线；不能将 10 月的新配置失败解释成原来没有实现或验证过缩容续推。

本机对比 `46147ef9bc..9c7c52cb8b`：没有合并上游；报错处的前向后成员 bool 检查、设备 stop/restart/reinit 顺序、MC2 elastic_info 格式、缩容最小专家布局及更新、eager/dummy runner 主体均原已存在。新四卡通信域安装与 buffer 退休方法只在 recover 阶段触发，本轮尚未执行；但共享控制逻辑、成员版本和启动配置有变化，不能据“尚未 recover”排除整个新增分支的影响。9c7c52cb8b 的 CPU MLP 修改发生在第一轮三卡失败之后，也不能解释第一轮失败的起因。

本次恢复验收要求关闭执行图、关闭 overlap、从 dynamic 改为 static 路由，并固定 DeepEP default；这些是本次提出的配置变化，不能未经核对认定为 9 月成功实验的配置。应先建立下列对照，再决定是否需要上节同步算子定位：

| 对照 | 代码 | 参数与环境 | 目的 |
|---|---|---|---|
| A | 46147ef9bc | 用户 9 月验证通过的命令；固定当前软件栈 | 确认历史成功条件在当前环境仍成立 |
| B | 新分支 | 与 A 相同，不启用新 recovery 开关 | 检查新增代码是否破坏原缩容功能 |
| C | 新分支 | 本次恢复要求的 eager/static 等参数 | A/B 都通过后，再逐项定位新配置或 recovery 开关差异 |

A 失败时先核对软件栈和旧命令差异；A 通过、B 失败时追查新增提交；A/B 通过而 C 失败时逐项切换配置。每次独立启动四卡服务、保存独立日志和报告，不在已缩容/异常的进程上重复注入。新恢复开关要求 eager，因此 B 必须去掉 `--enable-npu-ft-recovery` 才能保留原图配置。

验收脚本新增 `--scale-down-only --cycles 1`：只执行四卡生成、杀指定 rank、scale_down、幸存 rank 真实生成，检查幸存 PID/创建时间保持不变后结束。它直接用 FT status 识别初始成员，不要求恢复开关，也不调用旧版可能卡住的 server_info；不会发送 recover。成功 scope 明确只覆盖缩容。

测试旧版时，先在新分支把脚本复制到独立路径，再切换服务器源码（先保留本地修改，勿强制覆盖）：

```bash
cp test/manual/ep/verify_npu_recovery_serving.py /tmp/verify_npu_scale_down_baseline.py
```

用 A 或 B 对应命令启动服务后，在**运行该服务的源码仓库根目录**执行（主进程 PID、端口和输出标签按该轮替换）：

```bash
python -u /tmp/verify_npu_scale_down_baseline.py \
  --server-pid "$(cat artifacts/npu_ft/server.pid)" \
  --base-url http://127.0.0.1:31000 \
  --fault-rank 2 --scale-down-only --cycles 1 --timeout 600 \
  --output artifacts/npu_ft/scale-down-baseline-A.json \
  2>&1 | tee artifacts/npu_ft/scale-down-baseline-A.log
```

报告的 git_commit 是运行脚本时当前仓库的 HEAD，需与实际服务源码一致；脚本在 /tmp 并不决定服务代码版本。本机隔离回归累计 105 tests、64 subtests 通过，新增模拟旧服务端点、首请求失败、幸存 PID 改变和不健康初始状态的门禁。A/B/C 尚未在本轮 NPU 执行。

## 4. 失败或完成后回传的材料

```bash
git rev-parse HEAD > /tmp/npu-recovery-commit.txt
git diff > /tmp/npu-recovery-local.patch
python --version > /tmp/npu-recovery-python.txt
python -m pip show torch torch-npu sgl-kernel-npu deep-ep > /tmp/npu-recovery-packages.txt
npu-smi info > /tmp/npu-recovery-devices.txt
```

请带回 `/tmp/npu-recovery-server.log`、`/tmp/npu-recovery-test.log`、`/tmp/npu-recovery-serving.json` 和上面五个记录文件。失败时保留完整 traceback 及失败前的阶段，不只截最后一条 HCCL 错误。若卡住超过配置超时，也保留日志和进程/设备状态；不要反复重跑覆盖第一轮证据。

## 实现范围与本机证据

- 新增显式 `POST /fault_tolerance/apply` 请求：`{"instruction":"recover","params":{"ranks":[2],"timeout":600}}`。必须先完成缩容，且一次恢复全部已移除的原始 ranks；控制器自行启动新进程，不需要手动启动 joiner。
- 幸存者暂停并清理在途请求；以成员版本隔离新 WORLD/TP/EP HCCL、Gloo 组；回归进程按当前专家布局加载 checkpoint。DeepEP 缓存和旧 dispatch handle 在暂停边界被替换；全员通信与模型 logits 检查通过后才开放路由。
- 本机用隔离导入/AST 提取的实际模块运行 9 个相关测试文件：105 tests、73 subtests 通过。新增覆盖 ops/default 与 dynamic/static 配置组合，并保留非法策略、执行图、overlap 等门禁。此前检查涵盖控制事务、失败门禁、进程替换、buffer 清理、专家布局与旧版验收模式。F/I lint 与空白检查通过；本机不是完整 SGLang/NPU 环境。
- 本机另以真实多进程 Gloo 验证新的 `RecoveryGroups` 默认组替换和原 rank 映射：`4→3→4` 求和 `10→7→10`，由新进程补回 rank 2，由原 rank 2 发起的对象广播正确。它不能替代目标 torch-npu/HCCL 私有接口兼容性验证。
- 尚未验证：本页真实模型流程、多轮 NPU 显存占用、设备真实硬故障后的重新分配、并发压力和完整输出精度。旧故障组和 buffer 保留到进程退出，避免未经验证的销毁操作；这会增加多轮恢复的内存占用，不保证无限次恢复。
- 恢复候选允许单机、`TP=DP=EP`、Attention TP=1、eager、无调度重叠、DeepEP low_latency 的 default/ops、static/dynamic 路由、非量化模型；允许配置不等于实机通过。PD、多机、图恢复和超初始容量的 `4→6` 不在本轮验收范围。
