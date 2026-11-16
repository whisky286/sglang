# NPU 容错实验自动化

本脚本将[文章实验计划](PLAN_npu_ft_article_and_experiments.md)中的操作连接起来。运行入口是 `run_npu_ft_bench.sh`，辅助文件为同目录的 Shell、awk 和 jq 文件。NPU 服务器只需要 Bash、curl、jq、awk、ps、Linux procfs、flock、ss、sha256sum 和已有的 SGLang Python 环境；不依赖 Python 辅助验收程序，不安装或升级包。

适用范围：本分支 `/fault_tolerance/status` 的 `.ranks[].rank/state` 接口，单机四卡 NPU，TP=DP=EP=4、Attention TP=1，MC2/DeepEP，关闭执行图与调度重叠。公开 PR 的 `/v1/fault_tolerance/…` 接口不是这里的协议。模型默认使用已确认的 `/mnt/share/weights/DeepSeek-V2-Lite-Chat`，保留现有 CANN 和自定义算子环境。

## 自动控制的条件

每轮均由脚本启动独立服务，不接管历史服务。31000 已被占用时直接退出；不会停止端口上已有的服务。每个服务的启动参数、主进程 PID 和 Linux starttime 都会记录。

1. 确认四卡 healthy，查找本轮主进程的全部后代，从进程标题识别 `sglang::scheduler_DP1_TP1_EP1`。同一 rank 匹配多个进程、拓扑不符、主进程身份变化时停止。
2. 向 rank 1 发送固定输入、输出上限 1536 token、`ignore_eos=true` 的流式请求。默认收到第一个输出 token 后立即准备故障注入，不等待人工观察，不用固定 sleep 猜测生成进度。
3. kill 前再次核验祖先关系、进程标题、PID 和 starttime。已经观测到长请求结束时，不发送 kill；本轮按未满足触发条件处理。记录发信号时已经观测到的输出量。
4. 发送一次 SIGKILL，然后轮询状态。**必须 rank 1 为 dead，0/2/3 全部为 unhealthy，才允许提交 scale_down。** 部分 rank 仍 healthy、状态接口不可用或等待超时，都不会发送缩容。
5. 只提交一次控制指令。HTTP 超时、空响应或失败后不自动重发，以免重复执行已经被服务端受理的操作。
6. 缩容后确认三卡 healthy，并实际生成；恢复后确认四卡 healthy，核对只有 rank 1 更换 PID/starttime，并把新生成请求明确发给 rank 1。

不能完全消除客户端观测与设备执行之间的时间差，因此还会检查故障长请求最终结果：正常完成、等待到客户端超时或采集错误，都不能作为有效的在途故障实验通过。

## 首次运行：一轮功能预试

在 NPU 机器的仓库根目录、原有 `sglang-ft` 环境执行。先确认 31000 空闲；输出不为空时不要执行启动命令。

```bash
ss -H -ltnp 'sport = :31000'
```

端口空闲后，整块执行：

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}" \
bash test/manual/ep/run_npu_ft_bench.sh cycle --rounds 1
```

预期完成：四卡生成 → rank 1 输出期间退出 → 等待 1 dead、0/2/3 unhealthy → 三卡生成 → 补回 rank 1 → 四卡 healthy、指定新 rank 1 生成。脚本成功后停止自己启动的服务；失败时保留服务、日志和目录，不自动从下一步继续。

如果确实要复用原方案的“观察到至少256 token后注入”，使用下面的完整命令。两种阈值属于不同实验配置，不混合汇总。

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}" \
FT_FAULT_TOKENS=256 \
bash test/manual/ep/run_npu_ft_bench.sh cycle --rounds 1
```

## 三组实验

以下为独立运行命令；每条执行前端口须空闲。默认1轮用于预试，下列正式命令各做5轮。`all --rounds 5` 会顺序完成全部组；先确保预试和指标采集符合实际环境，再开始正式运行。

实验1：相同故障和流量，对比整实例重启与故障缩容。两种策略都等到相同状态条件满足后触发，奇偶轮交换顺序。故障后发送负载600秒，所有请求再等到各自终态或期限。缩容组在这段窗口保持三卡，重启组为四卡；补卡不混入这组耗时。

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}" \
FT_LOAD_RATE=0.2 \
bash test/manual/ep/run_npu_ft_bench.sh recovery --rounds 5 --window 600
```

实验2：故障前四卡、缩容后三卡、补回四卡，使用同一阶梯请求速率。每档至少120秒，按目标200个样本延长，最长600秒。各档之间确认前一档已经结束，并通过空闲时的 `/flush_cache` 返回确认无遗留任务；prefix cache 本身关闭。

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}" \
bash test/manual/ep/run_npu_ft_bench.sh capacity --rounds 5 \
  --rates 0.1,0.2,0.4,0.8,1.6,3.2 --step 120 --min-samples 200
```

这些速率是预试起点，不代表已找到容量边界。若全部通过，结果只能称“最高已验证请求率”；应扩大速率网格后，以相同配置重做三种状态。若三卡和四卡都达不到某档，也保留原始结果。服务端默认 `max-running-requests=8`、`max-total-tokens=4096` 可能先形成限制；需放宽时，通过环境变量生成新的实验配置，三种状态全部重测。

没有外部时延门槛时，首轮健康四卡预热后，运行20条串行256输出请求，使用两项 P95 各乘2作为实验门槛，后续轮次保持不变。正式业务数据可使用下面的显式门槛命令；这里的数值仅为命令格式示例，需要按实际业务门槛登记，不能作为厂商 SLO 引用。

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}" \
FT_TTFT_LIMIT=2 FT_TPOT_LIMIT=0.1 \
bash test/manual/ep/run_npu_ft_bench.sh capacity --rounds 5 \
  --rates 0.1,0.2,0.4,0.8,1.6,3.2 --step 120 --min-samples 200
```

实验3：健康三卡运行1536 token长流，观察到256 token后发起补卡，与不补卡的长流对照。补卡期间每秒发起一个新接入探针，保存拒绝和恢复接入记录。另用独立新服务测试1秒排空超时：返回408、三卡状态及 PID不变、已有流继续，新请求重新可接入。每个用例都从新启动的四卡服务开始，通过故障缩容进入三卡。

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}" \
bash test/manual/ep/run_npu_ft_bench.sh drain --rounds 5
```

全部实验顺序运行：

```bash
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/DeepSeek-V2-Lite-Chat}" \
bash test/manual/ep/run_npu_ft_bench.sh all --rounds 5 \
  --rates 0.1,0.2,0.4,0.8,1.6,3.2 --window 600 --step 120 --min-samples 200
```

## 数据与口径

输出位于新的 `artifacts/npu_ft/bench-日期-进程号/`，不覆盖已有实验，也不改写手动验收的 `current-recovery-dir.txt`。

- `manifest.json`、提交、工作树差异、依赖、NPU信息、启动参数：固定实验环境。脚本将本仓库 `python/` 放在加载路径前面，确保启动当前源码。
- `input-ids.json`：通过服务本身的 `/v1/tokenize` 取得输入，截取256 token并保存校验值，后续全部请求复用。输出固定256 token，温度0，忽略EOS；长流输出1536 token，恢复探针16 token，接入探针1 token。
- `events.jsonl`、`status-observations.jsonl`：故障、首次故障上报、满足全部 unhealthy条件、指令提交/返回、实际生成结束。每次状态查询保留原响应。
- `pids-*.json`：只属于本轮服务的 rank/PID/starttime。kill前再次查找，恢复后核对不变的进程。
- 每条请求目录：原始SSE、逐行接收时间、逐事件累计token数、curl退出码、HTTP状态、终态分类和请求汇总。HTTP200后abort不算成功；缺少终态、长度不符、采集错误分别保留。
- 各档 `summary.json`：请求正常完成率、错误分类、输出tokens/s、P95首token时延、P95每token平均耗时、实际发送相对计划的最大延迟。只有样本数足够、成功率至少99%、时延达标且客户端发包没有明显落后，才标记该点通过。
- 补卡的 `recovery-log-events.jsonl`、`phases-summary.json` 和 `admission-summary.json`：排空日志观察时间、新请求拒绝和恢复接入的探针边界。
- 根目录 `summary.json`：各轮结果、恢复时间中位数及范围、容量保留率统计。失败立即停止，已有轮次和失败现场保留；失败轮不能从正式报告中删掉。

恢复时间从发送kill信号前的记录时刻，计算到处置完成后发起的新请求正常生成结束；不是HTTP控制请求返回时间。记录首次故障状态和全部 unhealthy 就绪两种时刻，避免把检测与等待其他rank暂停混为一段。

请求集合按发送时刻归入窗口，等待最终结果。有效输出吞吐按窗口内收到的token增量计算，只计入最终正常完成的请求；跨窗口请求只计本窗口收到的部分。探针、故障触发流和主负载分开保存，不混入主负载吞吐。生成质量不由固定长度合成负载证明。

全部客户端事件使用 `/proc/uptime` 单调时钟，**时间分辨率约10ms**。SSE一个事件可能携带多个token，以累计token数的增量计数；最长无输出间隔表示客户端相邻token增量事件的间隔。日志观察器额外有约50ms轮询及调度延迟，接入探针约1秒采样间隔；内部微观性能分析需另用更高精度工具。

压测采用固定发送时间表，不因服务响应变慢而主动降低请求率。超过客户端并发上限时停止并报错；实际发包延迟超过默认0.25秒时，该点不会被标记为达标，避免把客户端能力误当作服务容量。

## 验证状态

本机已检查 Shell 语法、jq指标计算及模拟的HTTP/SSE控制流程，包括未达到全部unhealthy时禁止POST、PID身份变化或请求结束后禁止kill、HTTP200+abort和丢失终态等情况。CPU测试使用 `write-sglang-test` 规范，位于 `test/registered/unit/fault_tolerance/test_npu_ft_bench.py`。本机没有Torch/NPU环境，测试隔离加载仓库的 `CustomTestCase` 并使用模拟HTTP/进程数据；**尚未运行本脚本的NPU完整流程或得到任何新增性能数据**。

在已有开发测试环境中，可执行：

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m unittest discover -s test/registered/unit/fault_tolerance \
  -p test_npu_ft_bench.py -v
```

失败时回传对应实验目录的 `failure.json`、`events.jsonl`、最后状态、server.log最早完整异常、原始请求结果及PID快照。不要在未知结果的控制操作后重复发送缩容或恢复。
