# TB2.1 周期评测与 TensorBoard

## 目标与边界

- TB2.1 全部89题只用于评测，不从NexTask训练集划分验证集。
- 固定官方registry版本 `terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`。
- 使用Harbor 0.21.0原生Task/Trial/verifier与既有E2B环境适配，绝不调用NexTask pairwise verifier，也不把solution/tests加入模型prompt。
- 复用训练的TerminalToolAgent、bash工具、系统提示和上下文机制，只把公开任务指令替换为TB2.1 instruction；评测预算单独配置。这是TB2.1任务上的本项目harness分数，不冒充官方其他Agent的leaderboard结果。
- TB2.1单题做题预算独立于RL采样，默认 `TB21_AGENT_TIME_BUDGET_SEC=21600`（6小时）；Harbor看门狗额外保留60秒收尾。整题墙钟默认32400秒，覆盖E2B创建、Agent、原生verifier及清理，且必须严格大于Agent看门狗。
- 采样对齐step90评测为temperature1.0/top_p0.95/top_k20、单轮65536 token及high reasoning。`TB21_MAX_TURNS=100000`只作为失控保护，正常终止由6小时时间预算决定；Agent setup统一3600秒、verifier统一1800秒，任务环境构建600秒，NexAU E2B keepalive默认最长24小时。
- 当前TerminalToolAgent尚无Claude Code在196608 token触发的语义摘要压缩。训练轨迹prefix merging只减少训练表示重复，不会缩短下一轮推理prompt；在真正实现并验收摘要闭包前，不能宣称上下文管理与step90完全对齐，也不使用任意丢弃旧消息或“保留最近5轮”冒充压缩。

## 接入方式

TensorBoard使用已有 `--use-tensorboard`，独立 `TENSORBOARD_DIR=$RUN_ROOT/tensorboard` 传入Ray runtime env。默认开启，缺依赖在提交前失败。

`--eval-config` 固定全量数据、每题1次和采样参数，不进入训练数据源/动态过滤/GRPO group。最初的同步模式通过OpenAIAdapter调用当前已同步权重；默认方式现已改为下文的固定checkpoint后台评测。Harbor在独立Python3.13子进程执行，以隔离镜像Python3.12及冻结NexTask依赖。

准备工具从官方registry解析固定版本，将89个任务包缓存到共享存储，记录每个文件SHA及官方task digest，导出eval-only JSONL和manifest。启动与每次评测均验证完整89题集合、文件digest和指令，禁止静默过滤长prompt或只汇总成功启动的子集。

每个trial保留原生result、verifier输出和独立sandbox清理回执。基础设施异常不视为模型0分，整次评测失败并保留证据；正常模型失败由原生verifier判0。指标为89题单次运行均分（success rate），不使用16条GRPO采样平均。

## 保存与评测时序

- `TRAINING_PHASE=smoke`：三步验证，默认 `NUM_ROLLOUT=3`、每步checkpoint；周期评测每5步，因此三步本身不触发全量TB2.1。可显式设 `EVAL_INTERVAL=1` 做更昂贵的逐步评测。
- `TRAINING_PHASE=long`（默认）：每5步checkpoint，每5步评测全89题。
- 默认跳过训练前额外评测；当前步训练、checkpoint保存及推理权重同步之后，异步模式只提交请求，评测由独立worker执行；显式同步模式才等待评测完成。
- 当前每rollout只更新一次，因此5个rollout等于5次optimizer更新；框架末次rollout还会额外保存checkpoint，不改变此安全机制。

## 验证

CPU测试覆盖固定数据身份、89题完整性、篡改/泄漏边界、异常与清理fail-closed、shell实际参数和TensorBoard事件。还需真实Harbor/E2B单题与真实模型推理冒烟；未执行的验收明确标注，不通过空配置或进程exit0宣称TB2.1已测通。不重启现有Ray或重复申请GPU。

## 2026-10-16：独立8卡后台评测

用户授权额外8张H100，总评测并发20。部署为单机4个TP2 SGLang实例，每实例最多5个trial；不加入训练Ray集群、不借用训练/采样的64卡。

训练周期hook只原子写入共享outbox：绑定绝对checkpoint目录`iter_<rollout_id>`、optimizer更新步`rollout_id+1`、固定89题配置/manifest摘要、模型配置与Agent预算。hook不运行Harbor、不转换模型、不等GPU。无新分数时只记录queued指标，不能把未完成评测记为0分。

单独worker按FIFO串行消费，持有进程锁。等待原生checkpoint tracker发布目标迭代（不把目录存在当保存完成），读取精确目标目录而非latest。训练不会覆写这些目录；outbox中的pin记录要求后续清理保留尚未评测的checkpoint。本次不实现或授权RL checkpoint删除。

worker仅用CPU把目标torch_dist的模型权重导出HF；不加载优化器、不量化、不覆盖来源或不完整输出。所有训练text权重必须来自checkpoint，只有未训练的vision权重可来自固定HF起点；对照起点逐tensor核验键/形状/dtype/有限值并保存SHA及来源回执，禁止漏权重回填基座。4实例在同一不可变HF目录上启动，89题全部结束、sandbox清理确认后才可切换下一checkpoint。失败保留产物、记failed状态，训练继续，不能将基础设施失败填0。进程中断后的running记录需显式核验/恢复，不能重跑覆盖。

评测结果独立写入训练TensorBoard的`tb21-background`子目录，以来源checkpoint更新步记横轴，同时记录完成时间与排队/转换/推理耗时。每5更新提交一次；SAVE_INTERVAL必须整除EVAL_INTERVAL，确保目标有checkpoint。异步存盘可由worker等待，不在训练线程等候。

当前三步恢复任务目标止于全局step4，不触发step5评测，也不为部署而中断它。后续启动使用异步hook；现有进程不会因编辑配置自动切换。

提速实验先保留当前基线：推理并发4→8；训练8K→16K打包；AllToAll→DeepEP/Flex。每项独立RUN，基于已验收checkpoint恢复，保持n16/global128/FP32/context与预算，比较同来源轨迹打包、显存、有效训练token吞吐和完整update耗时；失败回退，不把采样差异当同负载加速比。试验前确认无另一个已提交训练，不中断当前三步或重复申请训练GPU。
