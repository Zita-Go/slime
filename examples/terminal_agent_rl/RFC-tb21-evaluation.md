# TB2.1 周期评测与 TensorBoard

## 目标与边界

- TB2.1 全部89题只用于评测，不从NexTask训练集划分验证集。
- 固定官方registry版本 `terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`。
- 使用Harbor 0.21.0原生Task/Trial/verifier与既有E2B环境适配，绝不调用NexTask pairwise verifier，也不把solution/tests加入模型prompt。
- 复用训练的TerminalToolAgent（bash工具、系统提示、上下文及预算），只把公开任务指令替换为TB2.1 instruction；这是TB2.1任务上的本项目harness分数，不冒充官方其他Agent的leaderboard结果。

## 接入方式

TensorBoard使用已有 `--use-tensorboard`，独立 `TENSORBOARD_DIR=$RUN_ROOT/tensorboard` 传入Ray runtime env。默认开启，缺依赖在提交前失败。

`--eval-function-path` 指向独立TB2.1评测函数，`--eval-config` 固定全量数据、每题1次和采样参数。评测函数仅调用当前已同步权重的OpenAIAdapter，不进入训练数据源/动态过滤/GRPO group。Harbor在独立Python3.13子进程执行，以隔离镜像Python3.12及冻结NexTask依赖。

准备工具从官方registry解析固定版本，将89个任务包缓存到共享存储，记录每个文件SHA及官方task digest，导出eval-only JSONL和manifest。启动与每次评测均验证完整89题集合、文件digest和指令，禁止静默过滤长prompt或只汇总成功启动的子集。

每个trial保留原生result、verifier输出和独立sandbox清理回执。基础设施异常不视为模型0分，整次评测失败并保留证据；正常模型失败由原生verifier判0。指标为89题单次运行均分（success rate），不使用16条GRPO采样平均。

## 保存与评测时序

- `TRAINING_PHASE=smoke`：三步验证，默认 `NUM_ROLLOUT=3`、每步checkpoint；周期评测每5步，因此三步本身不触发全量TB2.1。可显式设 `EVAL_INTERVAL=1` 做更昂贵的逐步评测。
- `TRAINING_PHASE=long`（默认）：每5步checkpoint，每5步评测全89题。
- 默认跳过训练前额外评测；评测在当前步训练、checkpoint保存及推理权重同步之后进行。
- 当前每rollout只更新一次，因此5个rollout等于5次optimizer更新；框架末次rollout还会额外保存checkpoint，不改变此安全机制。

## 验证

CPU测试覆盖固定数据身份、89题完整性、篡改/泄漏边界、异常与清理fail-closed、shell实际参数和TensorBoard事件。还需真实Harbor/E2B单题与真实模型推理冒烟；未执行的验收明确标注，不通过空配置或进程exit0宣称TB2.1已测通。不重启现有Ray或重复申请GPU。
