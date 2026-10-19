# NexTask Terminal-Agent RL

本目录把 NexTask RFC-0012 v3 动态题包接入 slime。正式训练默认在每题独立E2B模板内运行固定Claude Code 2.1.193，CC通过AnthropicAdapter调用当前SGLang策略模型并使用其原生工具/子Agent/自动压缩；结束后由冻结的NexTask runtime创建候选快照，并使用题包自带的shell-pairwise verifier与参考快照比较。轻量TerminalToolAgent仅保留为显式诊断回退。

基础设施失败产生ABORTED结果时，fully-async会把整组放回buffer。一条原始采样可含多个训练段，因此恢复入口按Sample叶子处理会话和生成，只恢复未完成段；默认已完成段的token、mask、logprob和reward保持，partial模式仍沿用既有off-policy mask规则，外层n个独立采样槽不展平。Terminal重试必须新建session ID，保留旧会话的closed保护，避免迟到请求串入重试或收到HTTP503。修复源码不会热重载到已运行的RolloutManager；下一次正常启动生效，不为应用补丁中断当前训练。

## 安全边界

- 模型只能通过宿主侧 `bash` 工具进入 `dynamic_verification_manifest.json` 冻结的 `template_identity`；默认在模板内的 `/app` 工作。
- 冻结模板只提供运行时；`bridge.py materialize` 直接复用 NexTask Bench 的严格 public-export 逻辑，以最终 `package_manifest.json` 的 `solver_visible + public_contract/public_workspace + public_workspace_digest` 为权威闭包上传 `/app`。即使旧 workspace manifest 错把 `verify.py` 标为可见，最终 package 的 `private_audit` 判定也会将它排除。
- `reference_solution/`、`verify.py`、`test.sh`、动态 receipts 和 package manifest 始终留在宿主侧。
- 数据 JSONL 只保留 rollout worker 所需的宿主侧 `package_dir`，不序列化 `reference_dir`；`generate.py` 也不会把 package 路径或内容直接写入模型沙箱。
- reward 只接受确定性结果：`EQUAL = 1`，`NON_EQUAL = score_ratio`；`INDETERMINATE`、E2B 故障和 framework integrity 故障会丢弃该轨迹，而不是误记为 0 分。
- 每个候选先发布为短期 E2B snapshot，比较结束后立即删除。候选 snapshot 清理未确认时，该轨迹同样按基础设施故障丢弃。

## 为什么使用独立 bridge 进程

Slime 与 NexTask 有各自固定的 Python 依赖。`bridge.py` 由 NexTask 的 Python 环境运行，负责严格加载题包、准备沙箱、创建快照和打分；slime rollout worker 只通过 JSON 调用它。这样无需把 NexTask editable 安装到 slime 环境，也不会让某次依赖升级悄悄改变 verifier 语义。

当前固定 runtime：

```text
/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012-v3-runtime
commit 7f746b9832fcedbab4904fdd39d200aab3120ec7
```

GPU训练使用 `RL/vendor/nextask-runtime-py313/bin/python` 运行bridge。该Python 3.13虚拟环境链接到共享GPFS解释器，复用原NexTask虚拟环境的site-packages；不依赖CPU容器内的`/app/.venv`路径，也不向Slime的Python 3.12环境添加这些依赖。冻结runtime/NexAU源码和verifier身份保持原提交。CPU导出入口仍可使用原环境；`NEXTASK_RUNTIME_PYTHON`可显式覆盖训练配置。

Ray job入口使用Slime仓库的绝对`train.py`路径。driver默认继承Ray服务工作目录，提交客户端的`cd`不会改变driver工作目录；源码通过runtime env中的`PYTHONPATH`从共享仓库加载。

训练rollout的单题Agent做题预算默认是 `TERMINAL_RL_AGENT_TIME_BUDGET_SEC=14400`（4小时），单轮最大输出32768 token。CC最大thinking为64000，并在196608 token触发原生auto-compaction；外层rollout guard默认再加workspace、判题和清理余量，因此不会在Agent预算耗尽前截断。CC训练禁用`Task`/`Agent`子Agent调用，不注册worker；保留单Agent的Bash/文件工具及自动压缩。32K输出会提高生成量和训练显存峰值，长程使用前必须通过独立单步资格测试。固定Node/CC包和网络门槛见[RFC](RFC-claude-code-harness.md)。

`TRAINING_PHASE=smoke` 默认三步、每步保存checkpoint；`TRAINING_PHASE=long`（默认）每5步存盘、每5步评测完整TB2.1。当前每个rollout只做一次优化，因此这里的step是参数更新而非microbatch。框架在末次rollout/epoch边界还可能额外保存恢复点；保存成功仍需核验模型、optimizer和数据进度元数据，目录存在本身不代表可恢复。

指定`LOAD_PATH`完整恢复时，入口同时传入Megatron原生`--use-checkpoint-opt-param-scheduler`，保留checkpoint的调度器配置和已消耗样本进度，不使用`finetune`/跳过optimizer或RNG加载。这样原1步任务的scheduler长度128不会与新4步任务的512触发冲突。当前constant学习率1e-6在超过原长度后仍保持1e-6，不会变成0；不重置Adam动量、step或修改原checkpoint。`NUM_ROLLOUT`为总终点：原rollout0恢复后新增3次更新应设为4，而非3。新的输出RUN必须与原checkpoint及失败RUN分离。

## TensorBoard与TB2.1测试集评测

默认开启TensorBoard（`USE_TENSORBOARD=1`），事件写入`$RUN_ROOT/tensorboard`，不依赖外网/W&B账号。`TENSORBOARD_DIR`会显式传入Ray worker，不是只设置在提交shell。可在能读取该共享目录的机器上查看：

```bash
tensorboard --logdir /绝对路径/RUN/tensorboard --host 127.0.0.1 --port 6006
```

默认评测配置为[tb21_eval.yaml](tb21_eval.yaml)：固定TB2.1全部89题、每题1次，直接使用其测试集，不从训练题库另外拆分验证集。指标`eval/tb21`是原生二元reward的89题均分；`train/loss`、`train/grad_norm`、rollout reward等由已有Slime日志接口记录。反复据测试集调参后的结果属于内部开发评测，不能当成完全未接触测试集的最终泛化成绩。

TB2.1使用Harbor 0.21.0的Task/Trial/原生verifier和独立E2B环境，不调用NexTask pairwise verifier。当前评测入口仍是`slime-terminal-bash`，尚未随训练迁移到CC；两种Harness的结果必须分开标注。tests/solution仅留在host task包中，由Harbor在正确阶段处理；模型prompt只有公开instruction。

TB2.1旧Harness的参数已对齐step90可移植数值，但这不等于CC Harness对齐；后续须复用同一ClaudeCodeHarness/AnthropicAdapter并独立验收后再替换评测入口。

默认`TB21_EVAL_MODE=async`：每5更新仅提交固定checkpoint请求，不等待89题完成。`TB21_EVAL_QUEUE`默认指向共享盘`RL/tb21-eval-queue`。额外单机8H100运行4个TP2实例，每实例5题，总并发20；worker不加入训练Ray，不连接训练推理服务。独立Pod使用与训练相同的保护E2B配置及专用代理、Harbor Python：

```bash
EVAL_EXPECTED_POD=<已核验的独立评测Pod全名> bash examples/terminal_agent_rl/run_tb21_worker.sh
```

worker须通过Pod本地monitor托管，启动前核验精确Job/8GPU/数据与E2B链路。等待队列不等于推理服务已部署：真实导出/加载/89题结果分别保存在`jobs/*/evaluation/{hf-snapshot,sglang-*.log,inference-ready.json,trials}`。仅`status=completed`且89题/清理全部确认才有分数；失败不是0分，保留日志并且不终止训练。worker中断后的running任务必须人工核验，不覆盖重跑。每步结果按来源optimizer step写入`$TENSORBOARD_DIR/tb21-background`，不按完成时的最新训练step归属。

TB2.1沿用`top_p=0.95`时，评测session会显式设置`custom_params.return_top_p_token_ids=true`，要求SGLang返回逐token候选ID和offset。缺失时Adapter继续fail-closed；不能改成`top_p=1`规避校验，也不能事后补造support。

checkpoint请求是pin，所有尚未完成的评测来源都不能被清理。本实现不删除任何RL checkpoint或HF快照；容量不足需处理后再继续。`SAVE_INTERVAL`须整除`EVAL_INTERVAL`。如显式选择`TB21_EVAL_MODE=sync`，才沿用原来等待当前同步权重评测的方式。当前已启动的三步进程不会自动更换hook；其全局终点step4本来不触发每5步评测。

提速参数保持分开测试：`SGLANG_SERVER_CONCURRENCY=8`（原4）、`MAX_TOKENS_PER_GPU=16384`（原8192）、`MOE_DISPATCHER=deepep`（原alltoall，对应flex/backend deepep）。默认值不变；必须先验收当前恢复任务与有效checkpoint、无其他活跃训练，再以独立RUN逐项试验，不修改n16/global128/FP32/context/Agent预算。比较完整更新耗时及显存，不用采样等待期的低显存推断反向安全。

`run_performance_trial.sh rollout8|pack16k|deepep`强制每候选只从同一验收来源新增一次更新、save1、独立RUN，诊断试验不触发TB2.1以免污染耗时对比；正式长程仍save5/eval5。必须显式传入`LOAD_PATH/PROMPT_DATA/RUN_ROOT/PERF_ACCEPTANCE_RECEIPT`，后者必须包含`ok=true`、`additional_optimizer_updates_verified>=3`、与LOAD_PATH相同的`checkpoint_root`，不能用排队/单个目录存在代替验收。脚本拒绝活跃Ray提交、空闲GPU不足64或存储不足，不申请卡、不重启Ray。每项仅改变自身参数，结果未验收前不自动设为长程默认。

准备固定版本测试集（当前共享快照已准备，无需重复运行）：

```bash
PYTHONPATH=$PWD /gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau/.venv/bin/python \
  -m examples.terminal_agent_rl.tb21_dataset \
  --output /gpfs/shared_data/zhangguoqiang/terminal-code/RL/datasets/tb21-7d7bdc1c
```

准备器只接受官方固定registry版本、完整89个task identity；输出已存在时拒绝覆盖。每次启动和评测重验manifest、task文件SHA与公开指令，不能静默截去长任务或漏题。运行时通过`TB21_RUNTIME_PYTHON`选择独立Harbor Python3.13，不污染Slime Python3.12/训练vendored SDK；E2B只使用既有保护配置和专用代理，密钥不进入Ray配置。默认`TB21_CONCURRENCY=8`。

三步验证用`TRAINING_PHASE=smoke`，默认`SAVE_INTERVAL=1`；长程用`TRAINING_PHASE=long`，默认`SAVE_INTERVAL=5 EVAL_INTERVAL=5`。默认跳过训练前额外评测，周期评测在更新后的权重同步后进行。三步阶段不触发每5步的全量测试；可显式设`EVAL_INTERVAL=1`，但每步会额外完整跑89题。基础设施错误/缺失reward/沙箱清理失败会让该次评测报错，不冒充模型0分或缩小均分分母。每题原生result/verifier及清理回执在`$RUN_ROOT/tb21-eval/rollout-XXXXXX/`保留，失败轮次不覆盖。

没有GPU模型的真实环境/verifier冒烟入口是`python -m examples.terminal_agent_rl.smoke_tb21 --prompt-data <tb21.jsonl> --output <新目录>`。它用固定空回答，只验链路，不能当成模型分数或全89题通过。`ENABLE_TB21_EVAL=0`和`USE_TENSORBOARD=0`仅供明确的隔离诊断，默认不关闭。设计与验收边界见[RFC](RFC-tb21-evaluation.md)。既有TP8资源覆盖仍应显式传入，不能把旧96卡脚本默认拓扑直接用于当前64卡集群。

## 导出训练集

```bash
bash examples/terminal_agent_rl/export_dataset.sh
```

默认读取：

```text
/gpfs/shared_data/zhangguoqiang/terminal-code/NexTask/nexbench2408-rfc0012-nostage-verified
```

并原子写入该目录下的 `slime_train.jsonl`。导出过程会使用 v3 package loader 完整校验每个题包和参考快照身份，任何 schema、digest、template 或 workspace 不一致都会整体失败，不会静默跳题。

训练入口不固定题目数量。`validate_dataset.py` 校验非空原始题面、v3 schema/verifier协议、metadata身份/digest和重复题；当JSONL旁有`collection.json`时，还要求题目集合与manifest身份一致，避免collection扩容后仍训练旧子集。新增题目应先通过严格public闭包/verifier审计和E2B模板可用性检查，再重新导出；验证过程不修改题面或参考材料。

## 运行链路

```text
JSONL metadata
  -> frozen E2B template (/app only)
  -> host-side bash tool loop <-> slime OpenAI adapter <-> SGLang actor
  -> sanitize framework credentials
  -> immutable candidate snapshot
  -> NexTask v3 candidate/reference pairwise verifier
  -> determinate reward or dropped infrastructure sample
```

E2B 凭据不写入训练脚本或 Ray runtime JSON。训练任务只传递 `TERMINAL_RL_E2B_CONFIG_FILE` 路径；该文件必须是权限不宽于 `0600` 的普通 JSON 文件：

```json
{
  "api_key": "<secret>",
  "api_url": "https://...",
  "domain": "..."
}
```

内部 E2B 网关当前固定使用共享目录中的 `e2b 2.21.0`（wheel SHA-256 `360a55ff23335dd62ee5ca86c41dc55de597d73b846f5c2361f6eac6c712a371`）；训练脚本把该目录放在 Ray worker 的 `PYTHONPATH` 最前面，避免容器内较新的 SDK 调用不兼容接口。

GPU 冒烟和训练入口默认设置 `TERMINAL_RL_E2B_PROXY=http://10.51.6.27:5023`，并传入 Ray worker。Slime SDK 使用显式 `proxy=`；独立 NexTask bridge 在进程入口适配 SDK 配置默认值，覆盖文件、命令、快照创建/恢复/删除与 verifier sandbox。固定 NexTask/SDK 源码不变，设计见 [E2B代理方案](RFC-e2b-proxy.md)。

直接在当前机器调用 Python 入口时默认直连；bridge 会清除继承的通用代理。若需要 GPU 入口直连，显式设置 `TERMINAL_RL_E2B_PROXY=''`。该变量只控制 E2B，不代理模型推理/Ray通信。

CC回调模型的方向与上述E2B控制面相反。真实探针已确认E2B无法直连训练Pod的hostNetwork地址或当前CPU Pod地址，5023也不能把这些私网地址暴露给E2B。因此训练默认`TERMINAL_RL_CC_TRANSPORT=e2b_pull`：sandbox中的CC只请求`127.0.0.1`代理，rollout worker经既有E2B控制通道拉取请求并转发给本机loopback AnthropicAdapter。该链路已用固定CC 2.1.193、真实E2B和真实Adapter完成两轮Bash工具调用、Sample/有限logprob结构与sandbox清理验收；上游生成在网络资格测试中是确定性脚本，不能替代真实模型rollout或GRPO验收。`direct`模式仅供已有可达公开地址时显式使用，并要求`ADAPTER_PUBLIC_HOST`；完整设计与验收边界见[RFC](RFC-claude-code-harness.md)。

CC的`/v1/messages/count_tokens`请求同样经pull relay转发，不能在sandbox代理中伪造0。AnthropicAdapter默认对首轮、历史压缩/重写/分叉及临近阈值的请求使用与正式生成完全相同的Nex tokenizer/chat template精确计数；普通追加轮次复用上一轮正式响应的真实`input_tokens/output_tokens`，只对新增工具返回/消息和格式开销做保守估算。训练默认在`196608-32768=163840`以上强制精确计数；`TERMINAL_RL_CC_AUTO_COMPACT_WINDOW`与`TERMINAL_RL_CC_COUNT_EXACT_MARGIN`可显式调整，并同步传入Ray。该方案不额外调用模型，SGLang radix cache也不改变逻辑prompt token数。

只验证 E2B public workspace 与 verifier 闭环、不加载模型时，可运行：

```bash
PYTHONPATH=/gpfs/users/i-zhangguoqiang/terminal-code/RL/vendor/e2b-2.21.0:$PWD \
TERMINAL_RL_E2B_CONFIG_FILE=/gpfs/users/i-zhangguoqiang/terminal-code/RL/.secrets/nextask-e2b.json \
python -m examples.terminal_agent_rl.smoke_e2b \
  --package /path/to/final/task_xxx \
  --template-identity '<template_identity>'
```

## E2B 代理验收记录

2026-10-09 15:04 CST，无模型的真实 public-workspace/verifier 冒烟通过；代理为 `http://10.51.6.27:5023`，使用代理接入提交 `58164d39`。

- 第一题公开资产89项，digest为 `9f5eb5cb54506ef527acca016c9f54e20ad6c2d689695a03e928cdded406aa92`，与冻结 Final manifest 的严格公开闭包一致；Agent workspace 未包含reference/verifier/私有测试。
- 未运行模型求解的初始候选得到确定性 `NON_EQUAL`，score=0/100、ratio=0.0；这是有效判分结果。
- 候选snapshot和live sandbox清理均confirmed；本地候选临时目录为空。
- 真实回执：`/gpfs/users/i-zhangguoqiang/terminal-code/RL/RUN/e2b-proxy-verifier-1009-065754/receipt.json`；monitor=`20261009T065831Z-13a45f10`，exit_code=0、completed。

这次验收证明代理覆盖准备工作区、snapshot和verifier链路，不代表模型Agent轨迹、Slime loss/logprob、Megatron参数更新或64卡正式训练已通过。完整GPU rollout/训练验收仍需单独执行。

## 模型转换

模型下载完成后，申请一台至少 2×H100 的开发 Pod，在 Pod 内执行：

```bash
bash examples/terminal_agent_rl/convert_nex_n25_mini.sh
```

脚本会生成 `${HF_CHECKPOINT}_torch_dist`，已存在完整 checkpoint 时幂等退出；发现不完整目录时会拒绝覆盖。

## GPU 冒烟

在专用至少 2×H100 Pod 内依次验证 SGLang TP=2 和一条真实 Slime/E2B rollout：

```bash
JOB_NAME=<NorthJob JobAlias> bash examples/terminal_agent_rl/run_gpu_smokes.sh
```

第一阶段要求普通对话和强制 `bash` tool call 都通过 OpenAI-compatible API；第二阶段使用 `--debug-rollout-only`，并检查落盘样本的 token、loss mask、rollout logprob、reward 和确定性 verifier verdict。总回执写入 `RUN/<job>/gpu-smokes-receipt.json`。

Nex-N2.5-mini 的原生上下文窗口为 262,144 tokens（256K）。GPU smoke 和正式训练默认都使用该值；初始题面最长仅 1,704 tokens，Slime smoke 与正式训练的 prompt 过滤上限均为 65,536（64K）。正式训练默认使用64张actor卡，TP=2、CP=32、DP=1、PP=1、EP=32、Expert TP=1；另有32张rollout卡，总共96卡。训练侧动态打包上限由 `MAX_TOKENS_PER_GPU` 控制，默认8,192。

EP=32将每卡专家数从16降至8，保留FP32梯度累积/归约和原有采样预算。相对上一轮EP=16，专家BF16权重及FP32梯度理论每卡减少约5.625 GiB。CP=32进一步分摊长序列；真实旧dump重放时，配合8K打包目标，最大局部token从14,080降到8,448。两份FP32 logits加bool mask的对应张量体积约减少5.86 GiB；该估算不能与专家存储节省直接相加作为实测峰值下降。完整ref、反向、梯度归约和参数更新仍需GPU验收；`CP_SIZE`、`EP_SIZE`均可显式覆盖。

训练端MoE默认使用Megatron原生 `alltoall` dispatcher。EP16配合DeepEP/Flex时实测进程总显存与PyTorch分配显存相差约十几GiB，12K microbatch仍在反向阶段OOM；切换 `alltoall` 用于减少持久通信缓冲，可能牺牲MoE通信吞吐。该变化不修改模型精度、专家路由或训练样本预算。

当前模型有2个KV groups；旧固定镜像的Megatron output gate路径在TP大于2时没有与Query同步切分gate，TP=4历史实测在首个ref forward出现形状不匹配。当前stable/latest补丁已回移NVIDIA #3529修复，TP2/TP4/TP8的CPU切片及梯度回归通过；已有不可变镜像须显式应用gate补丁，再执行[GPU诊断](../../docker/README.md)，完整GPU模型训练仍需验证。当前排队96卡任务继续保持TP=2/CP=32/EP=32，不因加入修复而自动切换拓扑。

## 多机训练

本训练入口的默认拓扑为12×8 H100（96卡），其中8节点训练actor、4节点运行16个TP=2 rollout engine。通用NorthJob启动器仍默认8节点；本模型须显式传入 `NNODES=12 ACTOR_NODES=8 ROLLOUT_NODES=4`，不能直接沿用旧64卡申请：

启动器分别校验attention分组的 `TP×PP×CP` 与expert分组的 `ETP×EP×PP` 能否整除actor卡数，遵循Megatron parallel folding，不能将attention TP直接乘到EP上。TP8验证时必须重新选择CP/PP，不能叠加当前CP32。单个gate诊断通过之后仍需全模型前向、反向、梯度归约及显存验收；当前线性注意力适配会聚合序列并重复计算，不能假设所有层都按TP8分摊。只有完整TP8验证通过后才逐项回退OOM缓解配置。

正式训练默认 `N_SAMPLES_PER_PROMPT=16`，`ROLLOUT_BATCH_SIZE=8`，每批更新一次，`GLOBAL_BATCH_SIZE=128`。这是首轮实验起点；若 E2B 吞吐不足，可设 `ROLLOUT_BATCH_SIZE=4`，global batch 自动变为64。`micro-batch-size=1`，训练侧按每CP rank 8K token动态打包。降低 `MAX_TOKENS_PER_GPU` 不截断轨迹或改变global batch，但会增加microbatch数；超过上限的单条长轨迹仍会独占一个microbatch，padding也可能使实际局部长度略高于目标。

`LOG_PROBS_CHUNK_SIZE` 默认1024，控制词表 logits 生成后的 logprob 计算分块。若这一步 CUDA OOM，可用 `LOG_PROBS_CHUNK_SIZE=128` 验证更低的临时显存峰值；它不缩小完整 logits 张量，不改变上下文长度、采样参数或 GRPO 组大小。

`SGLANG_SERVER_CONCURRENCY=4` 与16个engine对应64条在途轨迹、约4个同题组并发；128条轨迹不要求同时启动。启动sandbox的并发仍限制为16。单轨迹GPU smoke继续使用n=1且不更新模型。

```bash
NNODES=12 ACTOR_NODES=8 ROLLOUT_NODES=4 \
MODE=train \
TRAIN_SCRIPT=/gpfs/users/i-zhangguoqiang/terminal-code/RL/slime/examples/terminal_agent_rl/run_nex_n25_mini.sh \
DRY_RUN=false \
bash scripts/northjob/launch_slime_multinode.sh
```

NorthJob bootstrap 负责启动所有 Ray 节点；训练脚本只向现有 Ray dashboard 提交任务，不会重复 `ray start`。默认使用 fully-async rollout、GRPO、官方推荐的 `temperature=0.7/top_p=0.95/top_k=40`，并为 SGLang 设置 `qwen3` reasoning parser、`qwen3_coder` tool parser 和 `mamba-scheduler-strategy=extra_buffer`。
