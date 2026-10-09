# NexTask Terminal-Agent RL

本目录把 NexTask RFC-0012 v3 动态题包接入 slime。训练时，rollout worker 在本地运行多轮 agent loop，模型通过 `bash` 工具操作每题独立的 E2B 模板；结束后由冻结的 NexTask runtime 创建候选快照，并使用题包自带的 shell-pairwise verifier 与参考快照比较。

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

## 导出训练集

```bash
bash examples/terminal_agent_rl/export_dataset.sh
```

默认读取：

```text
/gpfs/shared_data/zhangguoqiang/terminal-code/NexTask/nexbench2408-rfc0012-nostage-verified
```

并原子写入该目录下的 `slime_train.jsonl`。导出过程会使用 v3 package loader 完整校验每个题包和参考快照身份，任何 schema、digest、template 或 workspace 不一致都会整体失败，不会静默跳题。

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

只验证 E2B public workspace 与 verifier 闭环、不加载模型时，可运行：

```bash
PYTHONPATH=/gpfs/users/i-zhangguoqiang/terminal-code/RL/vendor/e2b-2.21.0:$PWD \
TERMINAL_RL_E2B_CONFIG_FILE=/gpfs/users/i-zhangguoqiang/terminal-code/RL/.secrets/nextask-e2b.json \
python -m examples.terminal_agent_rl.smoke_e2b \
  --package /path/to/final/task_xxx \
  --template-identity '<template_identity>'
```

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

Nex-N2.5-mini 的原生上下文窗口为 262,144 tokens（256K）。GPU smoke 和正式训练默认都使用该值；初始题面最长仅 1,704 tokens，Slime smoke 与正式训练的 prompt 过滤上限均为 65,536（64K）。64卡训练默认使用 CP=8，使每个 CP rank 的 token 上限保持在约32K。

## 多机训练

默认拓扑为 4×8 H100，其中 2 节点训练 actor、2 节点运行 8 个 TP=2 rollout engine：

```bash
MODE=train \
TRAIN_SCRIPT=/gpfs/users/i-zhangguoqiang/terminal-code/RL/slime/examples/terminal_agent_rl/run_nex_n25_mini.sh \
DRY_RUN=false \
bash scripts/northjob/launch_slime_multinode.sh
```

NorthJob bootstrap 负责启动所有 Ray 节点；训练脚本只向现有 Ray dashboard 提交任务，不会重复 `ray start`。默认使用 fully-async rollout、GRPO、官方推荐的 `temperature=0.7/top_p=0.95/top_k=40`，并为 SGLang 设置 `qwen3` reasoning parser、`qwen3_coder` tool parser 和 `mamba-scheduler-strategy=extra_buffer`。
