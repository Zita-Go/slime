# NexTask Terminal-Agent RL

本目录把 NexTask RFC-0012 v3 动态题包接入 slime。训练时，模型通过 Codex CLI 在每题独立的 E2B 模板中操作真实终端；结束后由冻结的 NexTask runtime 创建候选快照，并使用题包自带的 shell-pairwise verifier 与参考快照比较。

## 安全边界

- 模型只进入 `dynamic_verification_manifest.json` 冻结的 `template_identity`，并在模板内的 `/app` 工作。
- `reference_solution/`、`verify.py`、`test.sh`、动态 receipts 和 package manifest 始终留在宿主侧。
- 数据 JSONL 中的 `package_dir` 与 `reference_dir` 是 rollout worker 的宿主侧 metadata；`generate.py` 不会把这些路径或内容写入模型沙箱。
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
  -> Codex CLI <-> slime OpenAI adapter <-> SGLang actor
  -> sanitize framework credentials
  -> immutable candidate snapshot
  -> NexTask v3 candidate/reference pairwise verifier
  -> determinate reward or dropped infrastructure sample
```

模型下载、Megatron checkpoint 转换、GPU smoke 和正式多机训练命令将在同目录的训练脚本中补齐。
