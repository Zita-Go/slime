# Terminal-Agent RL迁移到Claude Code Harness

## 目标

正式训练使用真实Claude Code CLI，而不是仅让模型模仿CC消息风格。NexTask v3题包、public workspace导出、E2B template identity、冻结verifier和GRPO奖励保持不变。

## 复用边界

- 复用`slime.agent.harness.ClaudeCodeHarness`安装并运行固定Claude Code 2.1.193。
- 复用`slime.agent.adapters.AnthropicAdapter`把CC的Messages请求转到当前SGLang策略模型，并保留逐token logprob/top-p support/loss mask。
- 复用`TrajectoryManager`处理CC子Agent与自动压缩产生的prompt分叉；不自行删除历史消息或伪造摘要。
- 继续由`examples.terminal_agent_rl.runtime`物化公开workspace、创建候选snapshot、调用严格pairwise verifier并确认清理。
- 轻量`TerminalToolAgent`只保留为显式诊断回退，不再是正式训练默认。

## 固定运行参数

- Claude Code版本：2.1.193；Node版本：20.19.5，均使用共享盘固定包。
- Agent墙钟：14400秒；上下文262144；自动压缩阈值196608。
- 采样：temperature1.0/top_p0.95/top_k40；单次生成上限32768。
- CC设置：bypass permissions、关闭WebFetch/WebSearch与slash commands，提供`worker`子Agent，最大thinking 64000。
- sandbox：每题冻结template identity；所有CC工具只在该sandbox内运行。

## 网络边界

CC运行在E2B sandbox内，必须回连rollout worker上的AnthropicAdapter。普通Kubernetes overlay Pod `10.233.*`的真实回连探针已超时；正式资格测试只能使用训练Pod的hostNetwork地址，并先做单请求HTTP/Anthropic探针。不可达时禁止训练，不能绕过Adapter直接请求NorthGate，否则会丢失on-policy token/logprob证据。

## 验收门槛

1. 固定Node/CC包在真实NexTask模板中安装，`claude --version`精确为2.1.193，sandbox清理确认。
2. E2B到hostNetwork Adapter双向请求成功，不使用训练通用HTTP代理污染E2B API链路。
3. 单题真实CC运行包含工具调用；如触发worker/auto-compaction，所有分支均能导出Sample，token/logprob/top-p offset/loss mask闭合。
4. public资产边界、候选snapshot、确定性reward与live sandbox清理继续通过。
5. 先跑单题、再跑8题×16的单步资格；验证显存、有限loss/非零梯度、权重同步和联合checkpoint后才允许长程。

## 回退

仅在显式`TERMINAL_RL_AGENT=terminal_tool`的诊断任务中使用旧轻量Harness；正式脚本默认`claude_code`，缺固定包或可达地址时启动前失败，不静默回退。
