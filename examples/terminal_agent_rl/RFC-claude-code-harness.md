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
- CC计数：`/count_tokens`经relay到Adapter；追加历史优先复用正式usage，压缩/重写/分叉、首次及距离196608窗口不足32768时使用Nex tokenizer精确重算，不发额外模型探针。
- sandbox：每题冻结template identity；所有CC工具只在该sandbox内运行。

## 网络边界

CC运行在E2B sandbox内，必须通过AnthropicAdapter访问rollout策略，不能绕过Adapter直接请求NorthGate，否则会丢失on-policy token/logprob证据。真实网络探针已经确认以下E2B入站路径均不可达：

- 训练Pod的hostNetwork地址，直连与显式`http://10.51.6.27:5023`代理均超时；
- 当前CPU Pod的`10.233.*`地址，直连与同一代理也均超时。

因此正式默认使用`TERMINAL_RL_CC_TRANSPORT=e2b_pull`。每个sandbox内启动仅监听`127.0.0.1`的轻量Messages代理；rollout worker通过既有、已认证的E2B文件/命令控制通道长轮询请求，将其转发到worker本机的AnthropicAdapter，再把响应写回sandbox。Adapter仍只监听worker loopback，CC的session令牌、逐token logprob与top-p support链路保持不变。空闲轮询在sandbox内部每100ms检查、最多30秒才返回一次控制面RPC，避免按100ms从host高频请求E2B。

`TERMINAL_RL_CC_TRANSPORT=direct`只保留给已经单独证明E2B可达的公开地址，并强制要求`ADAPTER_PUBLIC_HOST`；5023继续只用于GPU Pod访问E2B API/文件/命令，不被误当作E2B访问集群私网的反向代理。

网络资格探针使用确定性脚本代替模型生成，只证明固定CC、真实E2B、pull relay、真实Adapter和工具回合的协议闭环；它不证明SGLang模型rollout、reward或GRPO参数更新。

## 验收门槛

1. 固定Node/CC包在真实NexTask模板中安装，`claude --version`精确为2.1.193，sandbox清理确认。
2. E2B pull relay到loopback AnthropicAdapter双向请求成功；CC真实Bash工具回合和第二轮`tool_result`通过，不使用训练通用HTTP代理污染E2B API链路。
3. 单题真实CC运行包含工具调用；如触发worker/auto-compaction，所有分支均能导出Sample，token/logprob/top-p offset/loss mask闭合。
4. public资产边界、候选snapshot、确定性reward与live sandbox清理继续通过。
5. 先跑单题、再跑8题×16的单步资格；验证显存、有限loss/非零梯度、权重同步和联合checkpoint后才允许长程。

## 回退

仅在显式`TERMINAL_RL_AGENT=terminal_tool`的诊断任务中使用旧轻量Harness；正式脚本默认`claude_code`，缺固定包或可达地址时启动前失败，不静默回退。
