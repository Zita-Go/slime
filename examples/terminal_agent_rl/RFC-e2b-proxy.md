# E2B 代理接入方案

## 问题与目标

GPU Pod 无法直连 E2B。代理 `http://10.51.6.27:5023` 已在 GPU Pod 中通过创建、文件、命令与清理验证。
RL 还需要覆盖 NexTask bridge 的同步 SDK、快照创建/恢复/删除与 verifier 沙箱操作。

## 配置与范围

- 使用 `TERMINAL_RL_E2B_PROXY` 作为唯一的 E2B 网络配置；空值表示直连。
- 仅 GPU 冒烟和训练入口默认设置为上述代理，Ray runtime 传递该变量。
- 当前机器直接调用 Python 入口时默认直连，不继承 `HTTP_PROXY` 等通用代理。
- Slime 创建 sandbox 时显式传入 SDK 的 `proxy=`，后续操作沿用实例连接配置。
- bridge 子进程清除继承的通用代理，并按 E2B 配置设置自己的网络环境。
- NexTask 固定源码不提供 proxy 参数，因此只在独立 bridge 进程中适配 SDK `ConnectionConfig` 的默认 proxy，覆盖它内部新建的所有客户端；不修改冻结 runtime 或 SDK 源码。
- 若调用方显式指定不同代理，拒绝混用路由；SDK API 与隔离/判分语义保持一致。
- 导出训练集不涉及网络，不安装 SDK 适配。

## 验证

检查直连时的环境清理、Ray 配置传递、Slime 创建参数、bridge 子进程默认连接与显式连接。
覆盖同步/异步 SDK API 客户端与 sandbox transport 的代理路由，并保留既有终端 Agent 测试。
真实 public-workspace/verifier 冒烟用于核验快照、确定性判分和清理；与完整模型 RL 验收分别记录。
