# NorthJob GPU 开发 Pod

这里的脚本参考 `Distill/training/scripts/launch_qwen3_5_35b_a3b_taw_sft_northjob.sh`，提供单节点开发 Pod 和多节点 Ray/slime 集群两种入口。

## 多节点训练集群

默认以 dry-run 渲染一个 8×8 H100（64卡）的 Ray 集群清单，不调用提交 API，其中 4 个节点用于 actor、4 个节点用于 rollout：

```bash
bash scripts/northjob/launch_slime_multinode.sh
```

确认后创建交互式多节点集群：

```bash
DRY_RUN=false bash scripts/northjob/launch_slime_multinode.sh
```

看到共享目录 `RUN/<job-name>/cluster.ready` 后，使用同一个进入脚本进入 master。此时所有节点已经加入同一个 Ray 集群，不需要逐台登录：

```bash
bash scripts/northjob/enter_slime_dev_pod.sh <job-name>
ray status
```

停止交互式资源池：

```bash
touch ../RUN/<job-name>/cluster.stop
```

正式训练模式由 master 自动执行指定脚本；该脚本应连接已经存在的 Ray 集群，不应再次执行 `ray start`：

```bash
MODE=train TRAIN_SCRIPT=/gpfs/.../train_terminal_agent.sh DRY_RUN=false \
  bash scripts/northjob/launch_slime_multinode.sh
```

资源拓扑可以覆盖，例如回退到32卡时设置 `NNODES=4 ACTOR_NODES=2 ROLLOUT_NODES=2 GPU_PER_NODE=8`。启动器要求 `ACTOR_NODES + ROLLOUT_NODES == NNODES`。

多节点提交通过仓库内 `submit_slime_northjob.py` 在 NorthJob 上传前加入 **必选 Pod 反亲和**：同一 Job 的训练 Pod 必须分布在不同的 `kubernetes.io/hostname` 上。仅设置 `nnodes=2` 不能保证两台物理机器。选择器只匹配本 Job，既有节点亲和配置保留；不修改共享 SDK。dry-run 会在 `RUN/<job-name>/` 写入真实 JSON/YAML，断开上传入口，正式提交前再次验证落盘清单。

固定Slime镜像没有NorthJob runtime。多机入口通过 `--user-env-path` 在调用northlaunch之前source `prepare_northlaunch.sh`，只链接GPFS中固定版本的northlaunch，不激活共享环境或替换训练Python。可用 `NORTHLAUNCH_BIN` 覆盖共享入口；准备前后必须保持训练Python一致。设计见 [启动入口方案](RFC-launcher-entry.md)。

共享目录从入口准备阶段即保存 `launcher-<rank>.log/.exit_code`；bootstrap在检查ROLE等变量之前保存 `node-<rank>-<role>.log/.exit_code`，因此入口失败也有证据。

长任务watch应检查启动结果，而不是只等ready文件：

```bash
python3 scripts/northjob/check_slime_startup.py --job-name <JobAlias> \
  --nodes 8 --gpus-per-node 8 --run-dir /gpfs/.../RL/RUN/<JobAlias>
```

该检查产出 `startup-status.json`；`waiting` 返回1，取得 `ready` 或 `failed` 结果后返回0。watch完成只表示结果已生成，后续必须检查JSON的status：只有ready才继续模型转换/训练；failed时先读错误和日志。Job终止、主容器退出（即使sidecar仍Running）或Job消失都会产生失败诊断，旧ready文件不能覆盖真实失败。

结果包含每个 Pod 的 `physical_host` 与 `physical_host_count`；GPU 验收还要求期望数量的不同物理主机，两个同机 Ray 节点不能作为跨机器通过。CPU 入口诊断不强制物理分散，但仍报告宿主机数。

无GPU `train/raw` 入口诊断可增加 `--result-file northlaunch-smoke-rank0.json`；输出的result_kind为entrypoint-only，只验收SDK角色环境和镜像Python，不能用于启动GPU训练。正常集群检查的result_kind为ray-cluster。

两个零GPU节点的Ray启动验收可设置 `CPU_STARTUP_SMOKE=true GPU_PER_NODE=0 NNODES=2 ACTOR_NODES=1 ROLLOUT_NODES=1 MEM_PER_POD=8 RAY_OBJECT_STORE_MEMORY_BYTES=536870912 MODE=interactive`。CPU模式必须显式开启且不能运行训练，默认仍为64卡配置；检查时使用 `--nodes 2 --gpus-per-node 0`，核验Alive=2/GPU=0。验收后用cluster.stop或平台删除命令释放诊断任务。

## 单节点开发 Pod

默认只预览命令，不会提交：

```bash
bash scripts/northjob/launch_slime_dev_pod.sh
```

申请一台 8×H100 机器：

```bash
DRY_RUN=false bash scripts/northjob/launch_slime_dev_pod.sh
```

资源可以通过环境变量覆盖，例如申请 4×H100：

```bash
JOB_NAME=slime-dev-4gpu GPU_PER_NODE=4 MEM_PER_POD=320 DRY_RUN=false \
  bash scripts/northjob/launch_slime_dev_pod.sh
```

任务调度成功并进入 `Running` 后，使用提交时显示的 JobAlias 进入：

```bash
bash scripts/northjob/enter_slime_dev_pod.sh <job-name>
```

只解析并打印实际的 `kubectl exec` 命令、不进入容器：

```bash
DRY_RUN=true bash scripts/northjob/enter_slime_dev_pod.sh <job-name>
```

Pod 使用固定 digest 的集群内部 slime 镜像。启动时会执行 `pip install -e . --no-deps`，因此 `/gpfs/users/i-zhangguoqiang/terminal-code/RL/slime` 中的源码修改会立即对新启动的 Python 进程生效。镜像内已有完整 CUDA、PyTorch、SGLang 和 Megatron 环境，不依赖当前 CPU 容器下的 `/root/micromamba`。

注意：`--dev-pod` 适合单机交互调试，会关闭 RDMA 和 host network。正式的多节点 Ray/slime 训练应使用单独的 `--job-type train --train-launcher raw` 提交脚本。
