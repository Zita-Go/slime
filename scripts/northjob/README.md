# NorthJob GPU 开发 Pod

这里的脚本参考 `Distill/training/scripts/launch_qwen3_5_35b_a3b_taw_sft_northjob.sh`，用于申请单节点、可交互的 slime GPU 开发环境。

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
