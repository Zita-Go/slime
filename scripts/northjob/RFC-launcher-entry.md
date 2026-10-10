# 固定 Slime 镜像的 NorthJob 启动入口修复

## 已确认问题

同镜像复现显示 northlaunch 不存在，northmirror 未安装，执行退出码127。
NorthJob train/raw 会在 bootstrap 之前调用 northlaunch，因此不能在 bootstrap 内补装。

## 方案

- 使用平台支持的 --user-env-path，在 northlaunch 调用之前 source 准备脚本。
- 只在 /root/.local/bin 放置指向 GPFS NorthJob 3.3.11 northlaunch 的链接；不激活共享环境，不把其整个 bin 放入 PATH。
- NorthJob 使用入口 shebang 的共享解释器，实际用户命令仍使用 Slime 镜像 Python。
- 准备前后校验训练 Python 位置，拒绝覆盖其他入口，验证 northlaunch --help。
- 调用启动器之前将 stdout/stderr 与退出码写入 launcher 日志；bootstrap 在必需环境检查前开启节点日志。
- 无GPU raw诊断验证入口、rank/master/role与镜像Python；随后进行小规模Ray验证，最后重跑训练。
- 启动结果检查同时读取真实Job状态、各Pod主容器退出状态和cluster.ready；失败时输出失败诊断，而不继续盲等。
- watch的完成目标是“取得启动结果”，就绪和失败都属于可核验结果；只有结果status=ready才允许后续转换/训练。

## 验收

CPU回归验证PATH隔离、入口冲突/缺失、早期错误日志与启动结果分类。
固定镜像的零GPU train/raw任务必须证明northlaunch可执行、角色环境齐全且用户命令仍用镜像Python。
小规模Ray验证必须核验Alive节点数和GPU总量。零GPU通过不代表多机GPU训练已通过。
