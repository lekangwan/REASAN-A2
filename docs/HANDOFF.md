# 给接手同学的使用说明

先读根目录 README 的实验含义与安装步骤。当前交付是官方 Go2 权重复现，以及 A2 MoE 的零样本迁移流程；训练和雷达估计器后置。

## 1. 获取完整代码

仓库为私有，需要仓库所有者授予访问权限。使用 `git clone --recurse-submodules`，不要只下载 GitHub 的源码 ZIP，因为它不包含子模块内容。已经普通 clone 的可补运行：

```bash
git submodule update --init --recursive
git -C upstream/REASAN rev-parse HEAD
```

最后一条应输出 `655bf4f1cc23e835d87bf9a887972af6760af162`。不要自动更新到上游最新提交，REASAN 随仓库修改过 Isaac Lab 和 RSL-RL。

## 2. 恢复模型与地图

```bash
python3 scripts/download_release.py
python3 scripts/restore_release.py
python3 scripts/fetch_isaac_assets.py
```

默认只恢复运行所需的 A2 资产、原始及导出权重、原生参考和冻结地图。官方 Go2 与导航、安全权重来自上游子模块。下载脚本复用现有 GitHub 认证，不写令牌；若使用 SSH clone 且没有 HTTPS 认证，可用 `gh auth login`，或从 Release 页面手动下载资产附件到 `release-downloads/`。手动下载同样执行 `restore_release.py`，会核对固定摘要。

七个 NVIDIA 资产从官方地址获取。若网络受限，可从已有环境拷贝 `assets/isaac45/`，再用 `python3 scripts/fetch_isaac_assets.py --verify-only` 核对。

## 3. 构建容器

先确保 `docker info` 正常、NVIDIA Container Toolkit 已配置，Docker 能使用 GPU。项目不替使用者安装系统驱动。

```bash
docker build -t reasan-a2:sim45-v1 runtime
python3 scripts/check_handoff.py
```

基镜像固定为 Isaac Sim 4.5.0 的 x86_64 摘要。Python 3.10.15、Torch 2.5.1/cu118 保持参考组合；附加依赖按参考镜像锁定。运行时使用 REASAN 子模块中的 Isaac Lab 与 RSL-RL 源码，仿真容器关闭网络。宿主环境不用安装 Python 仿真依赖。

已验证硬件为 RTX 4060 Laptop 8 GB、16 GB 内存。首次构建需下载 Isaac Sim 大镜像及附加包，时间取决于网络；首次启动也会编译缓存。已有兼容镜像可通过 `REASAN_RUNTIME_IMAGE=镜像名称` 显式选择，但自行更换版本后的结果需要重新核查。

## 4. 从短检查开始

根目录 README 提供两条检查命令。第一条验证学生导出、观测和 PD；第二条用六条验证路线跑到终止，包含原模型加载、仿真射线、三层控制和复位。每次使用新的输出目录。

正式运行入口：

```bash
bash scripts/run_static_evaluation.sh 10
bash scripts/run_a2_valid_spawn_evaluation.sh 10
```

每套 240 回合，预计各约 25～50 分钟，由使用者在终端运行。不要使用旧 `run_a2_static_evaluation.sh` 作为修正版成绩；它有意保留原固定高度初始化，用于历史对照。正式结果使用每类八个布局，每布局十组初态，不根据正式成绩改参数或挑权重。

检查 `summary.json` 的 `complete=true` 且 `completed_episodes=expected_episodes=240`。A2 修正版还记录 `spawn_configuration.json`、`spawn_records.json`；接触失败从第一步开始计入。具体协议见根目录 `EVALUATION_PLAN.md`。

## 5. 恢复历史记录与视频

```bash
python3 scripts/download_release.py --with-results --with-videos
python3 scripts/restore_release.py --with-results --with-videos
```

恢复后目录与原项目一致，根目录各报告的链接可直接打开。五段当前录像位于 `results/debug/20261118_122749_a2_static_videos/`，其中 `REPORT.md` 列出三类成功、迷宫超时和迷宫碰撞，并记录与正式测试的逐项对照。更早的出生穿地成绩及录像作为历史保留。

附件清单及摘要在 [RELEASE_ARTIFACTS.json](RELEASE_ARTIFACTS.json)。恢复程序不会覆盖已经存在且内容不同的文件。

## 当前结论与后续边界

Go2 为 220/240；A2 出生修正版为 207/240。两者使用本地静态评估定义，不直接与论文表格相等同。A2 的安全和导航策略仍来自 Go2，控制响应和身体几何存在迁移差异。如果后续工作在 A2 上训练新的导航策略，当前零样本结果只能作为迁移对照；主要算法对照应另行确定 A2 上的适配训练方案。

公开动态原环境已经检查运行；完整动态迷宫测试设置缺失，没有自行补造动态成功率。原始训练入口保留，但当前交付不宣称 A2 安全或导航训练已经验证。

## 常见情况

- **找不到权重、USD 或冻结地形**：先恢复资产附件，再执行 `check_handoff.py`。
- **Git 子模块为空或提交不同**：执行 `git submodule update --init --recursive`，不要用其他项目的 Isaac Lab 替换。
- **没有容器运行权限或 GPU 不可见**：先处理宿主 Docker / NVIDIA Container Toolkit；不要在宿主 Python 中安装 Isaac 依赖来绕过。
- **输出目录已存在**：换新目录。正式入口会自动生成时间戳目录。
- **MDL 地面纹理警告**：原离线资产存在此显示缺口。视频回放使用独立灰色材质；不要为消除显示警告改动物理场景。
