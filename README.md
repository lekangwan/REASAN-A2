# REASAN-A2：官方 Go2 复现与 A2 导航适配

本仓库交接 REASAN 仿真复现、A2 MoE 零样本迁移和 A2 导航微调权重。评测采用仿真射线。

## A2 静态迷宫最佳组合

**微调第 500 次导航 + 从第 600 次按原奖励续训 150 次的安全层，到达 60/80（75.00%）。** 四组使用同一批 80 条静态迷宫、相同批次顺序和实际初态，固定 A2 `model_103500.pt` 对应的 MoE 学生策略与仿真射线，按到达率比较。

| 导航 | 官方安全层 | 原奖励续训安全层 |
| --- | ---: | ---: |
| 原导航 | 48/80（60.00%） | 45/80（56.25%） |
| 微调第 500 次 | 53/80（66.25%） | **60/80（75.00%）** |

[v0.2.2 最佳组合](https://github.com/lekangwan/REASAN-A2/releases/tag/v0.2.2) · [下载导航权重](https://github.com/lekangwan/REASAN-A2/releases/download/v0.2.2/nav_a2_finetune_iter500.pt) · [下载安全层权重](https://github.com/lekangwan/REASAN-A2/releases/download/v0.2.2/filter_a2_original_reward_600plus150.pt)

已有本仓库运行环境时，先运行 `git pull --ff-only`。把 v0.2.2 的两份 TorchScript 和 `SHA256SUMS` 放在 `release-downloads/best-maze/`，可用下列命令在公开预留迷宫短测。

```bash
(cd release-downloads/best-maze && sha256sum -c SHA256SUMS)
REASAN_A2_FILTER_POLICY=release-downloads/best-maze/filter_a2_original_reward_600plus150.pt \
  bash scripts/run_in_sim.sh /workspace/scripts/evaluate_a2_trained_navigation.py \
  --split validation --family maze --num-envs 2 --repeats 1 \
  --out /workspace/results/tmp/best_maze_validation \
  --policy /workspace/release-downloads/best-maze/nav_a2_finetune_iter500.pt
```

用相同导航与官方安全层做对照时，省略 `REASAN_A2_FILTER_POLICY` 并另选输出目录。 [自有场景的接口说明](docs/A2_NAV500_HANDOFF.md) · [安全层配对与专项结果](docs/A2_FILTER_CANDIDATE.md)

## 官方权重的静态复现与零样本结果

每类八个正式布局，每布局十组固定初态，每类 80 回合。使用同一组静态稀疏、密集和迷宫地图；验证布局另留，不参与此表统计。

| 系统 | 稀疏成功 | 密集成功 | 迷宫成功 | 总成功 |
| --- | --- | --- | --- | --- |
| 官方 Go2 三层权重 | 80/80 | 80/80 | 60/80 | 220/240，91.7% |
| A2 MoE＋冻结 Go2 导航、安全层，出生修正 v1 | 80/80 | 80/80 | 47/80 | 207/240，86.25% |

本地判据：水平目标距离小于 0.5 米算成功；同一步碰撞或倾斜优先判失败；非足部接触超过 1 N 判碰撞；散布场景限时 15 秒、迷宫 30 秒。这是本地评估定义，不宣称完整复现论文的评估协议。重复初态不增加独立地图数量。

A2 原固定 0.4 米出生高度配合随机关节角曾造成穿地。独立 v1 仅按实际碰撞几何增加出生高度，留出 3.1 厘米地面间隙；原判据从第一步生效。旧 A2 92/240 作为含初始化问题的历史结果保留。

五段录像覆盖三类地形成功、迷宫超时与碰撞，位于 [v0.1.0 附件](https://github.com/lekangwan/REASAN-A2/releases/tag/v0.1.0)。恢复结果和视频后可打开 [录像报告](results/debug/20261118_122749_a2_static_videos/REPORT.md)。

## 第一次在另一台机器使用

需要 Linux x86_64、NVIDIA GPU、可用的 Docker GPU 支持、Git 和宿主 Python 3。已验证机器为 RTX 4060 Laptop 8 GB、16 GB 内存、驱动 570.133.07；建议预留至少 35 GB 磁盘。Python 依赖安装在容器里。

```bash
git clone --recurse-submodules https://github.com/lekangwan/REASAN-A2.git
cd REASAN-A2
python3 scripts/download_release.py
python3 scripts/restore_release.py
python3 scripts/fetch_isaac_assets.py
docker build -t reasan-a2:sim45-v1 runtime
python3 scripts/check_handoff.py
```

仓库与版本附件均公开，无需 GitHub 账号即可获取。下载脚本默认匿名下载并核对摘要；也可从 Release 页面手动下载到 `release-downloads/` 后运行恢复命令。

先做短检查，再运行完整评估。完整交接步骤、常见问题和结果恢复方法见 [HANDOFF.md](docs/HANDOFF.md)。

```bash
# 验证 MoE 导出、观测、关节映射与 PD，无需物理仿真。
bash scripts/run_in_sim.sh /workspace/scripts/check_a2_policy.py \
  --out /workspace/results/tmp/policy_check

# 六条预留验证路线，约一至两分钟；每次使用新的输出目录。
bash scripts/run_in_sim.sh /workspace/scripts/evaluate_a2_valid_spawn.py \
  --split validation --num-envs 2 --repeats 1 \
  --out /workspace/results/tmp/a2_validation

# 正式评估：每套 240 回合，由使用者在终端运行。
bash scripts/run_static_evaluation.sh 10
bash scripts/run_a2_valid_spawn_evaluation.sh 10
```

## 目录和改动范围

| 路径 | 内容 |
| --- | --- |
| `upstream/REASAN/` | 官方子模块，固定提交 `655bf4f1cc23e835d87bf9a887972af6760af162` |
| `a2_adapter/` | A2 关节映射、MoE 运控、机器人接入和独立出生初始化 |
| `scripts/` | 本地评估、记录、核查与交接入口 |
| `assets/` | 来源清单及从版本附件恢复的模型、机器人、冻结地图 |
| `runtime/` | 固定 Isaac Sim 4.5 基镜像与附加依赖 |
| `results/` | 从附件恢复的原始结果，以及本机新运行输出，不进入 Git 历史 |

官方 Go2、本地运行适配和 A2 迁移分别保留。`scripts/original/run_in_sim.local.sh` 保存迁移前的本机启动入口。可移植入口支持 `REASAN_RUNTIME_IMAGE` 显式选择镜像。原始训练入口随官方子模块提供。

项目当前状态见 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)，关键实验见 [WORK_LOG.md](WORK_LOG.md)，协议见 [EVALUATION_PLAN.md](EVALUATION_PLAN.md)，底层接入细节见 [A2_MIGRATION_PLAN.md](A2_MIGRATION_PLAN.md)。[第三方来源说明](THIRD_PARTY_NOTICES.md)保留原作者与许可证信息。

原论文与代码：[REASAN](https://github.com/ASIG-X/REASAN)、[论文](https://arxiv.org/abs/2512.09537)。
