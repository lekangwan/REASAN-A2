# 运行说明与历史入口

首次在另一台机器使用，请先读 [README.md](README.md) 和 [交接说明](docs/HANDOFF.md)，恢复附件并构建可移植镜像。下文保留本机各阶段入口及历史结果；Go2 与 A2 MoE 均已完成静态流程，当前 A2 属于零样本迁移，训练和雷达估计器仍后置。最新状态见 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)。

## 环境与来源

- 源码：`upstream/REASAN/`，固定提交 `655bf4f1cc23e835d87bf9a887972af6760af162`。
- 运行镜像：本机 `navrl-go2:sim45-lab21-rsl231` 对应的固定镜像 ID，记录在 `scripts/run_in_sim.sh`。使用 Isaac Sim 4.5 / Python 3.10.15 / Torch 2.5.1+cu118。
- Isaac Lab 和 RSL-RL 优先读取 REASAN 随仓库的源码。容器禁止网络访问，上游源码只读挂载，运行输出和缓存写入本项目。
- Go2 及显示资产：`assets/isaac45/`，复制自本机已有的 NVIDIA Isaac Sim 4.5 资产缓存；来源及文件校验值见 `assets/isaac45/SOURCE.json`。官方权重清单见 `assets/OFFICIAL_MODELS.json`。

## 短时间检查

从本项目根目录运行，终端需能访问 Docker 和 GPU。以下是 10 秒仿真检查，启动还需额外时间。

```bash
out="results/tmp/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_loco_check"
mkdir -p "$out"
bash scripts/run_in_sim.sh /workspace/scripts/check_pretrained.py \
  --task loco --seconds 10 --out "/workspace/$out"
```

安全策略检查将 `--task loco` 改为 `--task filter --dynamic`；导航改为 `--task nav --dynamic`。每次使用新的输出目录，名称中的任务名也相应修改。

`summary.json` 记录实际完成步数、环境终止和超时次数、模型来源等；`trajectory.json` 记录运动过程。临时检查结束后，保留有用结论到工作记录，再删除临时输出。正式导航成功率由后续独立评估入口统计。

## 本地入口与官方演示的差异

官方 `play_*.py` 持续演示并重新导出模型。本地短检查直接加载仓库的 `exported/policy.pt`，限定运行时长，不重新导出权重；物理仿真、观测与动作执行仍调用官方环境。

导出模型的外层循环状态随并行环境数量分配，并在对应回合结束时清零。单环境检查降低 PhysX 接触缓冲区容量，保留原物理和控制参数；步态演示在无窗口运行时只忽略镜头更新。详细覆盖项写入每次运行的 `summary.json`。

## 第一轮正式静态评估

评估入口已通过六条检查路线，结果见 [检查报告](results/debug/20261117_184842_static_eval_validation/REPORT.md)。测试采用已确认的 [本地判据](EVALUATION_PLAN.md)，先测试静态稀疏、静态密集和迷宫三个类别，不与含动态障碍的论文成绩直接比较。

在本项目根目录运行：

```bash
bash scripts/run_static_evaluation.sh
```

默认用两个并行环境，运行每类保留的八个布局，共 24 条路线，预计约 4～6 分钟。结果保存在自动创建的 `results/evaluation/YYYYMMDD_HHMMSS_official_go2_static/`；终端会逐条输出到达、超时或终止结果，结束后显示分场景成功率。

`summary.json` 给出是否完整结束、每类样本数、结果分类、成功率及 Wilson 95% 区间；`episodes.jsonl` 保存复位前的最终状态，`trials.json` 和 `initial_states.json` 分别保存计划与实际初始状态，`trajectory.jsonl` 保存运动过程，`console.log` 保存完整输出。用户报告运行结束后，agent 读取结果并分析。

需要增加固定初始朝向的重复测试时，可运行 `bash scripts/run_static_evaluation.sh 10`，此时共 240 个回合，仍只使用每类八个保留布局。重复回合数不能当成新的布局数量；比较不同策略时采用相同的路线清单和并行配置。

## A2 与 MoE 的相同路线测试

A2 资产和冻结学生策略已独立放入本项目，三策略串联通过六条验证路线及复位检查。导航、安全权重继续使用官方版本，动作执行改为 A2 自身的显式 PD。

正式运行：

```bash
cd /home/lekangwan/projects/REASAN
bash scripts/run_a2_static_evaluation.sh
```

自动启动同一 Docker 镜像，两个 A2 测试相同 24 条保留路线，预计约 4～6 分钟。结果保存到 `results/evaluation/YYYYMMDD_HHMMSS_reasan_a2_static/`，除原逐回合与轨迹文件外，`a2_control.jsonl` 保存三层输入、指令、关节动作和 PD 力矩。脚本检查资产和模型来源，不更新参数。

初始随机姿态可能导致 A2 小腿在起步时触地；导航仍按原非足部接触阈值判失败，具体记录见 [验证报告](results/debug/20261117_203940_a2_nav_validation/REPORT.md)。

## A2 成功与失败录像

首轮正式测试中的 `sparse_03_r00`（成功）和 `sparse_02_r00`（启动接触失败）已按首批原条件补录，结果及保存的轨迹、控制记录与正式测试一致。录像采用宽视野跟随画面及右上角地图，见 [视频与来源说明](results/debug/20261117_212153_a2_example_videos/REPORT.md)。

独立入口为 `scripts/record_a2_examples.py`，先以 `capture` 保存两条路线的逐周期机身和关节姿态，再以 `render` 回放姿态生成视频；两阶段均通过现有 `scripts/run_in_sim.sh` 在 Docker 内执行。补录不追加正式统计，原评估入口保持当前版本。

## 已确认的两套扩充评估

Go2 与 A2 各运行 240 回合，每类八个正式布局、每布局十组配对初态，每类共 80 回合。两套顺序运行，预计合计约 45～60 分钟，完整运行由用户在终端执行：

```bash
cd /home/lekangwan/projects/REASAN
bash scripts/run_static_evaluation.sh 10 && \
  bash scripts/run_a2_static_evaluation.sh 10
```

第一套正常完成后才启动第二套。两套均通过已有 Docker 镜像执行；结果分别写入新的 `results/evaluation/YYYYMMDD_HHMMSS_official_go2_static/` 和 `results/evaluation/YYYYMMDD_HHMMSS_reasan_a2_static/`，开头会打印实际目录，保存进度、逐回合结果、初态清单及完整日志。

保留原策略、权重、控制和环境参数、时限及终止判据，全部启动失败仍纳入分母。此次十组包含首轮同一组初态，新一轮单独按每套 240 回合统计，不与旧 24 回合相加。运行结束后核对 `summary.json` 的 `complete=true`、`completed_episodes=expected_episodes=240`，再比较两套清单、逐类与逐布局成绩及失败类型。

本次两套各 240 回合已经完成，见 [Go2 报告](results/evaluation/20261117_222645_official_go2_static/REPORT.md)与 [A2 报告](results/evaluation/20261117_225201_reasan_a2_static/REPORT.md)。A2 已定位出生姿态穿地问题，[诊断报告](results/debug/20261117_230203_a2_spawn_contacts/REPORT.md)记录证据。上述命令仍对应原初始化版本，出生修正版本使用下方独立命令。

## A2 出生高度修正版本的完整重测

全部 240 组初态已通过前 0.5 秒启动和落地检查，六条完整验证路线五条到达、一条迷宫超时。独立版本仅按实际碰撞几何增加出生高度，使整机最低碰撞形状与局部最高地面至少间隔 3.1 厘米。地图、水平起点、朝向、关节采样、权重、控制与终止判据沿用原版本。

```bash
cd /home/lekangwan/projects/REASAN
bash scripts/run_a2_valid_spawn_evaluation.sh 10
```

使用相同 Docker 镜像，每布局十组初态，共 240 回合，预计约 35～50 分钟，由用户执行。结果写入 `results/evaluation/YYYYMMDD_HHMMSS_reasan_a2_valid_spawn_v1/`，开头打印完整目录。原逐回合、轨迹和控制记录继续保存，新增 `spawn_configuration.json` 和 `spawn_records.json` 记录出生规则、原高度、修正高度及实际碰撞几何间隙。

该入口开始前核对已验证代码、地形、碰撞描述和模型来源。用户已完成修正版 240 回合正式评估：稀疏、密集各成功 80/80，迷宫成功 47/80，另有八次碰撞、25 次超时，总计 207/240（86.25%），实际用时约 30.5 分钟。结果见 [本轮记录](results/evaluation/20261118_110315_reasan_a2_valid_spawn_v1/summary.json)，修正规则和检查证据见 [修正与验证报告](results/debug/20261118_101030_a2_spawn_fix/REPORT.md)。

## 修正版的五类路线录像

三类地形各一条成功、迷宫超时与碰撞各一条均已完成，见 [五段视频与来源](results/debug/20261118_122749_a2_static_videos/REPORT.md)。独立入口 `scripts/record_a2_static_examples.py` 从正式首批按原顺序重放，采集所选五条姿态后离线渲染；`scripts/verify_a2_static_videos.py` 核对正式轨迹和控制记录，并检查完整解码。复现命令见录像报告，使用新的输出目录。
