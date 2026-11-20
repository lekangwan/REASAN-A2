# A2 安全层候选：原奖励续训 150 次

这是从 A2 安全层微调第 600 次的保存点恢复网络和 Adam、用原奖励继续训练 150 次导出的 TorchScript。文件为 `filter_a2_original_reward_600plus150.pt`。在自有仿真场景做配对时，两组使用相同的第 500 次导航权重、A2 MoE、射线、地图、起终点和初态，分别加载官方安全层或本候选。53/80 的正式迷宫成绩对应官方安全层。 [候选文件](https://github.com/lekangwan/REASAN-A2/releases/tag/v0.2.1) · [导航权重与输入说明](A2_NAV500_HANDOFF.md)

| 专项检查 | 官方安全层 | 原奖励续训候选 |
| --- | ---: | ---: |
| 直墙碰撞终止 | 4/8 | 0/8 |
| 空地平均平移速度误差 | 0.099 m/s | 0.233 m/s |
| 空地停车接触终止 | 0/36 | 0/36 |

候选在直墙工况中的最小墙距为 0.298 米。在 36 个空地工况里，有 30 个工况的误差比同工况下的官方层多 0.10 m/s 以上。空地请求前进 1.8 m/s 时，候选平均发给 MoE 约 2.04 m/s，机身实测约 1.98 m/s。直墙制动和速度跟踪应分别比较。

第 500 次导航配合两种安全层分别跑了六条预留路线，稀疏、密集和迷宫各两条。核对了实际初态、模型及评测入口。官方层到达 5/6，候选 6/6，均无碰撞。迷宫 1 从 30 秒超时变成 23.42 秒到达。这组数据记录的是六条预留路线上的表现。

## 使用

更新仓库后，从 v0.2.1 下载候选权重和 `SHA256SUMS`，放到 `release-downloads/filter150/`。已有的第 500 次导航模型可按根目录 README 放在 `release-downloads/nav500/`。

```bash
git pull --ff-only
(cd release-downloads/filter150 && sha256sum -c SHA256SUMS)
REASAN_A2_FILTER_POLICY=release-downloads/filter150/filter_a2_original_reward_600plus150.pt \
  bash scripts/run_in_sim.sh /workspace/scripts/evaluate_a2_trained_navigation.py \
  --split validation --num-envs 2 --repeats 1 \
  --out /workspace/results/tmp/nav500_filter150_validation \
  --policy /workspace/release-downloads/nav500/models/nav_a2_finetune_iter500.pt
```

该变量为本次仿真容器选用候选文件。用同一命令进行官方层对照时，省略该变量并另选输出目录。可用 `summary.json` 的 `models.filter` 核对实际加载的文件，候选 SHA-256 为 `60c47d3b19289cbc0519e404909999fa534917dbcee49f1c7363507fca2b941a`。

如果直接接自己的代码，沿用 [安全层接口说明](A2_NAV500_HANDOFF.md#安全层和步态) 中的 12 维状态、180 维射线、3 维速度输出及回合复位方式。
