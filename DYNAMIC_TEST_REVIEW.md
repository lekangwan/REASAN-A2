# 公开动态测试流程核查

2026-11-17。目标是忠实执行已公开流程；静态 baseline 保留为 22/24，不因结果调整参数。本次只查阅代码与资料。

## 公开材料支持什么

[论文 IV-A](https://arxiv.org/html/2512.09537v1#S4.SS1)描述 ScaSparse、ScaDense 同时包含静态和动态障碍，每回合 15 秒；DyMaze 使用周期性阻挡通道的动态障碍，每回合 30 秒。论文未给出足以还原这些测试的完整运动轨迹、初始相位和数量配置。

官方 [README](https://github.com/ASIG-X/REASAN/blob/655bf4f1cc23e835d87bf9a887972af6760af162/README.md)公开的导航演示命令为：

```bash
python scripts/play_nav.py --load_run nav_1 --confirm --with_dyn_obst
```

它加载导航课程地形 `GO2_NAV_TERRAIN_CFG`，默认三个并行环境，并非直接调用论文四类场景的批量测试。既有本地 `check_pretrained.py` 可以在同一原始环境中有界检查官方导出模型，避免重新导出权重。

## 动态模块已实现的部分

在[官方导航环境](upstream/REASAN/training/go2_lidar/go2_lidar/tasks/go2_nav_env.py)中：

- 开启动态开关后，每个环境创建九个候选物体；实际启用数量随课程变化，初始可为零。候选物体为圆柱或方柱。
- `_reset_obstacles()` 在机器人起点周围采样两个端点，`_update_dynamic_obstacles()` 使物体在端点间往返。距离目标端点不足 0.5 米时交换端点，并重新采样速度，默认范围为 0.5～1.5 米/秒。
- 物体采用运动学刚体，每个控制步更新其位置。原始布置包含高度、朝向及随机禁用条件，不应为获得更多碰撞或更高成功率而调整。
- 默认三个导航环境的类型分别为一般散布、迷宫、迷宫。原代码把类型 2、3 的动态物体放到地下，所以真正的动态检查必须包含类型 1，单环境导航全部为类型 3。
- [自定义射线模块](upstream/REASAN/training/go2_lidar/go2_lidar/sensor/raycaster_dynamic.py)会读取动态物体位姿并参与射线计算。其类注释中“仅支持静态”的旧说明与当前更新函数不一致，应以实际实现为准。

## 不能直接作为 DyMaze 评估的原因

在[公开测试地形生成器](upstream/REASAN/training/go2_lidar/go2_lidar/terrain/test_terrain_generator.py)中，静态迷宫与动态迷宫复用同一组十个文字布局，通过相同函数生成墙体。函数虽支持解析 `o` 障碍标记，但这十个布局均不含 `o`；两个调用处也都用 `_` 丢弃了返回的障碍位置。

任务与演示代码没有消费 `dynamic_maze_start/goal`，也没有把上述测试地形接入周期阻挡通道的运动设置。直接打开 `--with_dyn_obst` 只启用已有课程环境的往返物体，不会自动构成论文 DyMaze。

通过官方 GitHub 接口重新核查：当前只有 `main` 分支，仍为固定提交 `655bf4f1cc23e835d87bf9a887972af6760af162`，没有 release。现有七个 issue 的标题与正文没有补充动态评估设置；未将此表述为对全部评论或外部资源的穷尽调查。[分支](https://api.github.com/repos/ASIG-X/REASAN/branches)、[release](https://api.github.com/repos/ASIG-X/REASAN/releases)、[issue 列表](https://github.com/ASIG-X/REASAN/issues?q=is%3Aissue)。

## 建议范围与准备好的检查

建议先验证官方原环境的动态导航演示，记录是否实际出现地面上的活动物体、策略数值和回合复位是否正常，将其作为流程检查。完整 DyMaze 成功率等待足够的官方设置；另建本地动态协议会引入新的评估选择，需要单独决定。

拟复用现有 Docker 入口与检查脚本，使用官方演示默认的三个环境，运行 20 秒仿真，允许原课程和复位逻辑自然执行，不强行增加活动物体数：

```bash
bash scripts/run_in_sim.sh /workspace/scripts/check_pretrained.py \
  --task nav --num-envs 3 --dynamic --seconds 20 \
  --out /workspace/results/tmp/YYYYMMDD_HHMMSS_dynamic_nav_check
```

上面的时间目录为占位符，实际执行前生成新目录。预计含地形与应用初始化约 2 分钟，属于有界流程检查。已有记录可核对实际地面物体数量；课程数量非零仍不能单独证明物体在场景中。若检查中没有出现实际活动物体，如实报告，不更改课程制造结果。

用户已选择先验证官方动态演示，上述有界检查已完成。三个 Go2 均运行 20 秒，完成 1000 次策略更新；第 15 秒各触发一次原超时并复位，之后一般散布环境实际出现两个动态物体，两个迷宫环境的物体仍隐藏。数值正常，没有因碰撞或倾斜触发环境终止，总耗时约 97.8 秒。

本轮完成原动态环境的策略运行与复位流程检查，不增加动态成功率。记录按现有脚本保存物体实际数量，没有逐物体轨迹及射线命中；不能将其扩展解释为完整避障效果验证。运行实现、权重与静态 baseline 均保持原样。[检查报告](results/debug/20261117_200502_dynamic_nav_check/REPORT.md)保留命令、结果和来源。
