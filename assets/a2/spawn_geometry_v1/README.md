# A2 出生碰撞几何 v1

`colliders.json` 从现有 A2 USD v1 导出，包含实际参与碰撞的 37 个形状（29 个盒、4 个胶囊和 4 个球）以及它们在各自刚体中的变换。原始 USD、URDF 和权重来源见上级目录的清单。

独立初始化环境按采样后的关节角计算碰撞几何，读取冻结地形中的局部最高地面，仅增加出生高度，使最低碰撞形状与地面至少间隔 0.031 米。若出生范围含高障碍物则报错，不通过抬到障碍物上方绕过问题。固定间隙用于避免初始穿透并留出离散接触余量；地面的自动 contactOffset 保持原设置，不能把 USD 的 `-inf` 自动取值标志视为实际偏移。

重新导出到新文件：

```bash
bash scripts/run_in_sim.sh /workspace/scripts/extract_a2_spawn_colliders.py --out /workspace/results/tmp/a2_colliders_check.json
```

检查报告和新评估命令见根目录的 `RUNNING.md`。
