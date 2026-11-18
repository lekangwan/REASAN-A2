# 第三方来源与使用范围

本仓库是实验复现与迁移交接，不代表原作者发布。

- **REASAN**：ASIG-X/REASAN，固定提交 `655bf4f1cc23e835d87bf9a887972af6760af162`，通过子模块引用。官方模型仍归原作者；模型摘要见 `assets/OFFICIAL_MODELS.json`。上游包含 Isaac Lab、RSL-RL 等组件，按各自目录中的原许可证使用。未对全部上游内容统一授予新的许可证。
- **A2 与 MoE**：来源 `kilsnakes-star/a2_rl_gym` 提交 `d3d8929ae615504eb9edf454714505c3c01dbcba`，使用已有 `model_103500.pt` 与对应学生导出。必要源码副本、原 LICENSE 及来源位于 `upstream/a2_reference/`；资产、参考与权重的逐文件清单位于 `assets/a2/SOURCE.json`。checkpoint 未包含完整训练配置。
- **NVIDIA Isaac Sim**：使用 NVIDIA 官方 4.5.0 容器及固定摘要。容器从 NVIDIA 获取；七个运行资产从 NVIDIA 官方地址下载并校验，不放入本仓库的 Git 历史或版本附件。运行适用 NVIDIA Isaac Sim 的条款，见 [官方镜像说明](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/isaac-sim/4.5.0)。
- **历史结果与录像**：由本项目仿真运行产生。具体路线、模型与评估定义随附件保留，不替代原论文成绩。

本仓库按用户要求公开交接，保留各来源的作者、许可证和逐文件来源记录。公开访问不改变第三方内容的权利归属；REASAN 通过官方子模块引用，A2 副本附原 MIT 与 BSD 许可证，NVIDIA 内容仍由使用者从官方获取。
