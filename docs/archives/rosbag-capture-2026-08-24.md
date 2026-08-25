# 2026-08-24 T3 录包归档

原始 rosbag2 数据已从工作区根目录与 `bags/` 移至
`artifacts/rosbags/2026-08-24/`。该目录由 `.gitignore` 排除，不随源码提交；本文件
保留可追溯的索引，避免把运行数据与代码、启动文件混在仓库根目录。

| 归档目录 | 消息数 | 结论 |
| --- | ---: | --- |
| `t3_t4_20260824_132953` | 2 | 仅有一条 `/Car/T3/mapping/grid_map` 与一条 `/tf_static`；不能复现完整规划链路。 |
| `t3_inputs_20260824_152300` | 0 | 空的 MCAP 记录输出；保留以追踪本次采集尝试。 |
| `t3_20260824_163252` | 626 | 有 odometry、`/tf`、`/tf_static` 和一条局部 GridMap；无全局图。 |
| `t3_20260824_183202` | 2 | 仅有一条局部 GridMap 与一条 `/tf_static`；不能复现完整规划链路。 |
| `t3_20260824_192040` | 175 | 含 odometry、`/tf`、`/tf_static` 与 14 条局部 GridMap；无全局图，仍不构成完整月表规划输入。 |

录包元数据列出 topic 不代表该 topic 有消息。重放或正式验证前，先运行：

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag info artifacts/rosbags/2026-08-24/t3_20260824_192040
```

正式采集应包含 `/Car/T3/mapping/global_overview`、
`/Car/T3/mapping/grid_map`、`/Car/T3/localization/odometry`、`/tf` 与
`/tf_static` 的实际消息，并在采集停止后再次检查每个 topic 的计数。
