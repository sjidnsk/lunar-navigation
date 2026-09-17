# Orin 静态 TF 与导航配置

P3 负责发布定位、高程与静态 map→odom；P4 适配器只接收 `/tf_static`
中的直接 map→odom，并以 5 Hz 原样发布到 `/P4/input/map_to_odom`。
输入 Reliable/TransientLocal；保留源时间戳，不构造单位变换、不发布共享 TF、
里程计或车辆控制。重复发布仅代表适配器存活，不代表新定位观测。

`config/orin_hardware.yaml` 是实机配置，不改变 demo 的 `/tf` 默认值。
相对平台配置路径按配置文件所在目录解析，部署不依赖工作目录。
`wheel_orin.yaml` 保留车体几何、坡度、起伏、间隙与运动基元，平台前后速度上限
为 0.2 m/s，起点盲区余量 2.0 m，总半径约 2.719 m。UNKNOWN 补全仍须通过
支撑平面和真实出口检查，不清除 BLOCKED。执行反馈保持开启，不伪造到达。

完成 Humble 构建后，在两个终端分别运行（不要与旧导航重复启动）：

```bash
bash scripts/orin/start_static_tf_input.sh
bash scripts/orin/start_hardware_navigation.sh
```

两脚本默认域 59，不启动 P3、探索器、控制器或 RViz 目标桥。网络配置默认读取
`/home/yanfa/program/cyclonedds.xml`，可预先设置 CYCLONEDDS_URI；运行文件默认
位于 deployment-evidence，可用 P4_RUNTIME_DIR 指定。SIGINT 停止对应终端。
P3 定位重启/原点变化后必须结束旧任务、重启导航与适配器。

回退使用原部署目录及归档启动参数；不能在旧地图会话中途换原点。
源码对齐及分层测试进度见 [104/216 对齐记录](ORIN_NON_TCP_ALIGNMENT.md)。
P3 当前由另一会话负责恢复。2026-09-16 已按用户要求单独切换 P4 到候选目录，
未发送目标或启动车辆控制。实时规划与实车到达不可由隔离测试代替，原生回归仍有未通过项。
