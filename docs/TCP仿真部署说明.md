# P4 TCP 仿真部署包

本包为 `feat/obj-tcp-simulation` 的源码部署包，参考 `release/orin-humble-20260911` 的 Orin 1.0 精简结构。具体源码提交和参考提交见 `ORIGIN.json`。不包含已编译程序，需在目标机编译。

## 内容与边界

包含 8 个运行包：增量导航 core/ROS、探索 core/ROS/messages、planning_messages、轮式控制器及 OBJ/TCP 仿真；保留控制器依赖的兼容消息，不包含旧规划器算法、源码测试、Git 历史、缓存和编译产物。开发分支保留测试。

附带 `unreal-start-500m-ros` 的完整预处理地图（约 83 MiB，含观测遮挡索引），无需原始 OBJ。地图坐标转换为 `x,z,y`、比例 `0.01`，元数据仍为 `NOT_EXTERNALLY_ALIGNED`；这不是 Unreal 地标配准通过的证明。车辆反馈已在 UE 端由厘米转换到米，桥接按当前协议转换轴和朝向，不要再次乘 0.01。

保持正式探索→导航→控制器→TCP 桥的原架构。默认 ROS 域 74，话题前缀 `/lunar_sim`，端口 6668。保留终点朝向要求，位置容差 0.10 m、朝向容差 0.05 rad；这些是配置要求，不代表坡面实测精度。`nav` 为已知地图选点导航，`explore` 为虚拟传感器观测与自动探索。

## 1. 环境和构建

目标环境为 Ubuntu 22.04 / ROS 2 Humble（Orin 在目标机本地编译）。也支持本机 Jazzy 隔离验证。使用未加载另一 ROS 发行版或旧工程 overlay 的新终端。需预装 ROS、colcon、rosdep、编译器及地图/可视化等依赖；联网安装可执行：

```bash
cd P4
source /opt/ros/humble/setup.bash
# 首次使用 rosdep 的机器需先按系统配置初始化 rosdep。
rosdep update
rosdep install --from-paths ros2_ws/src --ignore-src --rosdistro humble -r -y
bash scripts/build.sh
```

构建输出位于当前包 `artifacts/humble/`，不同源码包与发行版互不复用构建缓存。已有依赖时构建不需要联网。Jazzy 环境每个命令前加 `LUNAR_ROS_DISTRO=jazzy`。源码包不包含 ROS 和系统依赖。

## 2. 不连接 Unreal 的本机演示

```bash
bash scripts/start_local_demo.sh --mode nav
```

使用本机 TCP 运动学车辆与同一正式规划链，只验证程序连接和规划控制逻辑。等待 `KNOWN_MAP_READY`，在 RViz 用 **2D Goal Pose** 选点并拖出终点朝向。不能代替 Unreal 坡面物理验证。无桌面可加 `--no-rviz`，需要另行通过 Action 发送目标。

## 3. 连接 Unreal

先启动 Unreal 的车辆 TCP 服务，将下方地址替换为服务端真实 IP：

```bash
# 仅打印配置，不连接或发送控制
bash scripts/start_tcp.sh --host 192.168.1.100 --port 6668 --dry-run
# 已知地图导航，显示全局和局部窗口
bash scripts/start_tcp.sh --host 192.168.1.100 --port 6668 --rviz --local-rviz
```

地图就绪后，在 RViz 使用 **2D Goal Pose** 指定位置及朝向；观察车辆实际轨迹、全局参考路径和局部跟踪路径。首次联调核对车辆位姿、朝向及地形对应关系。默认附带地图是固定区域，不会根据车辆位姿自动迁移地图；车辆须在其有效范围内。

自动探索：

```bash
bash scripts/start_tcp.sh --host 192.168.1.100 --port 6668 --mode explore --rviz --local-rviz
```

探索启动后按配置自动下发任务；默认任务边长 120 m，传感器距离 30 m、360°。修改源码配置 `ros2_ws/src/lunar_obj_tcp_sim/config/simulation.yaml` 后重启即可由启动脚本读取。程序修改需重新构建。单独使用底层脚本时须自行指定 overlay 和地图。

停止时在启动终端按 Ctrl+C，operator 会取消任务并等待车辆停稳再退出。不要同时运行多套车辆控制器。更换预处理地图使用 `--map /绝对路径/地图目录`，需包含 metadata.json、elevation.npy、sample_xy.npy、triangles.bin、index_ids.npy、index_offsets.npy；地图坐标必须与当前车辆反馈约定一致。

详细协议、地图准备和诊断见 `docs/OBJ_TCP数字仿真使用说明.md`，其中旧绝对路径与历史验证只作参考；本包启动以本文的相对路径入口为准。

## 4. 校验与重建部署包

在解压目录执行 `sha256sum -c SHA256SUMS` 验证源码和地图。压缩包旁另有 `.sha256` 文件，用于传输校验。

在开发分支根目录执行（输出目录必须不存在，默认要求已提交）：

```bash
python3 tools/create_obj_tcp_deployment.py \
  --output /目标目录/P4 \
  --map /地图目录/unreal-start-500m-ros
```

导出器按白名单复制当前 TCP 源码，构建清单采用随源码保存的 Orin 1.0 模板，兼容消息及 TCP 控制器保留当前版本。`--allow-dirty` 仅供候选包验证，会在 ORIGIN.json 中标记，正式交付不用此选项。

## 5. 本次验证

2026-09-16 的本轮验证：

- 本机 Jazzy：开发分支依赖链 10 包构建通过；TCP/控制器 Python 测试 250 passed、7 skipped。跳过项目含显式启用的端到端场景，不能按通过计数。
- Jazzy 导航 ROS CTest 13/13、探索 ROS CTest 12/12 通过。首次导航测试暴露既有诊断异步时序问题（收到 PLAN_FOUND 就断言 EXECUTING）；等待目标诊断状态后复测通过，同一用例重复 10 次通过，未改变运行逻辑。
- Jazzy 本机 TCP 闭环：显式运行 nav、obstacle 两场景，2 passed。包含位置、终点朝向、绕障和有序停车检查。
- Ubuntu 22.04 / ROS 2 Humble x86_64 容器：候选部署包 8 包全新构建通过；显式运行 nav TCP 回环场景，1 passed。三个目标均 GOAL_REACHED，检查 0.10 m / 0.05 rad、PLAN_FOUND 诊断、ACTIVE 路径及退出停车。
- 内部包依赖完整性、Shell 语法、启动 dry-run 和源码/地图 SHA-256 校验通过。构建在只读源码挂载下完成；输出写入独立验证目录，未混入部署包。

上述闭环均使用本机运动学车辆，不模拟 Unreal 重力、轮胎接触或坡面滑移。本轮 Jetson Orin 原生运行、远端 Unreal 联调、GUI 人工操作及坡面物理精度均为 `NOT_RUN`。详细历史验证见配套记录；本轮证据目录 `/tmp/tcp-audit-20260916-103436/`，该临时路径不随包交付。

## P4/P3 现场联合运行（2026-09-16 同步）

源码包同时提供 `debug/p3_joint/` 的 Orin 现场配置，使用说明见该目录 README.md。它使用 P3 的地图和旧车辆接口，并通过域 57/10 常驻转发；不要与本包 TCP 直连启动方式同时运行。源码中的对应模板位于 `tools/tcp_deployment/templates/debug/p3_joint/`。这套配置保留现场绝对路径，迁移机器需检查配置。

探索覆盖率 80% / 99% 仅为阶段指标，不触发完成。详情见 `docs/exploration-completion-policy.md`。
