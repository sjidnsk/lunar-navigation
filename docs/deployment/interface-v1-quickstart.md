# interface-v1 Ubuntu 联调快速开始

该流程在 Ubuntu 22.04 amd64、ROS 2 Humble 和 ONNX Runtime CPU 上完成旧定义接口联调。它不会启动正式训练，不生成 TensorRT engine，也不代表 AGX 验收。

## 1. 外置 Release 构建

```bash
cd /mnt/data/WS/.lunar-navigation-worktrees/interface-v1-fed9
export LUNAR_VOLUME1_OUTPUT="$HOME/CodexDownloads/lunar_navigation/interface_v1/runtime-fed9"
./scripts/build_runtime.sh
source "$LUNAR_VOLUME1_OUTPUT/runtime/install/setup.bash"
```

构建包含 `lunar_model_contract`、`lunar_external_adapter`、`lunar_exploration_policy`、`lunar_planner_core`、`lunar_planner_ros` 和训练桥。构建、安装和日志都位于仓库外。

## 2. 安装运行依赖和固定文件

建议把 Python wheel 放在仓库外的独立环境：

```bash
python3 -m venv --system-site-packages \
  "$HOME/CodexDownloads/lunar_navigation/interface_v1/runtime-venv"
"$HOME/CodexDownloads/lunar_navigation/interface_v1/runtime-venv/bin/pip" install \
  'onnxruntime==1.23.2' 'rasterio==1.4.4' 'shapely==2.1.2' PyYAML
export PYTHONPATH="$HOME/CodexDownloads/lunar_navigation/interface_v1/runtime-venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
```

选择一个完整平台文件并直接替换固定路径；脚本只安装配置，不启动服务：

```bash
sudo ./scripts/install_interface_v1_config.sh \
  "$PWD/ros2_ws/src/lunar_navigation_config/config/platform_profiles/wheeled.yaml" \
  "$PWD/ros2_ws/src/lunar_navigation_config/config/interface_profiles/default.yaml"
```

以后切换足式或飞跃式，只把第一个参数换成 `legged.yaml` 或 `hopper.yaml`，然后重启下列节点。interface-v1 只接受身份清单中的三份冻结文件。

## 3. 启动并激活

统一 launch 会读取 interface YAML：标准消息直接连接 provider Topic，自定义消息连接显式 converter 输出。因此以后修改外部 Topic 或消息适配只替换文件并重启，不要手工修改 planner/policy 订阅。QoS 按通道固定并由节点实现，不能通过 YAML 改写。安装包已经自带 fed9 能力冻结资料，启动和 configure 不依赖源码仓路径：

```bash
source /opt/ros/humble/setup.bash
source "$LUNAR_VOLUME1_OUTPUT/runtime/install/setup.bash"
export PYTHONPATH="$HOME/CodexDownloads/lunar_navigation/interface_v1/runtime-venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
ros2 launch lunar_exploration_policy interface_v1_system.launch.py \
  model_dir:="$HOME/CodexDownloads/lunar_navigation/interface_v1/model-fed9-step251-20260811"
```

三个节点均为 lifecycle 节点；确认输入发布方已运行后执行：

```bash
ros2 lifecycle set /lunar_external_adapter configure
ros2 lifecycle set /lunar_external_adapter activate
ros2 lifecycle set /lunar_planner configure
ros2 lifecycle set /lunar_planner activate
ros2 lifecycle set /lunar_interface_v1_policy configure
ros2 lifecycle set /lunar_interface_v1_policy activate
```

## 4. 核对接口与滚动闭环

```bash
ros2 action list -t | grep /plan_motion
ros2 topic list | grep -E '/environment/map_(global|local)|/localization/odometry|/mission/exploration_task|/execution/motion_feedback|/lunar/motion_reference'
ros2 topic echo /lunar/interface_v1/status
```

外部执行器订阅 `/lunar/motion_reference`，完成后以其中相同的 `plan_id/segment_id`
发布 `/execution/motion_feedback`。状态消息包含当前平台、状态机、原因码和最近一次观测/推理耗时。
地面平台只在匹配 `SEGMENT_COMPLETE`，且全局图、局部图、里程计、定位状态和 map→odom TF
都晚于完成反馈后发下一次目标；飞跃式只在匹配 `LANDED_HOLD` 且同一组状态全部更新后再次选点。
输入还必须满足 1 秒新鲜度和 0.2 秒组内时间偏差限制；该反馈新鲜度与规划器 tracker
保持一致。若对接项目只需修改 Topic 名，使用 ROS remap；若修改消息字段，
在 `lunar_external_adapter` 增加显式 converter，不修改旧模型观测、动作和权重定义。

## 5. 回归和停止

```bash
export LUNAR_INTERFACE_V1_UNDERLAY="$LUNAR_VOLUME1_OUTPUT/runtime/install"
export LUNAR_INTERFACE_V1_PYTHON="$HOME/CodexDownloads/lunar_navigation/interface_v1/runtime-venv/bin/python"
./scripts/run_interface_v1_regression.sh \
  "$HOME/CodexDownloads/lunar_navigation/interface_v1/model-fed9-step251-20260811"
```

回归证据写入 `~/CodexDownloads/lunar_navigation/interface_v1/regression-*`。停止前先执行 lifecycle `deactivate`、`cleanup`，再在启动终端按 `Ctrl-C`；策略节点停用会取消其拥有的 Action，并且不会复用旧 episode 状态。
