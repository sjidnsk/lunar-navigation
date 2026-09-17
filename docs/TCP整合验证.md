# TCP 开发分支整合验证

日期：2026-09-17。本记录针对 `feat/tcp-development` 本轮本地整合；历史 Orin 验证见各功能原始文档。

## 结果

| 层级 | 结果 |
| --- | --- |
| 本机 Jazzy 仿真链路构建 | 10 个包成功；旧 planner 的 Time 初始化有既有编译警告 |
| 本机 Jazzy lunar_car_ctrl 构建 | 1 个包成功 |
| Python 仿真、控制器、协议、GUI 与停车策略测试 | 273 passed，7 skipped |
| 核心算法、导航 ROS、探索核心 CTest | 首轮通过 |
| 探索 ROS CTest | 首轮 synthetic 场景等待状态超时并缺失结果；同一测试目标独立重跑通过，未改测试或业务代码 |
| Python AST / shell 语法 | 13 个 Python 文件及覆盖层 shell 检查通过 |
| 差异与来源 | diff whitespace 检查通过，47 项原工作树来源 SHA256 保持一致 |
| 当前 Humble / Orin / DDS / Unreal 车辆闭环 | NOT_RUN；Orin SSH 超时 |

探索首轮失败发生在 `UnchangedApproachStallDoesNotResubmitOrComplete`：5 秒内没有产生预期 reference，状态保持初始值；
CTest 后续未生成 gtest XML。仅重跑 `test_synthetic_scenarios` 目标约 2 秒通过。
未认定偶发根因，也不掩盖首轮失败。保留历史结果时 `colcon test-result` 汇总为 867 tests、0 errors、1 failure；该 failure 对应首轮 CTest 报告，独立重跑成功不删除旧报告。7 项 Python 跳过为默认关闭的 TCP 端到端场景，本轮没有启用。
纯协议与策略通过不代表真实 Unreal 已显示路径或车辆能闭环到达。

## 本轮产物（仓库外）

- `/tmp/tcp-development-jazzy/`：Jazzy build/install/log 和各包测试结果。
- `/tmp/tcp-development-vehicle/`：Jazzy 旧车辆接口 build/install/log。
- `/tmp/tcp-development-build.log`、`/tmp/tcp-development-vehicle.log`。
- `/tmp/tcp-development-pytest.log`：Python 最终测试结果。
- `/tmp/tcp-development-ctest.log`、`/tmp/tcp-development-test-results.txt`：保留首轮失败。
- `/tmp/tcp-development-synthetic-retry.log`：场景重跑结果。

这些临时产物不提交，不加入源码包。

## 复现

在分支工作树执行：

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/tcp-development-jazzy/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=184 ROS_LOCALHOST_ONLY=1
export PYTHONPATH="$PWD/ros2_ws/src/lunar_obj_tcp_sim:$PWD/tools/tcp_deployment/templates/debug/p3_joint:${PYTHONPATH:-}"
python3 -m pytest -q \
  deployment/p4_vehicle_gui/test/test_ownership.py \
  deployment/p4_unreal_target/test/test_ue_path.py \
  deployment/p4_unreal_target/test/test_command_policy.py \
  ros2_ws/src/lunar_obj_tcp_sim/test \
  ros2_ws/src/lunar_pure_wheeled_controller/test \
  tools/tcp_deployment/templates/debug/p3_joint/test_command_gate.py \
  tools/tcp_deployment/templates/debug/p3_joint/test_feedback_geometry.py

colcon --log-base /tmp/tcp-development-jazzy/test-log test \
  --base-paths ros2_ws/src --build-base /tmp/tcp-development-jazzy/build \
  --install-base /tmp/tcp-development-jazzy/install \
  --packages-select lunar_incremental_navigation_core lunar_incremental_navigation_ros \
    lunar_pure_exploration_core lunar_pure_exploration_ros \
  --executor sequential --ctest-args --output-on-failure -j1
colcon test-result --test-result-base /tmp/tcp-development-jazzy/build --verbose
```

测试域需保持空闲，仅本机使用；不连接现场 Unreal。

设备端完整脚本同步与当前版本验收尚未完成，见 [开发说明](TCP开发分支.md)。
