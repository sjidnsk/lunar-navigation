# 传感器观测能力与探索闭环 Ubuntu 资格报告

日期：2026-08-09

状态：`sensor-qualified / runtime-state-external`

## 结论

批准的保守观测能力 `30 m / 360°` 已接入信息增益估算、实际 reveal、三平台探索宏步和正式
训练身份。本项目只近似传感器系统能可靠探测的范围，不复制多模态系统内部融合算法。30 m
不会自动把圆内栅格写成已知；策略状态和奖励只读取实际 reveal 产生的 valid/age/quality/count
及新增 ROI 覆盖。

本报告只表示传感器观测链具备正式训练资格，不复制训练实时状态。完整资格与外部运行状态入口以
`docs/validation/2026-08-08-formal-training-environment-qualification.md` 为准。

## 唯一现行语义

- 三平台共享 `sensor_range_m=30.0`、`sensor_fov_deg=360.0`；
- 4 m observed-only 全局层负责候选和全局摘要，0.2 m 局部层负责 reveal、局部裁剪和规划认证；
- 物理障碍按确定性 LOS 遮挡，首个阻挡栅格可见，后方不可见；
- `theta` 是候选处绝对 map yaw，不是传感器指向；360° FOV 不消除地面平台的运动学 theta
  学习信号，飞跃式 theta 在策略和 PPO loss 中 mask；
- 飞跃式参考推进剂只定义每次相同的单跳包线，不累计燃料；
- 候选潜在增益不能直接写入已观测状态或奖励。

现行训练语义由 `training_semantics_sha256()` 绑定进 cache、RunIdentity、checkpoint、性能报告
和评估。任何 30 m、360°、LOS 或 observed-only 边界变化都要求重新生成上述资格产物。

## 性能门

正式报告路径：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/qualified-current/sensor-performance.json
```

报告必须由干净 source commit、Ubuntu 22.04 amd64、ROS 2 Humble、RTX 4080 SUPER 和 Release
native runner 生成，并同时通过：

| 工作负载 | 固定输入 | p95 门限 |
| --- | --- | ---: |
| 4 m 候选信息增益 | 256×256、30 m、64 candidates | <= 5 ms |
| 0.2 m 实际 reveal | 320×320、30 m | <= 2 ms |
| 24-worker 闭环吞吐降幅 | W/L/H 各 8、当前 C++ v3/capability v2 | <= 10% |

精确 p50/p95、吞吐、主机、compiler、source commit、能力摘要和训练语义摘要以该 JSON 内部字段
为准；正式 `calibrate/train/resume/evaluate/formal-preflight` 必须严格重验同一报告身份。

2026-08-09 的阶段恢复与训练指标修复改变了被性能门覆盖的训练实现 source identity，因此报告
已在同一主机、同一 Release native 和固定 24-worker 工作负载上重新生成并通过。旧报告以带旧
提交后缀的外部文件保留为历史证据；`run-manifest.json` 只能通过受控 source migration 从已验证
旧 hash 切换到当前正式报告 hash，不能手工跳过身份检查。

## 复现命令

```bash
set +u
source /opt/ros/humble/setup.bash
source "$LUNAR_NATIVE_INSTALL/setup.bash"
set -u
test "$ROS_DISTRO" = humble
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"

python training/tools/benchmark_sensor_observation.py \
  --output /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/qualified-current/sensor-performance.json \
  --native-benchmark "$LUNAR_NATIVE_INSTALL/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark" \
  --workers 24
```

命令要求仓库干净；失败报告不能用于正式训练。外部 Isaac/ROS/RViz 仓仍是独立 Git 根，其旧
30 m/120° 快照只保留为历史证据，不能覆盖本训练语义。
