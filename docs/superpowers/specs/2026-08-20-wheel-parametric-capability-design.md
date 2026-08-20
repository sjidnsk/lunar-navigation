# Wheel 轮式平台参数能力与无 URDF 部署设计

**状态：** 已完成逐段设计确认，等待书面规格审阅

**适用平台：** `WHEELED`

**平台标识：** `wheel`

**能力版本：** `wheel-v1`

## 1. 目标与边界

本设计为一辆没有 URDF 和 mesh、但具有已确认尺寸与运动参数的轮式平台建立正式运行能力
基线。运行时直接消费参数化整车外包络和运动限制，不生成代理 URDF，不制作占位 mesh，也不把
参数几何描述成部件级机器人模型。

本阶段只启用前进、倒退、行进圆弧、原地旋转和停车换向。平台硬件具有每轮独立转向与驱动、
转向角范围 `±90°`，但规划器暂不使用横移或斜移能力。未来启用全向运动必须形成新的能力版本，
补齐轮组转向角速度、转向角加速度和执行器转换约束，并重新完成规划与执行兼容性验证。

本设计不替换
`ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml`，也不改变
现有正式训练能力摘要。`wheel-v1` 是部署主机上的运行时外部能力文件；当前 `fallback` 规划
可以消费它，但它不能自动证明任何既有策略模型与该平台匹配。

## 2. 方案选择

采用 `platform-control-capability-source/v2` 的参数原生几何扩展：

```yaml
geometry_source:
  type: parametric_envelope
```

加载器把 `geometry_source` 视为互斥联合：

- `type: parametric_envelope`：不允许 `urdf_file`，不访问 URDF 或 mesh；
- 现有 `urdf_file` 形式：保持原有 URDF、mesh、路径安全和 `base_frame_id` 校验；
- 参数形式与 `urdf_file` 混填时返回 `CAPABILITY_SCHEMA_INVALID`。

不选择自动生成基础 URDF，因为派生文件容易被误认为正式机器人模型；不选择占位 mesh，因为
它会把校验绕过物误报为几何证据。

## 3. 正式参数基线

### 3.1 身份和坐标系

```yaml
platform:
  platform_id: wheel
  platform_type: WHEELED
  capability_version: wheel-v1
  base_frame_id: base_footprint
```

`base_footprint` 位于整车平面外包络中心，`x` 轴向前、`y` 轴向左。所有足迹与轮心坐标均相对
该参考点表达。

当前未提供的质量、载荷、重心、悬架、驱动力、厂家额定速度、牵引系数和实测横坡极限不得
推测，必须保留为：

```yaml
unknown_fields:
  - bare_mass_kg
  - nominal_payload_kg
  - maximum_payload_kg
  - center_of_mass_height_m
  - suspension_type
  - maximum_drive_effort
  - manufacturer_rated_speed_mps
  - longitudinal_traction_coefficient
  - lateral_traction_coefficient
  - verified_cross_slope_limit_rad
```

### 3.2 参数几何

| 字段 | 正式值 | 来源 |
|---|---:|---|
| `body_extent_m` | `[1.301, 0.808, 1.363]` | `user_provided_dimension` |
| `footprint_xy_m` | 完整外包络矩形 | `derived` |
| `wheel_count` | `4` | `user_confirmed_capability` |
| `wheel_diameter_m` | `0.304` | `user_provided_dimension` |
| `wheel_width_m` | `0.148` | `user_provided_dimension` |
| `wheelbase_m` | `0.815` | `user_provided_dimension` |
| `track_width_m` | `0.623` | `user_provided_dimension` |
| `wheel_center_xy_m` | 由轴距和轮距推导 | `derived` |
| `minimum_underbody_clearance_m` | `0.214` | `user_provided_dimension` |
| `minimum_clearance_m` | `0.1` | `user_approved_planning_policy` |

完整矩形足迹按稳定顺序表达为：

```yaml
footprint_xy_m:
  - [ 0.6505,  0.404]
  - [ 0.6505, -0.404]
  - [-0.6505, -0.404]
  - [-0.6505,  0.404]
```

四个轮心为：

```yaml
wheel_center_xy_m:
  - [ 0.4075,  0.3115]
  - [ 0.4075, -0.3115]
  - [-0.4075, -0.3115]
  - [-0.4075,  0.3115]
```

加载器不再要求 `track_width_m == body_extent_y - wheel_width_m`。它必须验证轮心数量等于轮数、
轮心不重复、车轮平面外接矩形位于整车外包络内、足迹为非退化凸多边形、底盘净空小于车高，
并验证所有数值有限且尺寸为正数。

规划使用完整外包络矩形，再在障碍物周围额外叠加 `0.1 m` 净空。该几何覆盖整车最大外形，
但不提供车轮转向扫掠、悬架运动、部件级碰撞或 mesh 可视化。

### 3.3 运动和地形能力

| 字段 | 正式值 | 来源 |
|---|---:|---|
| `maximum_forward_speed_mps` | `0.2` | `user_confirmed_capability` |
| `maximum_reverse_speed_mps` | `0.2` | `user_confirmed_capability` |
| `maximum_acceleration_mps2` | `0.2` | `user_confirmed_capability` |
| `maximum_braking_deceleration_mps2` | `0.2` | `user_confirmed_capability` |
| `maximum_lateral_acceleration_mps2` | `0.2` | `user_approved_planning_policy` |
| `maximum_curvature_per_m` | `5.0` | `user_confirmed_capability` |
| `maximum_spin_rate_radps` | `0.389923188554511` | `derived` |
| `maximum_yaw_acceleration_radps2` | `0.389923188554511` | `derived` |
| `maximum_surface_slope_rad` | `0.174532925199433` | `user_confirmed_capability` |
| `maximum_local_obstacle_relief_m` | `0.2` | `user_confirmed_capability` |
| `allow_unsupported_gap` | `false` | `user_confirmed_capability` |
| `roughness_handling` | `COST_SPEED_AND_LOCAL_RECHECK` | `user_approved_planning_policy` |

最小行进转弯半径为 `1 / 5.0 = 0.2 m`。最大角速度和角加速度使用最远轮心半径保守推导：

```text
r_wheel = hypot(0.815 / 2, 0.623 / 2)
        = 0.512921533960118 m

maximum_spin_rate = 0.2 / r_wheel
                  = 0.389923188554511 rad/s
```

圆弧速度必须同时满足：

```text
v <= 0.2
abs(curvature * v) <= 0.389923188554511
abs(curvature) * v^2 <= 0.2
```

因此最紧 `0.2 m` 半径圆弧的速度不超过
`0.389923188554511 / 5.0 = 0.077984637710902 m/s`。原地旋转是 `v=0` 的独立原语，
不表示成无限曲率。

### 3.4 运动原语

局部 XY 分辨率为 `0.2 m`，航向离散为 `64` 个方向，单次航向变化为
`delta_yaw = pi / 32`（`5.625 deg`）。能力文件显式包含：

- 前进和倒退直线，名义位移 `0.2 m`；
- 前进/倒退的左、右圆弧，最紧半径 `0.2 m`；
- 顺、逆时针原地旋转 `pi / 32`；
- 停车换向；
- 不包含横移或斜移原语。

半径 `0.2 m`、航向变化 `pi/32` 的圆弧局部终点由固定公式生成：

```text
x = radius * sin(delta_yaw) = 0.019603428065912 m
y = radius * (1 - cos(delta_yaw)) = 0.000963054665561 m
```

左右、前后原语通过稳定的符号规则生成，四元数使用 `wxyz` 顺序。节点必须保存连续终点；
`0.2 m` 栅格和 yaw bin 只作为搜索键，不能把圆弧终点吸附到格心后再声称保持原始曲率。
`motion_primitives` 的来源为 `user_approved_planning_policy`；每个其余运行时字段必须按上述表格
逐字段出现在 `sources` 中，不能使用一个笼统的 `baseline` 来源代替逐字段证据。

## 4. 文件和加载流程

部署配置继续使用：

```yaml
capabilities:
  platform_file: /opt/luna/capabilities/platform.yaml
  observation_file: /opt/luna/capabilities/observation.yaml
```

当前加载器只允许 package-share 内的相对路径，与部署配置的绝对路径合同不一致。实现必须增加
外部文件加载入口并采用以下互斥规则：

1. `capability_package` 为空，且平台与观测能力路径均为绝对路径：从外部文件加载；
2. `capability_package` 非空，且两个文件路径均为安全相对路径：从 package share 加载；
3. 其余混合形式：以 `CAPABILITY_PATH_MODE_INVALID` 拒绝配置。

`capability_package` 默认值为空字符串。外部文件必须是可读取的普通文件；平台与观测能力文件
分别解析并校验。参数几何不解析任何附属路径。URDF 模式仍以 package share 为信任根，保持
现有防目录穿越和 mesh package 解析行为。

容器中的正式文件为 `/opt/luna/capabilities/platform.yaml`。实施时只写入已确认的 `wheel-v1`
参数，不复用现有轮式工程基线的尺寸。当前容器尚无 `/opt/luna`，因此实施步骤需要显式创建
该目录。

观测能力仍是独立必需输入。不得从平台参数推测 `sensor_range_m` 或 `sensor_fov_deg`，不得用
零值或任意默认值冒充真实传感器能力。缺少观测能力时可以独立校验平台文件，但完整 ROS 节点
必须拒绝配置。

## 5. 失败处理

以下情况必须失败关闭：

- 参数几何缺少必需字段或包含 `urdf_file`；
- 几何数值非有限、非正、足迹非凸或轮组超出整车外包络；
- `platform_id`、`platform_type`、`capability_version` 或 `base_frame_id` 不匹配；
- 运动原语重复、四元数非法、原语类型不在批准集合内；
- `sources` 缺少运行时字段或包含未批准来源类型；
- 能力路径模式混合、文件不存在或 YAML/JSON 无法解析；
- 观测能力缺失或非法。

参数校验失败使用稳定的 `CAPABILITY_SCHEMA_INVALID` 或 `CAPABILITY_VALUE_INVALID`；路径模式
冲突使用 `CAPABILITY_PATH_MODE_INVALID`。错误详情必须指出字段，不得静默回退到 URDF 模式、
现有冻结平台或内置代理值。

## 6. 实现范围

实现预计修改：

- `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp`：参数几何联合、正式字段和外部路径加载；
- `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/capability_loader.hpp`：外部文件加载入口；
- `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`：路径模式选择与空 package 默认值；
- `ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp`：参数模式、兼容模式和拒绝测试；
- `ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml`：参数几何合同；
- `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`：允许的几何来源形式；
- `docs/interfaces/external-input-baseline.md`：无 URDF 参数能力的正式消费边界；
- 部署配置测试：证明绝对路径参数能传入 ROS 节点且不要求 `capability_package`。

当前工作树中已有部署文档、CLI 和部署测试的用户改动。实现必须保留这些改动，只对必要且不
冲突的行做增量编辑，不覆盖或回退现有内容。

## 7. 验收

最低验收条件：

1. `wheel-v1` 在没有 URDF 和 mesh 的临时 share/外部目录中加载成功；
2. 加载结果精确保留平台身份、外包络、轮组参数、速度、曲率、坡度和来源；
3. 参数模式与 `urdf_file` 混填、轮组越界、缺失来源和非法数值均被拒绝；
4. 原有 URDF+mesh 能力加载测试继续通过；
5. 外部绝对路径模式成功，package-relative 模式成功，混合路径模式失败；
6. 最大曲率 `5.0 m^-1` 的速度上限不高于 `0.077984637710902 m/s`；
7. 相关 C++ 单元测试、部署配置测试、外部接口检查和仓库边界检查通过；
8. 容器中的 `/opt/luna/capabilities/platform.yaml` 与本规格逐字段一致；
9. 未提供 `observation.yaml` 时不声称完整运行时启动就绪。

本阶段不启用横移/斜移，不创建 URDF/mesh，不替换训练冻结摘要，不运行正式训练，不发布模型，
也不把参数外包络描述为部件级几何或硬件鉴定证据。
