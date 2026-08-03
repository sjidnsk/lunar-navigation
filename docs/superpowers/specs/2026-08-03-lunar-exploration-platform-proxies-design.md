# 月面探索参数化物理代理设计

## 目标

在已导入 Isaac Sim 的月球极区 USD 中，建立轮式、足式、飞跃式三类可辨识、可碰撞的参数化物理代理，用于验证本项目的规划输入、地形约束和参考输出边界。为现有 260 块月岩补充独立静态碰撞体，并把极区太阳明确为 5° 入射角、可投射长阴影的定向光。

## 范围与边界

- 场景根层仍为仓库外的 `lunar_polar_terrain_5deg_physics.usda`；新 USD 与所有脚本、验证报告均写入 `~/CodexDownloads/lunar_navigation/isaac_sim/`。
- 平台是实验代理，不是实物 URDF、飞行器、关节驱动器或经过动力学标定的数字孪生；不会输出真实控制命令或宣称足端可行性。
- 每块月岩必须拥有自己的碰撞 API 和凸包近似；岩石保持静态，不赋予刚体 API。
- 三个平台各自拥有一个动态刚体根、独立的简化凸碰撞体、质量和以米/千克/秒表示的自定义能力数据。它们不使用铰链、轮驱动或推进器动作器。
- 继续使用 1 米 stage unit 和 `1.62 m/s²` 月球重力。平台初始位置由地形网格采样决定，机体底部高出接触面 0.03 m，避免初始穿透。

## 极区环境

`/World/LunarPolarSun/LunarPolarSun_Light_002` 作为唯一直接太阳光。其沿地平线方向的投影指向东北，向下分量对应 5° 太阳高度角；光源角直径设为 0.53°，强度设为 2000，启用阴影。环境光不另外补亮永久阴影坑底。

岩石网格各自应用 `UsdPhysics.CollisionAPI` 和 `UsdPhysics.MeshCollisionAPI`，近似模式为 `convexHull`。月壤仍保留现有静态三角网格碰撞（`none`），让动态平台代理接触真实地形表面。

## 平台代理和能力数据

能力字段与 `platform-control-capability-source/v1` 的三类平台语义一致；下列数值是本次 Isaac 实验专用的显式代理参数，不替代外部平台控制单位未来交付的能力资料。

| 类型 | Prim | 形状与质量 | 代理能力参数 |
| --- | --- | --- | --- |
| `WHEELED` | `/World/LunarExplorationPlatforms/WheeledScout` | 1.20×0.85×0.35 m 车体、四个半径 0.18 m 轮、50 kg | `clearance_m=0.22`、`max_slope_deg=18`、`max_obstacle_height_m=0.18`、`max_speed_mps=0.70`、`max_acceleration_mps2=0.40`、`min_turn_radius_m=0.70`、`motion_primitives=arc_and_line` |
| `LEGGED` | `/World/LunarExplorationPlatforms/LeggedScout` | 0.95×0.65×0.28 m 机体、六条固定姿态腿/足垫、35 kg | `body_reference=base_link`、`body_height_m=0.55`、`clearance_m=0.25`、`max_slope_deg=28`、`max_roughness_m=0.12`、`max_step_height_m=0.22`、`max_speed_mps=0.45`、`max_acceleration_mps2=0.35`、`motion_primitives=body_lattice` |
| `HOPPER` | `/World/LunarExplorationPlatforms/HopperScout` | 0.55 m 球形机体、固定伸腿与足垫、20 kg | `landing_clearance_m=0.25`、`max_landing_slope_deg=15`、`max_landing_roughness_m=0.08`、`landing_region_radius_m=0.65`、`max_launch_speed_mps=2.20`、`max_flight_time_s=3.00`、`minimum_settle_guard_s=1.00`、`actuator_or_impulse_profile=proxy_impulse_v1` |

每个平台根 prim 都必须记录 `platform_id`、`platform_type`、`capability_version=proxy-v1`、`base_frame_id=base_link` 和上述数值，供后续读取而不是从几何反推。三者分别放在 `(-38,-30)`、`(-30,-30)`、`(-22,-30)` 米的地形安全采样位置，避免主永久阴影坑与彼此的初始重叠。

## 验证标准

1. 成功打开生成的派生 USD，且根层不是原始地形 USD。
2. 统计到 260 个岩石网格和 260 个带 `CollisionAPI` 的独立岩石碰撞体；每个岩石为静态、凸包近似。
3. 地形仍带静态三角网格碰撞，重力仍为 `1.62 m/s²`。
4. 三个平台根各有动态 `RigidBodyAPI`，子碰撞体均有效，质量分别为 50、35、20 kg，初始变换无相交。
5. 每个根 prim 的 `platform_type` 分别为 `WHEELED`、`LEGGED`、`HOPPER`，能力数据完整且与上表一致。
6. 唯一太阳灯启用阴影，光束向下分量与 5° 入射角一致；不额外设置环境补光。

## 非目标

- 不导入或制造任何真实机器人的 URDF、关节链、控制器、传感器模型或 ROS 2 bridge。
- 不把本次代理参数写入外部输入权威基线，也不覆盖未来真实平台能力资料。
- 不保存 `.blend`，不修改仓库已有 ROS 接口或用户未提交文件。
