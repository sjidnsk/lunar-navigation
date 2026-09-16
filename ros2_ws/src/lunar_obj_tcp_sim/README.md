# lunar_obj_tcp_sim

OBJ-backed virtual observations and the COMStructType.h / ControlComponent.cpp / TCPComponent.cpp TCP bridge for the formal incremental navigation, exploration and controller stack.

Entry points: `prepare_obj_map`, `tcp_vehicle_bridge`, `obj_virtual_sensor`. Launch: `obj_tcp.launch.py`. Operator scripts: `scripts/simulation/{build,prepare_obj_map,run_obj_tcp_sim}.sh` at repository root.

See `docs/OBJ_TCP数字仿真使用说明.md` for build, terrain conversion, nav/explore modes, optional RViz, protocol, configuration, stopping and limitations; `docs/OBJ_TCP数字仿真验证记录.md` for evidence. Tests use a loopback TCP vehicle, not remote Unreal physics. No timestamp extension; VehicleID framing follows the C++ reference. OBJ/feedback alignment remains an external commissioning step.

Local interactive RViz entry: `scripts/simulation/run_local_rviz_demo.sh [--mode nav|explore] [--map DIR] [--no-rviz]`. It starts a loopback-only kinematic TCP vehicle; no remote Unreal is required. It does not implement UE collision physics.

TCP 末端使用四轮独立转向/驱动运动学（`kinematics.py`），发送四轮速和四转角。轮序、符号、转角零点在 `config/simulation.yaml` 配置，转向轮序及正角向右已通过 Unreal 逐通道观察确认（steering_signs 全为 -1）；驱动槽前后身份和机械极限尚未逐项标定。车体路径跟踪仍使用 `v, ω`，不增加横移规划。
