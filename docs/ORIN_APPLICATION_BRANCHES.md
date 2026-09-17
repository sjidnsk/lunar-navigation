# Orin 应用分支

- `app/orin-unreal-simulation`：Orin 通过 TCP 连接 Unreal 的仿真应用线，允许包含 Unreal 协议、TCP 车辆适配和仿真工具。
- `app/orin-real-vehicle`：Orin 实车应用线，接收 P3 地图、位姿和 TF，并使用实车控制接口；不引入 Unreal/TCP 适配。
- `integration/pure-planner-orin`：通用算法集成主线。可复用的规划、探索、地图和控制算法修复应先在独立功能分支验证，再同步回该主线。

两条应用分支可以有不同的输入适配、启动脚本、配置和部署工具。通用算法差异必须明确记录，不能因部署方式不同而长期分叉。
