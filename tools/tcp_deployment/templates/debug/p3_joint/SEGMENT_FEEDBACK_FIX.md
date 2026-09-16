# 局部段切换后未到全局目标：诊断与配置修复

2026-09-16，P4 Orin 原生联调。

现场同一会话路径 revision=2，reaches_final_goal=true，终点 (14.288584,22.825619)，
控制器和转发器均收到同版本，控制器 FAILED / CURVATURE_INFEASIBLE。
控制器已停稳，导航器却持续 EXECUTING。

运行参数确认 enable_tracking_feedback=False，导致导航器没有订阅 TrackingStatus。
原项目 obj_tcp.launch.py 开启此选项；本次 P3 联调 navigation.yaml 漏配。
已在 navigation.yaml 的 navigation 下补 enable_tracking_feedback: true，
仅重启 P4 导航器，未修改 P1/P2/P3，不放宽曲率或到点容差。
原配置备份 navigation.yaml.before-tracking-feedback。

启用后，原代码会核验同会话/路径版本及停稳条件：
FAILED 触发已有偏离重规划逻辑，COMPLETED 参与段结束和最终完成确认。
本次未修改算法，不声称曲率超限本身已经根治。

第一次重启后立即下发目标因地图尚未到达返回 MAP_UNAVAILABLE；
地图收到后重新下发原全局目标，生成约 1.908 m 路径，
依次 ALIGNING、TRACKING、FINAL_ALIGN、COMPLETED，
Action outcome=0、reason=GOAL_REACHED，常驻转发随后 PARKED。
结束实测位置误差 0.0185 m，朝向误差 0.0136 rad。
这验证了启用反馈后的最终段到点；未从未知地图起点完整重演原两段场景，
也未在本轮人为注入 FAILED 验证其恢复分支。

证据 segment-stall.json / segment-feedback-verification.json。
常驻转发继续运行，可在 RViz 下发新目标。
