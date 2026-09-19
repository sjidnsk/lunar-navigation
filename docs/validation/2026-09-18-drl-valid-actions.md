# 有效探索动作与折扣学习验证

2026-09-18，本机 Jazzy / RTX 5070 Ti Laptop，feat/drl-exploration-redesign。

## 实现

GraphBuilder 统一过滤动作：移动位置超出原生位置容差；原地转向超出朝向容差，并有射线可见的任务相关待观测前沿。零即时收益移动和任务外绕行保留；访问信息不作为禁止回访的条件。空动作集合不代表耗尽，耗尽仍由 TaskAnalyzer 判定。

默认 gamma=0.995（支持0<gamma<1）、目标熵系数0.10。奖励、网络结构、alpha上限、8环境、batch64、30倍目标倍率、30分钟保存不变。task_graph_v2 / joint_pose_valid_observation_v2 阻止旧模型或回放被静默当作新契约恢复；旧模型和产物未删除。

新增固定小场景配置 config/drl_exploration_small.yaml；所有阶段均采样40～80m。正式课程配置仍保留原混合分布。两者输出分别为 training-output/drl-valid-actions-small 和 training-output/drl-valid-actions。

## 验证结果

- 新动作/配置测试：旧实现4项失败，修改后通过。
- 完整Python包：235 passed, 2 skipped，58.24s。跳过的是显式ROS开关测试。
- 原生ROS测试另外开启：2 passed，21.05s。实际导航、公共控制器、移动/旋转、PLAN_FOUND与引用、GOAL_REACHED、速度上限及冻结动作转移。
- 受影响包构建成功；6个修改的运行模块源码与安装副本逐字节相同。
- 8环境GPU短训：40m / 每回合8步 / warmup4 / 有效batch64；41条转移、8次更新，11条有新增面积、共49.08m²（跨场景求和），41次GOAL_REACHED，0次同时零移动零转向；正常保存退出。
- 从同一检查点恢复：新增10条转移，累计51条、11次更新；正常保存退出，owned_children_closed=true。
- 总短训统计：{"transitions": 51, "positive": 19, "zero_motion": 0, "reasons": {"GOAL_REACHED": 51}, "events": {"checkpoint": 3, "progress": 12, "transition": 51, "unfinished_excluded": 8}}

日志：/tmp/drl-valid-actions-suite-final.txt、/tmp/drl-valid-actions-native.txt、/tmp/drl-valid-actions-build.txt、/tmp/drl-valid-actions-smoke.txt、/tmp/drl-valid-actions-resume.txt。测试只保留/tmp/drl-valid-actions-smoke的一份恢复模型，无项目内新增模型。

首次短训调用DDS起始域230超出8环境允许范围，被启动检查拒绝；改用220～227后实际运行成功。未修改DDS约束。

## 验证边界

短训只证明采集、更新、有效动作执行与保存恢复，不证明消除了所有循环或学习收敛。固定失败状态的重新训练后策略改善、完整小场景训练、80/99指标、长通道/区域外绕行成功率、折扣0.999对照、Humble/Orin/真实课题三及车辆均NOT_RUN。未自动启动无限长训。
