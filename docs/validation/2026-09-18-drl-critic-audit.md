# Critic 设计审核与独立验证（2026-09-18）

对象：`docs/superpowers/specs/2026-09-18-drl-critic-action-alignment.md`。
范围：设计、CPU 原型与现有单元测试；没有改正式源码、模型、奖励、配置或训练产物，也未启动/停止训练。

## 结论

候选位置对应与定向静态剩余可见面积在当前二维观测契约下可行。
建议把候选双向插入真值骨架改为编码后的只读局部几何读取；减少候选集合变化对其他节点表示的意外影响。
g* 应保持为训练特征；明确反例证明它不能替代动作真实收益或用于筛除零值动作。
没有训练新的 Critic，因而不宣称价值排序或循环率已改善。

## 已执行

| 验证 | 结果与边界 |
|---|---|
| 独立原型 | 46 个条目，42 PASS、4 EXPECTED_LIMITATION，全部断言通过；约 4 s，进程峰值约 555 MiB，含 Torch/native 基础开销 |
| 独立视域 oracle | 24 张随机图 × 8 朝向，共 192 组逐格一致；oracle 为闭方格交点实现，不调用 native observe |
| 原生场景 | 月表、洞穴各一张，名义尺度 40 m、0.2 m 网格；各 12 站位 × 8 朝向，共 192 组与一次实际 SensorModel 静态测量的新增参考面积一致 |
| 位置支持 | 八邻接、禁止切角的已接受运动域，2/8 m 有界查询与 SciPy Dijkstra 一致；未验证连续实际起点适配 |
| 读取及动作契约 | 查询顺序、增删其他查询、骨架重排、世界平移、梯度、动作重排/重复、Actor 隔离、变长朝向批次通过 |
| 原有模型/SAC 测试 | 23 passed in 9.16s；验证现有基础契约，不是新 Critic 集成测试 |

原生场景 seed 均为 20260918。月表参考面积约 1208.60 m²，洞穴约 298.60 m²。
生成的观测历史是用于一致性检查的站位采样，不是一条已执行导航路线。

## 关键反例

1. 单个汇总区域的已观测比例均为 0.9880048，但同方向的静态剩余可见面积分别为 0 和 7.5 m²。说明比例有空间信息损失；不等于证明完整 Actor 输入发生混叠。
2. 终点朝向的 g*=0，但另一个途中转向采样可看到 12.25 m² 任务面积。证明 g* 不等于完整动作奖励；未运行控制器轨迹。
3. 在 10° 视野压力案例中，名义位姿可见 0.25 m²，偏航偏差 14° 后为零。它是可配置窄视野的容差反例，不是默认 90° 的统计失败率。
4. 简单消息传递中增加另一个双向查询节点改变骨架均值；只读查询避免这个表示依赖。不是正式模型已发生错误的运行证据。

薄墙案例：直线距离 0.4 m 的对侧节点在合法 8 m 路程内不可达，同侧支持点距离 1.8 m，验证不能直接按直线最近点绑定。
连续接入的抽象边界案例：栅格 r=2 m 覆盖加上 0.1 m 合法接入段后，必须搜索到 2.1 m 才包含代表点。证明正式适配需要计入起点接入代价；未执行原生连续位姿适配。
区外朝向案例：区外同一位置面向任务为 40.75 m²、背向任务为零；观测后重算为零。
简化精确 MDP 中零即时收益的离开动作 Q≈0.89525，返回动作 Q≈0.86637；这是必要转移的数学示例，不是 SAC 收敛实验。

## 计算和存储

单线程 CPU、warm cache、10 m/90°、0.2 m 网格；只测 g* 计算，不含图搜索/编码：

| 唯一位置 | 朝向动作 | 中位时间 | 当前＋下一状态 g* 的 float32 大小 |
|---:|---:|---:|---:|
| 8 | 64 | 8.97 ms | 512 B |
| 16 | 128 | 17.88 ms | 1024 B |
| 32 | 256 | 36.84 ms | 2048 B |

201²、501²、1501² 网格中的相同局部场景，八朝向可见面积相同、时间约 1.1 ms；局部视域 patch 最多 101²。
不能由此推断公里级真值图构建/编码已经通过性能要求。
每条经验两端各 128 动作的标量新增约 1 KiB，10 万条约 97.7 MiB，不含几何支持/容器开销。
不保存逐动作可见位图；不把当前理论存储预算当成已集成的序列化实测。

## 复现

在 DRL 独立 worktree 根目录执行，使用 Jazzy 隔离 overlay：

```bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/drl-exploration-redesign
source /opt/ros/jazzy/setup.bash
source /home/kai/.cache/lunar-drl-redesign/jazzy/install/local_setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  /home/kai/.cache/lunar-drl-training/venv/bin/python \
  training-output/critic-audit-20260918/audit.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/ros2_ws/src/lunar_drl_exploration:$PYTHONPATH" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  /home/kai/.cache/lunar-drl-training/venv/bin/python -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_drl_exploration/test/test_model.py \
  ros2_ws/src/lunar_drl_exploration/test/test_sac.py
```

首次 pytest 自动加载了无关 ROS launch_testing 插件，因训练 venv 缺 lark 在收集前退出。
随后只对这两个不需要 ROS launch 插件的单元测试文件关闭插件自动加载，测试全部执行通过。没有安装依赖或修改测试。
原始失败日志和通过日志均保留于忽略目录。

## 证据与未验证项

`training-output/critic-audit-20260918/` 保留脚本、逐项 JSON、运行日志、源码前后校验。
`results.json` 记录原型、原生扩展和相关 Python 源码的 SHA256，绑定本次证据。

NOT_RUN：正式连续位姿接入、新 Critic 集成、训练后 Q 排序、三组 SAC 消融、真实导航控制闭环、大尺度真值图端到端性能、Humble、Orin、DDS、实车。
没有证明策略必然不循环、必然耗尽探索或达到 99%。
