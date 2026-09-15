# 稀疏图 DRL 探索实现与验证

记录日期：2026-09-16。本实现位于
`/home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/drl-exploration-redesign`，
分支 `feat/drl-exploration-redesign`，从集成基线 `5c23c13cd0605955412ba6f470ee051d4fe8a9a2` 独立开发。
[设计](../superpowers/specs/2026-09-12-drl-exploration-redesign.md)、
[实施清单](../superpowers/plans/2026-09-15-drl-exploration-redesign.md)、
[复制执行的操作指令](../操作指令.md#drl-稀疏图探索独立-redesign-分支)分别记录契约、工作范围和操作。

实现已接通实测地图→任务图→CPU Actor→原生导航→公共控制器→轻量运动学平台→
沿程观测→经验回放→GPU SAC→保存/恢复。短程验证证明这条执行与学习链路能工作，
不证明未训练策略能达到 80/99、耗尽、收敛或泛化。最终独立 Task9 和全分支复审另行记录。

## 1. 已实现接口与不变约束

| 层 | 实现与所有权 |
| --- | --- |
| 地图 | 原生地图生产者独占高程/中心测量融合、固有 B 分类和足迹 M；`GetPolicyMap` 全量/增量瓦片、epoch/revision/处理时间与原生起点连接 |
| 观测/参考 | 有限确定性 beam fan，沿射线实际经过格及首个遮挡格的有效中心测量；真值可覆盖参考只用于训练/评估，Actor 不读取真值或分母 |
| 任务/图 | 同一累计 K 已知面积；先观测站位、再反向运动相关性；保留任务外道路、失联自由岛和 UNKNOWN 存储外部；19 维节点、真实位姿锚点、float64 目标 |
| Actor | 6 层、8 头、宽 128 的沿边稀疏注意力及一次当前位置全图读取；至多 20 位置×8 世界朝向，联合 softmax，部署联合 argmax，无 GRU/整图高程/N×N 注意力 |
| Learner | 双特权 Critic、目标网络；γ=1；奖励 ΔA/100−0.02ΔL/10−0.005ΔΘ/π−0.001；完整实际面积、XY 路程与绝对转角，无里程碑/完成奖金 |
| 更新 | batch64/micro16×4；Critic→Actor→温度→Polyak 各一次，梯度隔离；lr=1e−5、Polyak=.005、α初值5e−5/上限1e−4、目标熵.01log(N_valid) |
| 调度 | 默认8环境、CPU就绪批1–8、单GPU；worker/collector/learner数值线程1/2/4，DataLoader=0；warmup1024无额度债务，随后每4新增经验一次更新，每16次发布Actor |
| 回放/恢复 | 转移不可变且绑定原场景；场景每新回合传一次、引用计数释放；有界双向传输、接纳→计数→ACK、保留已完成消息；保存真实已发布Actor、优化器、全部RNG、额度/回放/课程，恢复新回合 |
| 运行 | 原生 `NavigateToPose`/`GetPolicyMap`，既有公共控制器唯一命令发布者；0.05s积分、2Hz观测、10m/90°、0.2m/s，前后退/旋转，无横移、端点瞬移、TCP或轮胎动力学 |
| 完成 | 仅实测任务耗尽 terminated且不自举；课程预算 truncated且用重置前真实后继自举；80/99仅评估，零增益不提前换图 |

任务多边形只限制覆盖/奖励；目标、道路与运动可在任务外，代价照常累计。
训练可从工程默认课程直接开始，无额外前沿基线/99%准入门槛。
默认课程累计计数边界 20k/60k、小中大混合 100/0/0→25/75/0→15/25/60，
地图尺度 40–80/80–150/100–300 m，预算 512/2048/8192；只在各槽新回合切换。
这些是可配置工程初值，不是学习质量承诺。

## 2. 源码、构建与证据来源

交付代码分两步固定：

- `6db75c0e8ef0cfa867dd97a4cfa2e7aca7d7722e`：基于已复审 `d1e7ee7`，修复成功 console 动作退出 1，以及旧 OccupancyGrid 发布者计数测试；主 163+67 转移探针使用此提交。
- `a5d06ebffc2a4a9ee963291ea104474ff0ccc746`：实测初始化继续现有最多四次扫描，直到原生连接和实测任务报告均可用；恢复记录精确 reset spec/stage；显示字段按回合隔离。后续三种子与短程 fresh/resume 使用该提交。
- `a35c74e`：任务外通行前沿改为有入口分量的精确原生光学见证，可跨越已测非可达足迹带；保留 first-pending、B遮挡和全局耗尽语义。最终四种子及集成验证使用此提交。
- `780292a`：静态配置白名单补入新增DRL YAML，只有测试改变，运行源码仍为a35c74e。
- 本文及操作手册是其后的文档提交，不改变执行代码。没有合并、推送或 PR。

独立安装根为 `/home/kai/.cache/lunar-drl-redesign/jazzy/install`；
Python 包位于 `lunar_drl_exploration/lib/python3.12/site-packages/lunar_drl_exploration/`，
原生扩展同级 `lunar_drl_terrain_native.cpython-312-x86_64-linux-gnu.so`。
Torch 使用 `/home/kai/.cache/lunar-drl-training/venv/bin/python`（Python3.12/Torch2.8/CUDA12.8），
rclpy 使用 `/opt/ros/jazzy`。`run.json` 记录实际 worker/native/collector/learner 路径与代码 revision；
外部缓存 `task9-{initial,resumed}-audit.json` 保存逐 Python 文件 SHA256、恢复摘要与模型指纹，
不是另一份历史模型。详细源文件/扩展身份见最终 provenance 记录。

本机为32逻辑CPU、32,775,319,552B内存、NVIDIA GeForce RTX5070Ti Laptop GPU
（12,820,938,752B），本轮为 x86_64 Jazzy/Linux 本机隔离 DDS。数值线程限制不等于进程总 DDS 线程数。
`training-output/` 已忽略；构建、测试、调查脚本与日志均在仓库外缓存。

## 3. 测试与已有闭环证据

所有 Python 测试在工作树根使用 `python -m pytest`，Torch测试使用上述venv；
本轮最后源码完成后的集成结果在最终补记中列出。console 的真实已安装脚本先复现导出文件成功但 exit1，
修复后实际成功导出 exit0 且 ActorPolicy 可加载，Python `main` 仍返回导出路径。
首次夹具把 str 与 Path 比较写错，先修正夹具再得到目标 RED；没有把该夹具错误当生产缺陷。

| 范围 | 实际结果与边界 |
| --- | --- |
| console/接口修正 | 39 PASS，21.70s；依赖闭包构建6包PASS，1.47s，无新编译警告；缓存 `task9-focused.log`/`task9-build.log` |
| 初始化/遥测修正 | RED：初始化2FAIL/1PASS，遥测3FAIL；GREEN：53PASS，35.21s；6包构建PASS；缓存 `task9-init-red.log`/`task9-telemetry-red.log`/`task9-corrections-focused.log`/`task9-corrections-build.log` |
| 原生 core/ROS（复用未变代码） | Task7串行 colcon test-result core261、ROS110，0错误/失败/跳过；数字包含CTest/gtest层级，不相加宣传独立场景数 |
| 公共控制器/旧接口 | root最终范围172PASS/5FAIL；一项旧发布者断言已修；4项缺少旧探索包，补齐依赖重测3PASS/1FAIL |
| 旧NO_PATH注入夹具 | 该1FAIL在不可变集成基线5c23c13同样复现：未观察到NO_PATH/候选替换，但有PLAN_FOUND/活动参考/GOAL_REACHED/多个目标；保留失败，不修改不相关导航断言 |

既有 Task7 native 实际测试覆盖相邻0.2m站位、反向平移、同位姿yaw、静态障碍足迹一致、
精确冻结目标、单命令发布者、最高0.2m/s，保留正式 `planning_outcome=0`、`PLAN_FOUND`、
`has_reference=true`/活动修订一致PathReference和最终 `GOAL_REACHED`。
详细夹具到位误差约.010–.011m；选中动作实际 L=2.117488m、Θ=5.355796rad，整体169沿程观测帧。
Task7后续实际40m洞穴全尺寸CPU Actor选择两动作，真实增益1.04/.52m²、RTF18.3071/16.5604，
951次物理步每次50,000,000ns、后续观测间隔10步。没有通过新版遥测日志单独的 GOAL_REACHED
倒推正式规划三元组，正式证据来自对应原生测试。

警告如实保留：Task7 core colcon提示所选工作空间未构建该包、使用专用已安装包环境；
其实际CTest工作目录/CMake源码目录均指向本分支独立缓存，退出0。
旧依赖补建有既有 `pure_plan_motion_server.cpp:1179` 的Time显式构造警告。
日志 `task7-core-final.log`、`task7-final-native-tests.log`、`root-final-native-controller-python.log`、
`legacy-validation-build-complete.log`、`root-final-legacy-live.log`、`root-baseline-no-path-probe.log`
均位于 `/home/kai/.cache/lunar-drl-redesign/`。基线对照的冗余二进制/源码归档已由其所有者移除，文本证据保留。

## 4. 实际八环境 GPU 采集、保存与恢复

首轮命令（提交6db75c0，输出未复用旧训练）：

```bash
scripts/drl/train.sh --probe --probe-warmup 64 --probe-extent 40 --probe-budget 8 \
  --max-transitions 160 --output-dir training-output/drl-exploration-redesign/task9-native-gpu-160 --domain-base 210
scripts/drl/train.sh --resume --probe --probe-warmup 64 --probe-extent 40 --probe-budget 8 \
  --max-transitions 64 --output-dir training-output/drl-exploration-redesign/task9-native-gpu-160 --domain-base 210
```

保持8环境、默认128/8/6网络、batch64/micro16、默认16更新发布、目标30和原始物理/奖励参数。
只显式覆盖40m、8决策预算和预热64。指标如下：

| 指标 | 首轮 | 恢复 |
| --- | --- | --- |
| 本次新增 / 累计转移 | 163 / 163（已完成超额3） | 67 / 230（已完成超额3） |
| 本次墙钟 | 38.55975s | 18.73422s |
| 新增采集率 | 4.227/s | 3.576/s |
| 累计完整GPU更新 | 23 | 40（本次17） |
| 保存精确额度 / 未结预约 | 7/4 / 0 | 3/2 / 0 |
| 实际保存已发布Actor | 16 | 32 |
| Adam四套步数 | 均23 | 均40 |
| 有限回放条数 | 163 | 230，前163条身份/动作/奖励/版本指纹一致 |

230条经验中，Actor版本0/16/32分别133/66/31条，实际执行了发布后选择的动作。
230次均记录GOAL_REACHED，0碰撞/geometry_failure，159次零增益；25次预算截断，0真实耗尽。
最大指令速度0.2m/s；累计动作XY路程268.74345m、绝对转角673.20227rad。
没有用重复单条经验伪装本探针的128条独立新增经验；SAC随机batch本身允许有放回抽样。

所有worker实测 `torch_loaded=false`/数值线程1，collector CPU/线程2/CUDA未初始化，
learner CUDA/线程4；run.json保存实际PID和import路径。检查恢复后的全部网络、目标网络、
Adam moments/steps、logα、schedule、回放序列化字节、Python/NumPy/TorchCPU/CUDA RNG、
replay/课程RNG及实际发布Actor/RNG与保存记录完全一致；恢复新增经验使用不相交的新episode UUID，
旧回放前缀的episode/action/reward/version指纹一致、学习权重改变、计数和优化器继续前进。检查脚本及输出是
`task9-checkpoint-audit.py`、`task9-{initial,resumed}-audit.{json,log}`；脚本初版误用
`Transition.action_index`，改为既有 `action` 后完成全部断言，这只是调查脚本错误。

动作RTF min/median/max = 1.56262 / 20.71816 / 28.08996，未达到目标30。
采样到的完整batch更新墙钟0.46248–0.65933s；最新回合初始化阶段1.12951–3.16814s，
此初始化计时从ROS建立开始，不包含Scene/Terrain/Reference构造，不能充当完整冷启动时间。
峰值采样 owned PSS=3,775,093,760B，最小采样MemAvailable=13,946,605,568B，
GPU峰值allocated=345,639,936B。PSS是当时明确owned PID集合，不叠加RSS；采样值不是硬上界。
多环境异步动作墙钟包含调度/争用，实测不足30不能归因于单一控制器；不存在跳积分步或扩FOV提速。

## 5. 初始输入失败与闭环发现

主探针有3次初始输入不可用恢复（E0/E6/E2），恢复段没有新增该类失败。
它们发生于reset后策略动作构造前，不是有参考的导航失败，也没有经验被丢弃来改善采集数。
两轮有序停止各排除2个真正未完成TaskCanceled动作；已完成消息全部ACK后保存，额度保留。
旧RECOVERY事件没有reset spec，原失败种子与槽位的对应只能由issued serial缺口推断：
`87019971607003137`、`87019971607003147`、`87019971607003160`，均moon40m。
保留全部原metrics/日志，不把推断称为直接记录。

随后对上述3个**确定种子**独立实际原生重置，直接重现 INPUT_UNAVAILABLE：
原生起点状态READY、可达R约4460–4991格，实测K约5597–6234格，但任务机会存在且尚无可用实测接口见证，
TaskAnalyzer报告available=false、exhausted=false。不是DDS丢连接；替换场景成功不能排除该确定性失败。
原实现看到任意start connection便结束扫描，已修正为原生连接和实测报告均可用才提前停止；
available+exhausted仍是合法零决策回合。最多四次quarter scan、原地实际运动与沿程观测、初始化代价均保留。

仅修正扫描停止条件后的三种子重测，在完整四次扫描后仍不可用；这一步失败证据保留。
对最终测量快照的直接分析进一步证明：三个场景的相关未知光学接口分别279/261/253个，
与R直接相邻的却均为0；它们全部有量程内原生可见见证，最近距离约4.472格（.894m）。
**根因是把观测K边界与足迹M/R边界等同，误要求光学接口必须与R四邻接。**
有效中心测量已超出足迹支撑边界，因而中间存在已测但非R的带状区域；这不是合法的地图拒绝条件。

TaskAnalyzer现对“相关且确实有R移动入口”的分量接口执行原生direct_witnesses，
使用同一first_pending_cells取得首个尚未测量中心；沿视线跨过已测非R带，保留B遮挡与量程支持。
没有入口的分量仍不能成为运动探索目标；直接看到任务的关系独立，耗尽仍由全局机会而非图候选数决定。
紧凑回归先复现相同INPUT_UNAVAILABLE，再验证可见带、透明但无入口、遮挡壁、完整已知和真实图动作；
现有穷举小图oracle继续通过。三个保存快照重新分析均available/nonexhausted，
最终首命中去重前沿279/256/251、动作24/24/40，分析与图构造.04247/.03546/.02918s。

加入后续RECOVERY直接记录的第四个moon40m种子`87019971607003142`，最终提交a35c74e上
四个种子都重新执行真实reset和一个全尺寸CPU Actor动作，均PLAN_FOUND/GOAL_REACHED，
目标/实际格一致、单命令发布者、速度≤.2、无geometry_failure。新算法没有增加扫描次数/传感量程、
更换起点或借助真值筛选。原失败日志、完整测量快照和逐扫描轨迹保留，以证明根因而非只记录替换成功。

原/扫描修正证据为 `task9-reset-diagnostic{,-fixed}.{py,json,log}`；最终根因与修复证据为
`root-outside-frontier-diagnosis.json`、`task9-reset-snapshot-<seed>.pkl`、
`task9-snapshot-check.{py,json,log}`、`task9-four-seed-native.{py,json,log}`。所有owned子进程关闭。
新RECOVERY日志记录当前完整reset spec和失败阶段，使今后无需从序号猜种子。

另一个发现是显示跨回合混用旧reason/gain/reference比例和新steps/pose；这属于异步UI缓存。
在既有RESET时清空每回合字段、SCENE时重新建记录，STATUS检测episode ID边界；回放中的
immutable Transition、累计K/reward计算与episode绑定未变。该显示修正不作为覆盖算法正确性的证明。

## 6. 大图成本、冻结导出和部署边界

复用root独立实际300m native+全尺寸GPU探针（提交b0882b3）：
TaskAnalyzer后续修正使这些reset/分析/动作RTF成为历史基线，不能代表最终代码；SAC/模型代码未变，更新成本仍可作为互补证据。

| 实际场景 | native reset | 初始观测节点/动作 | truth节点/参考格 | CPU Actor | 动作RTF | batch64/micro16更新 | GPU峰值allocated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| moon300m |15.6336s|246/40|1720/1,960,747|6.807ms|16.91|3.86708s|534,301,696B|
| cave300m |14.1862s|144/16|3129/95,467|3.856ms|22.41|2.38932s|921,470,976B|

truth数组分别16,097,541/1,220,293B。每次更新把**一条**实际转移重复64次，且关闭native环境后才学习；
这是实际大图CSR/初始观测成本，不是64条独立经验、稳态并发吞吐、地图后期成本或策略质量。
完整探针38.4248s、两次GOAL_REACHED、独占命令发布者、速度.2，四个owned子进程均退出；
`actual-300m-native-update-probe.{py,json,log}`保存源文件SHA256与精确执行记录。

参考池化是明确成本分量：独立实际300m参考格数的CPU micro16测试，moon中位.25542s、cave.009987s；
当前完整SAC每chunk执行3次truth pooling。该组件测试没有GPU/native闭环，且当时有并发构建，
不能从组件时间直接推断完整学习吞吐；大图节点数相同也不意味着参考CSR成本相同。
5999+5999节点合成压力测试经过激活重算后完整更新10.652s/2.097GiB，不能代替实际300m地图。

当前v5/finite_center_tip_prefix_v1无采样参考初始化：1000m月表148.7048s/峰1037.5MiB，
洞穴128.1334s/802.7MiB；主要成本是原生terrain构造（147.8263/126.1412s）。
原生参考全heading分支与最终±π边界修正的代码相同，方向kernel另行复验。
独立合成1km测量图19999节点/19998边，图与分析4.77128s、图2.137MiB、输入瓦片275MiB、
进程峰958.5MiB。可查明性/图数值验证与完成1km ROS探索任务严格分开；后者NOT_RUN。

Task8既有Actor导出与冻结评估使用全尺寸模型：moon/cave40m、seed20260915、每类2决策，
4次GOAL_REACHED、0碰撞/基础设施错误、两回合预算截断，最终参考覆盖.0991958696/.0347771103，
未达到80/99或耗尽；Actor-only加载通过。infer仅启动/停止，退出0；未在该次启动探针实际发送导航任务。
Task7的实测输入生命周期测试提供推理动作边界证据，两类证据不混淆。

以下仍为 **NOT_RUN**：正式训练收敛、80/99质量/耗尽成功率、跨地图/量程/FOV泛化、
完成1km ROS探索、真实课题三传感器/rosbag、Humble、Jetson AGX Orin、跨主机DDS与实车。
部署策略只选探索目标和朝向，复用现有地图/导航/公共控制器；运行时没有参考分母便不报告虚构覆盖率。
轻量SE2积分排除3D表面路程、roll/pitch、轮胎/侧滑动力学，Jazzy模拟观测不等于实物传感器保真。

## 7. 实施裁决：理由与代价

下表归纳实现过程中全部技术裁决；人员调度决定不属于算法或运行契约。

| 决定 | 理由 | 代价/后续边界 |
| --- | --- | --- |
| 有效中心地形测量、原生阈值/足迹独占 | 解决3×3分类支撑与首遮挡观测矛盾，不发布隐藏邻居高程 | 仿真测量近似需真实传感器复验；height-only实物输入继续可用 |
| 先求可能观测站位，再反向运动相关性并保留见证 | 避免窄视线丢失和假耗尽，分清B视线/M运动 | 相关性/图成本增加；若漏路必须恢复 |
| lr1e−5、Polyak.005、20k/60k课程及混合作为可配置初值 | 补齐实现参数，不增加训练准入条件 | 需冻结评估后调整，不保证最优 |
| 只把仿真观测原点量化到native格中心，plant/reference共用M扫掠 | 消除格角观测与参考分母偏差；连续yaw/控制锚点保留 | 损失亚格观测精度；首版平移外参0，非零外参需扩展参考 |
| 同高程部分地形统计保留更强已测下界；完整统计可替换，变高程失效 | 不丢新阻塞证据，避免缺支撑导致UNKNOWN退化 | 动态/不一致源可能保守滞留，需场景融合验证 |
| 只读导出已有有界起点prefix连接，输入不可用与耗尽分开 | 冷启动可能无法锚定R，旧基线缺查询接口 | 依赖原生连接正确性，不另造全图规划器 |
| 图压缩保留物理通道/绕障替代路线和见证 | 空场人工规则格小环不是独立物理通路 | 具体替代路线遗漏需重建和泛化复验 |
| 有界dirty字段历史和精确accepted revision时间；缺历史才全量 | 让native地图真实增量传输及测量新鲜度一致 | 小型元数据日志；客户端滞后回退全量 |
| 仿真暂存已命中未确认中心测量直到native消费 | 加速GridMap最新值合并不能丢真实观测 | 有界传输状态；面积仍来自应用后的native K |
| 可执行目标float64、网络feature float32 | 实际位姿不能经float32往返而改变原地目标 | 微量额外存储，无动作语义改变 |
| 参考只纳入有限且有效分类B非UNKNOWN中心 | 可见但支撑不足的测量不能计入可达分母 | 参考版本变化；旧模型/经验不能混用 |
| 实际路程/速度采用SE2平面运动 | 与既有Pose/nav/controller及批准轻平台一致 | 不证明3D里程、roll/pitch、轮胎动态 |
| 有限确定性beam fan记录每个实际经过格及首遮挡同时命中格 | endpoint-only会隐藏仍投影阴影的遮挡物；FOV由beam角决定 | 共享native参考/utility重验和版本变化，无ring-only捷径 |
| DRL导航/控制器统一.05m到位，传统默认不改 | .3m到位区域可吞掉.2m观测格的独特站位 | 更严格跟踪和末端时间；需真实闭环精度证据 |
| PrivilegedScene身份绑定reference_id | 同地形不同任务可有不同CSR，不能按terrain_id混淆监督 | 小型身份/回放修正，不设额外昂贵准入检查 |
| 梯度路径逐层激活重算、独立Q backward后各一次optimizer | 5999节点micro16原路径实测11.10GiB OOM | 额外forward换内存；全模型/effective64不缩小，Actor-only不重算 |
| 兼容调整microbatch/lr/Polyak且恢复Adam，saved logα优先 | 不应因不改目标的性能调参丢失训练历史 | 仍拒绝不兼容学习/奖励/观测语义，无迁移框架 |
| 共享实测累计K历史覆盖TaskAnalyzer/DecisionCore/训练/推理 | 当前分类36→35→36不能重复奖励旧格 | 随观测域增长的增量位图；当前B/M/wire不改 |
| RTF步预算计入计算，只睡剩余时间、不追赶漏步 | 完整额外sleep让即使低负载也达不到目标倍率 | CPU负荷/异步顺序可能变化；物理dt/速度不改 |
| 光学接口使用原生R见证跨越已测非R带，候选分量仍须有移动入口 | 实测任务外起点有数百可见接口却因四邻接假设被拒绝 | 额外有界光学查询；保留遮挡/首未知语义并复测完整/混合160m成本 |
| 包__init__惰性导出且保留公共API | 避免CLI/worker数值线程设置前过早导入NumPy | 延迟导入解析，需公开API与import guard回归 |

本轮额外裁决：初始化依赖已有实测报告可用性，沿用四次扫描上限；
恢复日志必须带失败输入；跨回合显示只保留同episode数据。这些均来自实际失败证据，
修复了确定性有效起点的错误拒绝，未把它改写为“成功耗尽”。


## 8. 最终源码复验与保留产物

提交 `a35c74e85b40caab78ca91b093945178cbf0cfa6` 的TaskAnalyzer修复通过
44项任务分析/图/可查明性测试（3.19s），完整受影响依赖闭包6包构建通过。
原有160m洞穴夹具只复验一次：完整理想观测available/exhausted、0前沿，分析.078409s；
只缺一个墙面测量时available/nonexhausted、1前沿，分析5.340046s（既有基线5.428351s）。
两者原生R与truthR相同、已知面积分别1240.60/1240.56m²；成本不是实车或2Hz保证。
日志 `task9-{closure,mixed}-phase.log`，新profile使用专用名字，没有覆盖旧profile。

最终提交再执行有界8环境全尺寸GPU fresh/resume：

```bash
scripts/drl/train.sh --probe --probe-warmup 64 --probe-extent 40 --probe-budget 8 \
  --max-transitions 80 --output-dir training-output/drl-exploration-redesign/task9-final-acceptance --domain-base 210
scripts/drl/train.sh --resume --probe --probe-warmup 64 --probe-extent 40 --probe-budget 8 \
  --max-transitions 16 --output-dir training-output/drl-exploration-redesign/task9-final-acceptance --domain-base 210
```

首轮84新增、3次GPU更新、26.256674s、额度2；恢复19新增、累计103条/8次GPU更新、
12.739232s、额度7/4；两次未结预约均0、无基础设施恢复，分别排除2/4个停止时未完成动作。
主模型/物理/奖励参数不变。最终103条均GOAL_REACHED，7次预算截断、0耗尽、80次零增益，最大指令.2m/s。
最终动作RTF min/median/max=.079237/17.812964/25.500038，仍未达到30；最小值来自一个
L=0、Θ=0的已到位动作，.25仿真秒/3.15508墙钟秒，控制/消息握手与调度成本占主导。
采样完整GPU更新.55947–.87419s，GPU峰allocated362,966,528B，采样PSS峰3,597,168,640B，
最小MemAvailable14,182,756,352B（`task9-final-metrics-summary.json`）。这些短期样本不代表长期上界或均衡吞吐。
最终短探针尚未到16次发布，保存Actor0；
真正发布16/32后的实际选择证据来自第4节主探针，并明确属于较早任务分析源码。
最终每个checkpoint再次验证完整网络/Adam/温度、所有RNG、精确schedule、回放恢复字节、
实际发布Actor/RNG一致；恢复后新episode ID与旧回放不交、计数/Adam3→8、权重继续变化，
前84条episode/action/reward/version指纹保留。缓存
`task9-final-checkpoint-audit.py`、`task9-final-{initial,resumed}-audit.{json,log}`保存检查细节。

最终综合测试：venv完整DRL包 **219 PASS、2个显式native opt-in SKIP，50.67s，无警告**
（`task9-final-package.log`）；最终实际四种子闭环单独执行，不用较低层测试代替。
静态消息/接口/隔离/launch契约首次67PASS/1FAIL，原因是旧配置文件白名单遗漏新增的
`drl_exploration.yaml`；补入合法配置后 **68 PASS，.12s**（`task9-final-source-contracts-green.log`）。
该测试修正和文档提交不改运行代码，不重复原生构建/探针。最后UTF-8、Python AST、shell语法、
文档围栏/本地链接、`git diff --check`通过。复用未改动native/controller历史测试，不再次制造重复测试总数。
本轮没有重新跑300m/1km完整初始化：TaskAnalyzer相关历史耗时已注明版本；
最终测量修复的真实性由四个实际失败种子闭环与160m成本复验覆盖。

保留项目产物（默认均已忽略）：

| 目录/文件 | 字节 | 用途 |
| --- | --- | --- |
| task9-native-gpu-160/resume.pt |原103271078，已移除|冗余中间模型；SHA256与恢复审计已留存|
| task9-native-gpu-160/metrics.jsonl |751435|含原失败/Actor16和32执行证据|
| task9-native-gpu-160/run.json |42811|主恢复段实际参数、路径、进程和计数|
| task9-final-acceptance/resume.pt |92687910|最终源码唯一可恢复状态|
| task9-final-acceptance/metrics.jsonl |342898|最终fresh/resume实测记录|
| task9-final-acceptance/run.json |47212|最终源码运行身份|

中间 `task9-final-init-probe` 的39条遥测/run文本保留，其44,730,287B的resume.pt与上表主探针冗余模型
在记录精确路径/字节/SHA256后已逐文件移除，共148,001,365B；
只保留一份Task9最终源码可恢复模型。`task9-redundant-model-{inventory,cleanup}.json`记录操作。
Task8及旧训练产物、共享venv/build均未清理，不递归删除目录。保存的三个失败测量快照合计约4.13MiB在外部缓存，
属于可复现失败输入，不是默认训练输出。正式默认输出根目录没有开始无界训练；以上全部为显式探针子目录。
最终 `task9-source-provenance.json` 记录当前安装源码与扩展SHA256、
模型以外证据路径和精确字节清单。Humble/Orin/field与策略质量边界仍如第6节。
