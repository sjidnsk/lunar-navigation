# fed9 旧定义 interface-v1 完整链路设计

状态：已批准，进入实施

日期：2026-08-10

## 1. 目标

先完成一版能够与其他 ROS 2 Humble 项目对接的探索决策接口。该版本必须让旧检查点只运行在训练它时使用的旧定义、旧候选生成、旧覆盖率语义和旧规划算法上，形成可重复演示和联调的完整闭环。当前正在演进的 platform-coverable/v7 定义不进入本版本；它将在 interface-v2 中与新模型一起整体升级。

interface-v1 的用途是接口联调，不是正式策略发布。它必须真实标记为 `integration_only`，不得伪造正式训练完成、发布评估通过或 AGX/TensorRT 验收结论。

## 2. 冻结身份

interface-v1 以 Git 提交 `fed9ea92c8ae55eb8423a008598935f515e801a3` 为唯一源码基线，并冻结以下检查点身份：

- 检查点文件：`reward-v3-69453ec/calibration/checkpoints/latest.pt`
- 检查点文件 SHA-256：`5c19113953bc91854abac88835a833e7bebe25ed6865cf0a09f36e828b359727`
- 检查点正文 SHA-256：`001ad541eb4ba19930350c70cfd86d392c6443e73f0914bcde5f32dacdadb101`
- checkpoint schema：`lunar-ppo-checkpoint/v6`
- checkpoint step：`251`
- checkpoint phase：`joint`
- observation contract：`lunar-observation-contract/v3`
- action contract：`lunar-action-contract/v2`
- training semantics：`lunar-training-semantics/sensor-30m-360-theta-mask-roi95-unbounded-common-start-subset/v5`
- training semantics SHA-256：`93797e459cfdbeedf59ca68642122c9e3546730cd1dd4c9d5ea78bb495786e44`
- capability freeze SHA-256：`60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`
- reward SHA-256：`ec2ba24c8afd2a8c4416d4ece079155bcf33f326ed16453b0aded6e1bd77eeec`
- V3 environment SHA-256：`8fb5d43781ff372ad2398d91d646caebb1c2ca79580244add22902eeec4687cb`
- run config SHA-256：`1891988a68c5fdc08806c4918e99b159bbdf1fba867ed51dbf5fb95d0a9da891`

导出时加载完整 `model_state`，包括旧 value head。禁止将其解释为 warm start，禁止恢复 optimizer/RNG 后继续正式训练，也禁止把旧权重重新标注为新定义模型。

## 3. 版本拓扑

源码使用独立分支 `feature/interface-v1-fed9`，从 `fed9` 创建。最终通过验证后冻结独立 tag，不把 v7 工作树的候选、覆盖、平台能力或训练语义合入本分支。

允许从后续部署工作中移植与语义无关的通用设施，例如固定路径 YAML、外部消息适配器、四文件包校验器和 ONNX Runtime 包装器；每项移植必须在 fed9 上重新测试。任何改变网络输入数值、候选集合、动作解释或规划结果的实现均不得作为“通用设施”移植。

## 4. 完整数据流

完整闭环固定为：

1. 外部系统发布全局/局部 GridMap、Odometry、TF、定位状态、探索任务和执行反馈。
2. `lunar_external_adapter` 按固定接口 YAML 做 Topic 映射、字段校验和必要转换；外部项目继续拥有数据。
3. interface-v1 决策节点在同一时间边界锁定任务版本、地图快照和机器人状态，构造不可变 observation identity。
4. 使用 fed9 的 `ObservationBuilderV2`、`CandidateBuilderV2`、30 m/360° 传感器语义、ROI95 覆盖语义和 C++ v3 投影，生成 Observation V3 的七输入。
5. `policy.onnx` 通过 Ubuntu ONNX Runtime 运行，输出 Action V2 的四输出。选择规则为有效候选中的确定性最大 logit；地面平台使用被选候选的 `theta_mu`，飞跃式按 fed9 的 theta mask 语义处理。
6. 被选候选转换成 `PlanMotion` goal，并连同同一 observation identity 对应的 mission revision 与时间戳发送到 `/plan_motion`。
7. fed9 规划器返回 `MotionReference`。轮式和足式按滚动局部参考执行；飞跃式完成单跳后进入 `LANDED_HOLD`，收到匹配的落地反馈和新状态后才允许第二次决策。
8. 执行系统发布 `/execution/motion_feedback`；决策节点只在反馈、地图和定位共同形成新的决策边界后构造下一观测，防止旧参考、旧地图或旧平台 episode 状态串入下一步。

所有动作必须来自同一份候选快照。若 identity、任务 revision、地图时效、TF、候选 mask 或模型输出不一致，则拒绝发起规划并进入可诊断 HOLD，不允许用默认全零输入、旧候选或随机候选继续。

## 5. ROS 接口与可替换配置

每个平台保留一个完整能力 YAML：

- `wheeled.yaml`
- `legged.yaml`
- `hopper.yaml`

部署时选择一个文件复制到固定路径 `/etc/lunar_navigation/platform_profile.yaml`，重启后生效。代码不同时拼接多份局部配置，避免运行时能力身份不明确。

外部 ROS 接口使用 `/etc/lunar_navigation/interface_profile.yaml`。默认配置保持以下稳定输入：

- `/environment/map_global`
- `/environment/map_local`
- `/localization/odometry`
- `/tf`
- `/localization/status`
- `/mission/exploration_task`
- `/execution/motion_feedback`

接口 profile 负责 Topic 名、消息类型、frame 和显式适配方式，源码中的业务算法不硬编码其他项目的 Topic。每个内部通道的 QoS 由节点按接收合同固定，不能由可替换 YAML 改写；外部 provider 必须匹配该通道 QoS。未来上游定义正式同名消息包时只能原子替换 schema provider，不允许两个同名包共存。

## 6. 模型包和运行时

模型目录仍严格只有四个物理文件：

- `policy.onnx`
- `manifest.json`
- `golden_inputs.npz`
- `golden_outputs.npz`

interface-v1 使用独立 manifest schema，显式包含：

- `qualification: integration_only`
- 源提交、checkpoint step 和全部冻结 SHA-256
- Observation V3/Action V2 的精确张量名称、dtype 与形状
- 三个平台能力文件的版本和内容 SHA-256
- 四文件的内容 SHA-256
- ONNX opset、数值容差和导出环境

该 schema 不含也不接受虚假的 `release_evaluation.passed=true`。正式发布包 schema 与 interface-v1 schema 互不冒充，加载器必须按 qualification 分流并 fail closed。

当前 Ubuntu 主机使用 ONNX Runtime 完成导出等价性和完整链路。TensorRT engine 不是四文件模型包成员，也不是本阶段完成条件；AGX 上生成 engine 和性能验收属于后续设备阶段。

## 7. 组件边界

实现划分为以下组件：

- `lunar_model_contract`：interface-v1 manifest、四文件目录验证、hash 和 golden tensor 等价验证。
- `lunar_policy_training` 的 fed9 冻结代码：只用于 checkpoint 读取和 ONNX 导出，不进入运行时优化器逻辑。
- 运行时中立的旧语义输入构造层：封装 fed9 observation/candidate/动作解释，不暴露训练循环、reward 或 PPO 更新。
- `lunar_exploration_policy_ros`：ROS 生命周期节点，负责快照、ONNX Runtime 推理、PlanMotion action、反馈状态机和诊断。
- `lunar_external_adapter`：外部 Topic/profile 到仓内稳定输入的适配。
- `lunar_planner_ros`：保持 fed9 规划与 `MotionReference` 行为。

运行时节点只能读取部署模型和 YAML，不读取 `.pt`、训练 cache、optimizer 或训练 job state。

## 8. 状态机与失败语义

决策节点最少包含 `WAITING_INPUTS`、`READY`、`INFERENCING`、`PLANNING`、`EXECUTING`、`LANDED_HOLD` 和 `HOLD_ERROR`。

- 轮式/足式：执行当前局部参考期间可准备下一决策输入，但只有匹配 feedback 与新状态满足边界条件时才提交下一请求。
- 飞跃式：已提交 hop 后禁止替换动作；只接受匹配 platform、plan_id、segment_id 和递增 sequence 的落地反馈。落地后清除旧候选、旧 plan 和旧 episode 状态，重新构造第二次观测。
- 没有有效候选：发布可诊断 HOLD，不调用模型的 all-false row。
- 模型包、YAML、地图层、TF、时间戳、revision 或输出验证失败：启动或当前决策失败关闭，不降级为猜测值。
- PlanMotion 的不可行结果：保留任务，等待新的地图/状态决策边界；不得把一次不可行当成 episode 永久结束。

## 9. 验证策略

采用测试先行，每个测试必须验证可观察行为：

1. identity：修改任一 checkpoint、源码、能力、合同或模型文件都会被拒绝。
2. export：同一 golden input 上 PyTorch 与 ONNX Runtime 四输出在冻结容差内一致；三平台均覆盖。
3. observation：受控地图上七输入、64 候选 mask、目标坐标和 theta 的手算结果与 fed9 一致。
4. ROS adapter：默认 profile 能消费完整外部消息；缺字段、错误 frame/QoS 或旧 revision 被拒绝。
5. wheel：连续两个决策边界产生两次有效 PlanMotion 请求，第二次使用更新后的位姿和地图。
6. legged：同上，并保留 fed9 足式状态/净空语义。
7. hopper：第一次规划、落地反馈、重新建 episode 状态、第二次选点和第二次规划完整通过；第二次请求不携带第一次候选或 plan identity。
8. negative：all-false mask、陈旧地图、TF 缺失、模型 hash 不符、foreign/late feedback、planner 不可行均进入正确 HOLD，且不会误发动作。
9. repository/build：边界检查、Python 测试、ROS 2 Humble Release 构建和目标包测试通过，构建产物写在仓库外。

完整链路测试既包含进程内确定性回归，也包含 ROS 进程级 launch 测试；不以 mock 的存在代替真实模型、真实消息转换或真实 action 结果。

## 10. 完成标准

interface-v1 只有同时满足以下条件才算完成：

- 独立分支只包含 fed9 旧语义与明确审阅过的通用部署设施；
- 四文件模型包从 step 251 生成并通过三平台 golden 等价验证；
- 三个平台固定路径 YAML 与接口 YAML 可直接替换、重启生效；
- Ubuntu ONNX Runtime 完整链路的三平台正例和关键反例通过；
- 飞跃式连续两次决策回归通过，轮式/足式连续滚动接口回归通过；
- ROS 2 Humble Release 构建、仓库边界检查和相关测试通过；
- 生成 interface-v1 使用说明、模型身份清单和可复现命令；
- 冻结独立 interface-v1 tag，但不宣称正式训练或 AGX/TensorRT 资格。

## 11. 非目标

- 不继续训练 step 251 检查点。
- 不把 platform-coverable/v7 的覆盖定义、候选算法或平台能力反向套到旧权重。
- 不修正旧策略质量并将其冒充新模型。
- 不在本阶段完成正式发布门、AGX TensorRT 性能或真实设备验收。
- 不改变外部项目对传感器、地图、执行器和任务数据的所有权。

interface-v2 将从当前集成主线重新冻结定义、训练模型、导出四文件包并走正式资格链；它与 interface-v1 的模型和代码版本整体切换，不只替换权重文件。
