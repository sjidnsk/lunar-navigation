# Volume 3 月球极区 PPO 训练前就绪交接

状态：`pretraining-ready / blocked-on-capability`

## 边界与结论

Ubuntu 22.04 amd64 + ROS 2 Humble + RTX 4080 SUPER 上的训练前代码、短程验证和只读监督入口已经形成。当前证据只证明开发态前置链路可执行、可中断恢复，并证明正式入口在能力资料缺失或仅提供 test-only/proxy bundle 时会提前拒绝。

本交接没有启动 seed 4080 正式 rollout，没有消耗正式 24 小时 GPU 预算，也没有生成 ONNX、TensorRT engine、四文件候选、`model-package-ready` 或任何 AGX 标签。

## 相关提交与冻结合同

| 范围 | 提交 | 已冻结事实 |
| --- | --- | --- |
| 完整设计与执行计划 | `d0c127b`、`57a6262` | 月球极区真实数据、三平台共享 PPO、能力资料后置冻结 |
| Observation/Action V2 | `caedfb9`、`8a6b0c7`、`a6f2e65` | 七输入固定 shape、64 候选和连续 `theta` |
| 共享 policy/rollout | `de93c70`、`d446ef8` | 单一三平台网络和 development-smoke padding |
| 极区数据与地图链路 | `bfbd89c` 至 `5864348` | source lock、split、固定地图、危险物和确定性候选 |
| C++ v3 投影与环境边界 | `2d36778` 至 `c56a16e` | 同一 v3 bridge 的 traversability/plan、64 yaw bins、决策身份 |
| Reward V2/PPO/预算 | `b1cca75`、`ddd7b48` | 成功主导奖励、固定 PPO、30 分钟 latest checkpoint、显式预算扩展 |
| capability/checkpoint v3 门 | `536e371`、`d12eb02`、`bdc8553`、`b5bd229` | 三平台外部 closure、formal 前置拒绝、完整 run identity |
| 官方极区数据锁 | `4734ef9`、`a6c4648` | NASA DEM/count、JAXA 六站点 holdout、seed 4080 固定 split |

## 官方极区数据身份

仓库外 aggregate source lock 为 `16,707` bytes，文件 SHA-256 为
`8d422cd9ef478ca15e7e36831ea14e9ba09e9af72565adbf0625a9137b1acab0`。其三个官方源为：

| source id | bytes | SHA-256 |
| --- | ---: | --- |
| `NASA_LOLA_87S_DEM` | 3,465,285,714 | `417a85715406c346e2ecb2fc3abc93d3717121466e3d4950e1a6b977f207b881` |
| `NASA_LOLA_87S_COUNT` | 145,895,726 | `dab531d817b9e7cfddf8ac23ffde9ccfe737efe526e98763ca7ded7a2afcae04` |
| `JAXA_LUPEX_DATA_S1` | 76,134,049 | `4a2cbb1d9f6ed4a1abd804f9faeee45c6c1f847030b31ba7fb986f280e8ecd77` |

seed `4080` 的 split manifest 共 294 行：NASA 为 train 192、validation 48、test 48；
JAXA 的 `CR1`、`GR1`、`GR2`、`LP1`、`MP1`、`MP2` 六站点只进入 holdout，archive
inventory 共 18 个 GeoTIFF 成员。内部 `split_sha256` 为
`5d458081972e1ee767c5f91dd5cb42d519214a0111a6e28dfaa7283025ec99e2`；split manifest
文件为 `162,602` bytes，文件 SHA-256 为
`d51b9b824e5e9b2ebe67da70ab9a46c6d4a66489ec8ff063a766347058ba4a2e`。

这些锁只固定数据身份和划分，不解除正式训练的 capability gate；数据文件、锁文件和 split
manifest 均保持在仓库外。

`ObservationContractV2` 的网络输入顺序和 shape 为：

```text
prior_channels    float32 [B,4,256,256]
coverage_summary  float32 [B,3,256,256]
local_crop        float32 [B,4,32,32]
frontier_features float32 [B,64,12]
pose_features     float32 [B,6]
candidate_mask    bool    [B,64]
platform_context  float32 [B,3]
```

`platform_context` 仅是一项固定长度的平台类别 one-hot，不包含尚未确定的运动能力数值。候选不足 64 项时只做零填充并置 mask；全 false 时绕过策略，不制造当前位置候选。动作由 masked 64-way frontier 和所选候选条件下的连续 `theta` 组成。

`RewardWeightsV2` 的成功首次跨越奖励为 `+50`，大于覆盖与优先级的全部正向 dense shaping 上限；无成功终止为 `-10`，硬安全违规再扣 `-50`。冻结权重 SHA-256 为 `46f0400e934113cfb863c4e4b64174027eac667f8654d14b2e5d333657335bd4`。

checkpoint 使用 `lunar-ppo-checkpoint/v3`。`RunIdentity` 固定 `run_kind` 以及 data、split、generator、capability、reward、v3 六项 SHA-256；resume 逐项比较，checkpoint 与 run manifest 必须保持相同 identity 和 global step。development-smoke 的 checkpoint、proxy 评估和 test-only capability 永远不能晋级正式候选。

## 前置验证证据

验证日期为 2026-08-05，测试 artifact 均位于 pytest 临时目录或仓库外部 build/install/log 目录。

- 只读 watcher：`13 passed`。覆盖 600 秒默认节拍、`--once` 单次 JSON、pause 优先、能力 closure 完整校验、PID/cmdline 所有权、checkpoint 陈旧、finite manifest 指标、GPU 查询、磁盘阈值和 symlink 拒绝；watcher 不启动、重启、终止进程，不修复 manifest，不改预算或激活状态。
- CPU development smoke：合成 DEM/count/JAXA archive 经 aggregate source lock 和 288+6 split，生成程序月岩/月坑/no-go，构造 V2 observation/candidates，经共享 policy 选择动作；同一个 `PlannerBridge` 依次执行真实 C++ v3 traversability projection 和 plan。随后完成第一个 PPO update、checkpoint v3、restore 和第二个 update，并验证 manifest/checkpoint identity 与 global step 一致。
- formal 前置拒绝：公共 `train`、`resume`、`evaluate` 在 capability lock 完全缺失时均在 artifact/CUDA 前拒绝；完整三平台 `formal_eligible=false`、`test_only=true`、`proxy=true` bundle 也在 artifact/CUDA/worker 前拒绝。
- 完整非 CUDA model-contract/training：`566 passed, 1 skipped, 3 deselected`。
- RTX 4080 SUPER 有界 CUDA：`2 passed, 12 deselected`；包括 FP32 forward、完整 update 边界中断、latest checkpoint 和恢复后一项 update，总计不超过两个 update。
- ROS core/bridge：外置 Humble 构建目录执行 core 15 个 CTest 和 bridge 17 个 pytest，`80 tests, 0 errors, 0 failures, 0 skipped`。
- 仓库边界与 foundation：`repository boundaries: OK`，`13 passed`。
- Python 编译、UTF-8 读取和 `git diff --check` 通过；Git 未跟踪任何 `.tif/.tiff/.zip/.npy/.npz/.pt/.pth/.onnx/.engine` 训练或发布 artifact。

## 能力阻塞与唯一解锁顺序

尚未确定轮式、足式、飞跃式三份实际运动能力资料。下一阶段必须按以下顺序解锁：

1. 由外部平台项目提供三份明确版本的 platform/observation/URDF/mesh 资料和运动能力数值。
2. 生成 `lunar-training-capability-freeze/v1`，通过 schema、三平台精确集合、资源 closure、内容一致性和 SHA-256 校验。
3. 使用冻结能力重新生成平台化 traversability/candidate cache，并重新冻结 worker 数和 micro-batch。
4. 重新执行前置回归后，才允许启动一个正式 seed 4080 的可暂停训练；正式验收、ONNX/TensorRT 和 AGX 流程仍按后续卷独立推进。
