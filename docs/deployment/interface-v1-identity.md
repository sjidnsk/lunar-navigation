# interface-v1 身份清单

本接口版本只用于先完成旧定义联调链，资格固定为 `integration_only`，不得描述为正式训练完成或设备发布合格。以后采用新覆盖语义、新平台能力或新路径算法时应建立独立 interface-v2，不把旧权重迁移成新语义权重。

## 不可混用的身份

| 项目 | 冻结值 |
|---|---|
| 源提交 | `fed9ea92c8ae55eb8423a008598935f515e801a3` |
| checkpoint step | `251` |
| checkpoint 文件 SHA-256 | `5c19113953bc91854abac88835a833e7bebe25ed6865cf0a09f36e828b359727` |
| checkpoint body SHA-256 | `001ad541eb4ba19930350c70cfd86d392c6443e73f0914bcde5f32dacdadb101` |
| Observation | `ObservationContractV3`，七输入 |
| Action | `ActionContractV2`，四输出 |
| 训练语义 | `lunar-training-semantics/sensor-30m-360-theta-mask-roi95-unbounded-common-start-subset/v5` |
| 能力冻结 SHA-256 | `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95` |
| ONNX | opset 17，动态 batch，CPU Runtime 联调 |

## 四文件模型包

当前外部候选目录：`~/CodexDownloads/lunar_navigation/interface_v1/model-fed9-step251-20260811`。目录只能包含下列四个文件；模型包不进入 Git。

| 文件 | SHA-256 |
|---|---|
| `policy.onnx` | `19a2d6db4633987d9ef70a73716794f742508bf691865bd27ee9a86d9b390dd3` |
| `golden_inputs.npz` | `3dbb359c789d9b0d9c40fa29bda341339ee426c5bc448609fd4f4cde8efe0203` |
| `golden_outputs.npz` | `8d9608377a34c09f924e0fd497cf67315e6b3b5928ec9d4896365d98b71e8c97` |
| `manifest.json` | `de85c37ea42e7f6045420a5d645f0f4ea98e7298a47143bd21ac8256fcde9550` |

## 平台文件身份

| 平台 | capability version | 完整 YAML SHA-256 |
|---|---|---|
| WHEELED | `wheeled-engineering-baseline-v1` | `3a4f87e310cf7721be71818f5e4abc6cc73e78644bcdb0f8465426e8bcef9360` |
| LEGGED | `quad48-approved-baseline-v1` | `1f25b2fc4796e50ef7e966473c03cf902b1073291e2d1ccb09983ec90f0b17f0` |
| HOPPER | `hopper-engineering-baseline-v1` | `1af41026d4c81500ce3351639d1fa7443f0c161b4de67f5da13d643b44dff841` |

节点配置时同时验证四文件模型清单、所选 YAML 的完整文件哈希和 capability version；任一不一致都会配置失败。飞跃式仍使用旧能力中的单跳可达性，但探索闭环不累计燃料库存。
