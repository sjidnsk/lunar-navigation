# 月球极区正式训练环境资格报告

日期：2026-08-08

状态：`formal-training-ready / training-not-started`

## 结论

正式训练前的数据、场景、三平台能力、观测、规划、恢复和评估入口已经闭环。当前实现使用
通用多分辨率规则选出的 `4.0 m` 全局层处理 1024 m 训练窗，同时保留 `0.2 m` 局部规划、
传感器 reveal 和网络局部裁剪。该选择不是按“千米地图”名称写死，也没有把局部物理认证降到
4 m。

本机 full cache、正式传感器报告、非 proxy formal preflight 和 runtime calibration 均已通过。
校准冻结为 24 workers、micro-batch 4；run manifest 的 `global_step=0`，没有启动 seed 4080
训练，没有正式 checkpoint，也没有生成 ONNX、TensorRT 或模型发布候选。

## 唯一现行数据流

```text
NASA/JAXA source lock + split v2
  -> 冻结矢量场景和确定性障碍
  -> 256x256 @ 4.0 m 全局真值/三平台静态投影
  -> 按需 320x320 @ 0.2 m 局部瓦片
  -> 30 m/360 deg 真实 reveal
  -> observed-only 4 m 候选与 0.2 m 局部裁剪
  -> capability v2 + 当前 C++ v3 planner
  -> 宏步执行边界、覆盖奖励和下一观测
```

全局尺度因子为 `[1, 2, 4, 8, 16, 20]`，基础分辨率 `r0=0.2 m`。消费者选择满足目标单轴
格数 `T=256` 与硬资源上限的最细层。因此 1024 m 窗使用 L5
`256x256 @ 4.0 m`；较小地图仍可使用 0.2/0.4/0.8/1.6/3.2 m 中更细的层。局部图固定
`0.2 m`，不随全局层降采样。

## 正式场景与 cache

full cache 位于：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/360fe26/cache
```

资格事实：

- schema：`lunar-formal-training-cache/v1`；
- 物化场景：1734，其中 train 1536、validation 96、test 96、JAXA holdout 6；
- 文件：1736 个，共 458800547 bytes；
- `formal_eligible=true`，生产加载器以 `require_full=true` 重新打开通过；
- cache manifest 内部 SHA-256：
  `e8b2321507217fbb69a30ba7948d18a8d7309eb52ed13ef8e3ecbdedeca5d01c`；
- cache manifest 文件 SHA-256：
  `d047d83ae1e8e6ee507a9865a147f295c3c6e1c6302e0183be777cfd20f39f43`；
- 场景 manifest SHA-256：
  `0f09ea19d5a996c328388398f1791ff0bee3bad3ce30f3ac40ae53980f17db49`；
- source lock 文件 SHA-256：
  `8d422cd9ef478ca15e7e36831ea14e9ba09e9af72565adbf0625a9137b1acab0`；
- split v2 文件 SHA-256：
  `4434b342dddd184ac7ce2cd5b5c2e5e199ef28247f2602c7d9cfa5dfc51e1e2d`；
- split 内部 SHA-256：
  `6da68f54d7349142f787d0e82e3cc08b44770f99ae6deddb9f98b7cc5bfcbd18`；
- capability SHA-256：
  `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`；
- C++ v3 语义 SHA-256：
  `8fb5d43781ff372ad2398d91d646caebb1c2ca79580244add22902eeec4687cb`；
- 训练语义 SHA-256：
  `739ce3e1f6eaab4ee44a0136ff848f246d158a15ebb7f8b6c8b2c7ec1bca8cb5`。

cache 由提交 `360fe26` 生成；其身份只绑定数据、split、场景生成器、reward、capability、C++ v3
和训练语义。随后提交 `e7c0c3b` 只修正策略候选不得等于机器人当前全局栅格，不改变任何 cache
内容；正式 run manifest 另以完整 `source_commit` 绑定该运行时代码。

## 传感器性能

报告路径：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/e7c0c3b/sensor-performance.json
```

- 报告内部 SHA-256：
  `95ba584aa18b9f57ec087f295215ecc6c185dd0a3ad725ebb27ecbb525a9a1d4`；
- 文件 SHA-256：
  `c919f9beb2c9d255d1c599b134561f69c9f485f520009e3a8593a655d3f952ff`；
- 4 m 候选信息增益 p95：`0.132299 ms`，门限 `5 ms`；
- 0.2 m reveal p95：`1.381889 ms`，门限 `2 ms`；
- 24-worker 吞吐降幅：`6.899653%`，门限 `10%`；
- Release runner、当前 capability v2、训练语义和源码提交 `e7c0c3b` 校验通过。

## Formal preflight

报告路径：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/e7c0c3b/preflight/formal-preflight.json
```

- 报告内部 SHA-256：
  `14ab85bc7f977da962452bce663202320c1fcf49806e4d0a95c4326053ef9c1d`；
- 文件 SHA-256：
  `096603d60e3fac166511a078b2824a27cc9c93956e94c1d5be33eb73c5ac1427`；
- `proxy=false`、`training_started=false`；
- `cache_and_identity`、`three_platform_worker_construction`、`same_world`、
  `deterministic_request_and_planner`、`multiresolution_4m_global_0p2m_local`、
  `hopper_no_cumulative_fuel`、`update_boundary_resume`、`nonproxy_evaluation` 和
  `qualified_worker_configuration` 九项检查全部为 true；
- 18 与 24 worker 均通过进程级构造，preflight 推荐 24 workers、micro-batch 2。该 micro-batch
  只是短时 preflight 固定值，最终以完整 runtime calibration 的选择为准。

候选生成器曾允许当前位置栅格进入候选集，导致飞跃平台原地落点在下一宏步因毫米级高程差
进入数值不定状态。提交 `e7c0c3b` 在信息增益估算和规划之前排除当前全局栅格，并增加回归测试；
正式 preflight 随后通过连续飞跃决策与非 proxy 评估。

## Runtime calibration

校准 root：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/e7c0c3b/calibration
```

`run-manifest.json` 文件 SHA-256 为
`29db77e5d21b6d12842a8e2f3cb9ca69560bfb3b3fa7849a669320893a6113a1`。校准比较
18/24 workers 与 micro-batch 1/2/4，未出现 OOM、IPC failure 或 planner timeout；最终冻结：

```text
workers = 24
WHEELED/LEGGED/HOPPER = 8/8/8
micro_batch = 4
formal_seed = 4080
global_step = 0
```

校准消耗 `112.984881` GPU seconds，只属于训练前资格预算；`global_step=0` 证明没有执行 PPO
训练更新。

## 自动化验证

- 干净仓库外 Release 构建完成 7 个 ROS 包；`colcon test-result` 为 302 tests、0 errors、
  0 failures、0 skipped；
- `model_contract/tests` 与完整训练测试为 690 passed、1 skipped；
- `tests/foundation tests/ros tests/differential tests/performance` 为 179 passed、2 skipped；
- 能力冻结校验通过，摘要与 cache/报告一致；
- ROS 外部接口安装前缀与 schema v5 校验通过；
- 仓库边界检查通过，专项测试 14 passed；
- full cache、传感器报告、calibrated root 和四个 split 均由生产加载器重新打开通过；
- UTF-8 读取与 `git diff --check` 通过。

跳过项需要显式设备或资格开关，不计作通过项。

## 运行边界

下一步若另行授权正式训练，应直接对上述 calibrated root 运行公开 `train`。`train/resume/evaluate`
必须重新校验 full cache、传感器报告、capability、训练语义和源码身份；任一身份漂移都必须重新
执行 preflight 与 calibration。不得把旧 Volume 3 cache/checkpoint、proxy smoke 或 JAXA
holdout 接入训练 split。

本资格只闭合正式训练前准备，不代表训练、模型选择、ONNX/TensorRT、AGX 或实际平台验收完成。
