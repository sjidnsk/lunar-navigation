# Lunar Navigation Ubuntu 存储指纹补充设计

**状态：** 已批准，待实施

**日期：** 2026-08-02

**适用范围：** 当前 Ubuntu 22.04 amd64 训练主机环境指纹，以及后续使用同一工具采集的 AGX Orin 指纹

## 1. 背景

当前 `lunar-platform-fingerprint/v1` 已记录 OS、架构、内核、CPU、内存、ROS、工具链、GPU、驱动、CUDA、TensorRT 和 Jetson 专用字段，但没有记录存储。Ubuntu 交接目标明确要求核查并补全存储信息，因此现有最终指纹不足以单独证明该项完成。

2026-08-02 现场核查确认当前主机的三个运行相关文件系统分别为：

- 系统根目录 `/`：`/dev/nvme0n1p7`，ext4，总容量 `50,072,293,376` bytes；
- 仓库工作区 `/mnt/data/WS/lunar-navigation`：挂载于 `/mnt/data` 的 `/dev/nvme1n1p1`，ext4，总容量 `1,967,864,131,584` bytes；
- 仓库外验证输出目录：挂载于 `/home` 的 `/dev/nvme0n1p8`，ext4，总容量 `430,841,765,888` bytes。

已用和可用容量会随运行变化，以上数值只说明设计时观察到的现场拓扑；最终证据必须由工具重新采集，不得把这些数值写入平台 baseline 或用作 readiness 阈值。

## 2. 目标与非目标

### 2.1 目标

- 在环境指纹中同时记录系统根目录、仓库工作区和调用方指定输出目录所在文件系统。
- 每个记录保留请求路径、实际挂载点、设备来源、文件系统类型、总容量、已用容量和可用容量，容量统一使用 bytes。
- 使用稳定 locale 和可注入命令运行器，使解析、失败证据和单元测试保持确定性。
- 任一存储探测失败时保留其他成功结果以及失败命令、退出码和原始诊断。
- 在仓库外重新采集当前 Ubuntu 主机最终指纹，并把三类存储记录均可用列为本次交接完成门槛。

### 2.2 非目标

- 不枚举整机所有磁盘、分区、loop、临时 USB 设备、序列号或 SMART 数据。
- 不监控容量变化，不建立磁盘健康、性能或剩余空间告警。
- 不把磁盘型号、容量或剩余空间加入训练或 AGX readiness 判定。
- 不修改 RTX 4080 SUPER、ROS 2 Humble、AGX Orin、外部接口或卷一 Task 3–5 的既定职责边界。
- 不改写或伪造此前已经生成的 `lunar-platform-fingerprint/v1` 仓库外 artifact。

## 3. 版本与兼容性

新增必备顶层字段会改变完整指纹合同，因此新采集结果使用：

```text
lunar-platform-fingerprint/v2
```

`lunar-platform-baseline/v1` 是独立的期望平台合同，不随本次观测字段增加而升级。历史 v1 指纹继续作为当时的现场证据保留，但不能作为“已记录存储”的证明。实现后生成的最终 Ubuntu 指纹必须是 v2。

本次不新增存储 readiness 条件。`validate_fingerprint()` 继续只判断冻结平台身份和 compute/deploy 必需能力；存储探测失败不会被误报为 GPU 或 AGX 不就绪。Task 3–5 的最终验收命令另行断言 `storage.available=true`，从而区分“平台计算就绪”和“本次环境记录完整”。

## 4. 指纹合同

顶层新增 `storage`：

```json
{
  "storage": {
    "available": true,
    "targets": {
      "root": {
        "available": true,
        "requested_path": "/",
        "mount_point": "/",
        "source": "/dev/nvme0n1p7",
        "filesystem_type": "ext4",
        "total_bytes": 50072293376,
        "used_bytes": 34744004608,
        "available_bytes": 12751519744
      },
      "workspace": {
        "available": true,
        "requested_path": "/mnt/data/WS/lunar-navigation",
        "mount_point": "/mnt/data",
        "source": "/dev/nvme1n1p1",
        "filesystem_type": "ext4",
        "total_bytes": 1967864131584,
        "used_bytes": 512236937216,
        "available_bytes": 1355590529024
      },
      "output": {
        "available": true,
        "requested_path": "/home/kai/CodexDownloads/lunar_navigation/volume1-foundation/final-v2",
        "mount_point": "/home",
        "source": "/dev/nvme0n1p8",
        "filesystem_type": "ext4",
        "total_bytes": 430841765888,
        "used_bytes": 145589194752,
        "available_bytes": 263291826176
      }
    }
  }
}
```

示例容量是设计时快照，不是测试固定值。三个 role 即使位于同一挂载点也分别保留，因为它们表达不同的运行路径语义，不做去重。

每个成功记录必须满足：

- `requested_path` 和 `mount_point` 是规范化绝对路径；
- `source` 和 `filesystem_type` 是非空字符串；
- `total_bytes` 为正整数；
- `used_bytes` 和 `available_bytes` 为非负整数；
- 不要求 `used_bytes + available_bytes == total_bytes`，因为文件系统保留块会造成差值。

若某个 role 探测失败，该 role 使用统一失败结构并额外保留 `requested_path`：

```json
{
  "available": false,
  "requested_path": "/path",
  "command": "findmnt ...",
  "returncode": 1,
  "error": "原始 stderr、stdout 或退出码"
}
```

顶层 `storage.available` 仅在三个 role 均为 `available=true` 时为 true；失败时仍保留全部 role，不用一个总错误覆盖已取得的证据。

## 5. 探测与数据流

`tools/capture_environment.py` 为每个 role 运行：

```bash
findmnt --bytes --json \
  --output SOURCE,FSTYPE,SIZE,USED,AVAIL,TARGET \
  --target <requested-path>
```

既有 `run_command()` 继续显式设置 `LC_ALL=C` 和 `LANG=C`。解析器要求返回 `filesystems` 数组且精确包含一个记录；命令失败、JSON 畸形、零项、多项、缺字段、空字符串或非法容量均形成可审计失败对象。

路径来源固定为：

1. `root`：`/`；
2. `workspace`：CLI 的 `--workspace-root`，默认值为仓库根；
3. `output`：`--output` 的父目录。

CLI 必须先拒绝仓库内输出并校验 workspace，再创建仓库外输出父目录，随后规范化三个路径并采集指纹。`capture_environment()` 接收显式 workspace/output 路径参数，单元测试不依赖测试进程的当前目录。

## 6. 错误处理与边界

- 输出路径仍必须位于 Git 仓库外；不得为了存储探测放宽现有 artifact 边界。
- `--workspace-root` 必须解析为已存在的目录；不存在、不是目录或无法解析时，CLI 在任何指纹写入前以退出码 2 拒绝。
- 输出父目录创建失败时报告错误并退出，不产生虚假的存储或 readiness 结果。
- `findmnt` 缺失或失败时保存命令、退出码和诊断；不得使用计划值、`df` 文本或硬编码现场数值回填。
- 任一 role 失败不会阻止其他 role 和其他环境字段继续探测，也不会改变既有 readiness 规则。
- 当前 AGX 尚未进行现场采集；本设计只使未来 AGX 指纹包含同一结构，不把 Ubuntu 存储结果外推为 AGX 能力。

## 7. 测试与验收

实现遵循测试先行，至少覆盖：

1. `findmnt` 正常 JSON 被规范化为精确字段和整数容量；
2. 命令失败、JSON 畸形、零项、多项、缺字段和非法容量被拒绝并保留诊断；
3. 一个 role 失败时另外两个成功 role 不丢失，`storage.available=false`；
4. 三个 role 位于同一挂载点时仍分别保留；
5. 新指纹顶层键和 `schema_version` 精确匹配 v2；
6. 存储失败不改变训练或 AGX readiness 的既有判定；
7. CLI 拒绝无效 workspace 和仓库内 output，且不提前创建仓库内目录；
8. CLI 将 output 父目录传给探测并写出 UTF-8、LF、带尾换行的 JSON；
9. foundation 全量测试、仓库边界检查、三个 ROS 包仓库外重建、生成接口测试、来源检查和 colcon test 全部继续通过；
10. 最终仓库外 Ubuntu v2 指纹满足 `readiness.ready=true`、`storage.available=true`，并现场记录三个 role。

实施记录只报告本次新采集的实际值及 artifact 路径，不把变化容量写成冻结平台能力。原工作区 `AGENTS.md` 和 `.vscode/` 用户改动继续保持不受影响。
