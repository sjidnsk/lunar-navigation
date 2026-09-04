# Incremental Navigation 复杂地形性能基准与优化建议

- 日期：2026-09-04
- 分支：`feat/incremental-complex-terrain-benchmarks`
- 基线提交：`0211add55e2bfeb0d3808e041000ca24948056d2`
- 环境：本机 x86_64、ROS 2 Jazzy、GCC 13.3、`RelWithDebInfo`
- 范围：只新增测试场景、离线基准和文档；未修改生产规划算法、ROS 接口、launch、配置或控制器

## 结论

简单稀疏障碍地图确实会掩盖当前瓶颈。新增复杂地形后，热点不是单一的 A*：

1. **首次或全量 fine traversability 派生**是 0.1 m、640 × 640 大片可行区域的首要热点；
   风险/未知带场景一次观测为约 `6.84 s`。现有增量 halo 路径有效：同图单格更新只重算
   `293` 个输出格、触及 `1301` 个高程格，约 `3.83 ms`；32 × 32 patch 更新重算
   `2432` 个输出格，约 `57.47 ms`。因此应优化全量派生，但不能破坏已有增量边界。
2. **轮式路径简化**在长蛇形通道中占主导：640 × 640 连续三次的局部规划 p50 为
   `1534.35 ms`，其中 postprocess p50 为 `1501.47 ms`，约占 `97.9%`。搜索本身不是该场景的
   主要问题。
3. **足式有向边和终点认证**在大范围搜索中占主导：640 × 640 台阶/缺口场景连续三次
   p50 为 `2688.80 ms`，每次固定记录 `4,999,407` 次 transition evaluation。
4. **全局路线缓存只覆盖同起点**。精确缓存和不影响路线的 revision 更新很便宜，但起点沿路线移动
   一个格仍重新搜索；窄通道场景的 moved-start 时间和冷启动基本相同。
5. 目标选择全窗扫描和已发布路径 revision 复检目前不是优先项：本机扫描 409,600 格约
   `2.4 ms`；320 × 320 路径复检约 `0.01–0.34 ms`。

建议先做三个局部、可回滚的实现优化：fine evaluator 稠密临时缓存，简化器折点预压缩，足式认证
廉价前置门控；随后再扩展全局路线的同一路径后缀缓存。暂不引入 D* Lite、LPA*、多层新规划器或
coordinator 重构。

## 新增复杂场景

场景由 `test/support/complex_terrain_scenarios.*` 以固定种子生成，测试尺寸最小 32 × 32，命令行
基准最大 640 × 640。

| 场景 | 主要压力 | 预期 |
| --- | --- | --- |
| `dense_rock_field` | 约 38% 固定种子岩石，并凿出弯曲窄通道 | 可达 |
| `alternating_wall_maze` | 交替上下开口的长墙，制造大量绕行和失败捷径 | 可达 |
| `narrow_passages_and_dead_ends` | 蛇形单通道和短死胡同，产生超长 raw path | 可达 |
| `risk_and_unknown_bands` | 高代价条带与 UNKNOWN 斑块，大片区域无法靠障碍提前退出 | 可达 |
| `enclosed_goal` | 可行目标外围闭环障碍 | `NO_ROUTE` / `NO_PATH` |
| `legged_step_gap_field` | fine 层保持 FREE，但高程突变必须由足式有向边认证 | 可达 |

正确性测试不使用易波动的绝对耗时断言，而是验证：场景确定性、端点状态、复杂度特征、可达场景
真实返回路径、封闭目标不发布路径、足式场景实际发生 elevation transition 认证。

## 基准职责与指标

新增可执行目标为：

```text
lunar_incremental_navigation_core_complex_terrain_benchmark
```

它只在 `BUILD_TESTING=ON` 时构建，不安装到生产 overlay，也不注册为带硬耗时门槛的 CTest。职责拆分为：

- `complex_terrain_scenarios.*`：只负责确定性输入和直接测试快照；
- `complex_terrain_benchmark.*`：只负责阶段编排、计时和指标采集；
- `complex_terrain_benchmark_main.cpp`：只负责 CLI 和 CSV 输出；
- 生产 planner、builder、coordinator：作为被测对象，不包含 benchmark 分支。

每次运行记录以下阶段：全量高程写入、全量 fine 派生、单格及局部 patch 增量派生、guidance 派生、
全局冷规划、精确缓存、off-route/on-route revision 更新、moved-start、全窗目标选择、wheel/legged
局部规划、coordinator 周期和路径 revision 复检。CSV 同时输出：

- `elapsed_ms`、`postprocess_ms`；
- expanded/generated/evaluated transitions/open peak；
- raw/final path 点数；
- fine 更新格、唯一高程缓存格、分配容量；
- target selector 的 tile/candidate/guidance 扫描量；
- 结果状态和 `cache_reused`。

`--deadline-ms` 只传给已有 global/local/coordinator 搜索截止时间；fine/guidance 派生和 target selection
当前没有同类 deadline。场景构造和直接 fixture snapshot 创建在计时区间外。`elevation_cells_examined`
是 evaluator 唯一 intrinsic cache entry 数，不包含同一格被 clearance 循环重复查询的次数。

## 可复现命令

```bash
export PROJECT_ROOT=/path/to/pure-planner-orin
export COMPLEX_BENCH_ROOT="$(mktemp -d /tmp/lunar-complex-bench.XXXXXX)"
export COMPLEX_BENCH_BIN="$COMPLEX_BENCH_ROOT/build/lunar_incremental_navigation_core_complex_terrain_benchmark"
source /opt/ros/jazzy/setup.bash

cmake -S \
  "$PROJECT_ROOT/ros2_ws/src/lunar_incremental_navigation_core" \
  -B "$COMPLEX_BENCH_ROOT/build" \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$COMPLEX_BENCH_ROOT/build" -j2

ctest --test-dir "$COMPLEX_BENCH_ROOT/build" \
  -R 'complex_terrain_(scenarios|benchmark)_test' \
  --output-on-failure -j1

"$COMPLEX_BENCH_BIN" \
  --size=320 --resolution=0.2 --iterations=3 \
  --deadline-ms=3000 --scenario=all > /tmp/complex-terrain-320.csv

"$COMPLEX_BENCH_BIN" \
  --size=640 --resolution=0.1 --iterations=3 \
  --deadline-ms=5000 --scenario=dead-ends \
  --no-legged --no-coordinator > /tmp/complex-terrain-dead-ends-640.csv
```

可选场景名为 `all|dense-rock|maze|dead-ends|risk-unknown|enclosed|legged-step-gap`；可用
`--no-legged` 和 `--no-coordinator` 隔离阶段。性能对比应在相同构建类型、CPU 条件、尺寸、分辨率、
种子和 deadline 下运行，至少记录原始样本并比较 p50/p95；不要把 CTest 变成主机 wall-clock 门槛。

## 320 × 320、0.2 m 全场景观测

下表是本机一次全矩阵观测，用于确认不同地形能暴露不同热点，不是生产 SLA：

| 场景 | full fine (ms) | 单格 incremental fine (ms) | global (ms) | wheel total / post (ms) | wheel raw → final | legged (ms) | transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense rock | 83.56 | 0.19 | 10.60 | 27.30 / 4.94 | 408 → 54 | 109.31 | 354,026 |
| alternating maze | 208.51 | 0.39 | 105.08 | 498.22 / 425.85 | 11,895 → 78 | 499.62 | 1,162,274 |
| narrow + dead ends | 195.11 | 0.28 | 17.30 | 176.11 / 168.56 | 12,590 → 78 | 35.69 | 153,453 |
| risk + UNKNOWN | 570.13 | 0.72 | 136.33 | 94.27 / 0.04 | 596 → 2 | 637.03 | 1,354,331 |
| enclosed goal | 567.74 | 0.74 | 163.25 | 93.97 / 0 | 0 → 0 | 696.56 | 1,411,403 |
| legged step/gap | 210.77 | 0.42 | 0.59 | 0.42 / 0.03 | 316 → 2 | 547.18 | 1,233,037 |

封闭目标的 global 为 `NO_ROUTE`，wheel/legged 均为 `NO_PATH`；表中的耗时不是失败。其余场景在 core
层返回 `PLAN_FOUND`。这仍不是 ROS Action 的正式成功证据，不能替代
`planning_outcome: 0 + reason_code: PLAN_FOUND + has_reference: true`。

## 640 × 640、0.1 m 定向压力结果

### 全量与增量 fine 派生

风险/未知场景在相同 0.1 m 分辨率下的尺寸扩展观测：

| 尺寸 | full fine (ms) | full 更新/唯一高程格 | 单格 incremental (ms) | 单格更新/唯一高程格 | global (ms) | wheel (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 × 160 | 312.35 | 25,600 / 25,600 | 3.86 | 293 / 1,301 | 28.25 | 21.81 |
| 320 × 320 | 1455.30 | 102,400 / 102,400 | 3.79 | 293 / 1,301 | 135.09 | 93.72 |
| 640 × 640 | 6843.96 | 409,600 / 409,600 | 3.83 | 293 / 1,301 | 600.29 | 401.82 |

最终加入 32 × 32 patch 阶段后，同一 640 场景观测为：更新 1024 个原始格，fine 重算 2432 个输出格、
触及 4556 个唯一高程格，耗时 `57.47 ms`；同次全量派生为 `6843.96 ms`。这证明当前成本能受 dirty
halo 约束，优化不得把局部 revision 更新退化成整图扫描。

### 长路径简化

640 × 640 窄通道/死胡同场景连续三次：

| 指标 | p50 或固定值 |
| --- | ---: |
| full fine | 788.94 ms |
| 单格 incremental fine | 0.71 ms |
| 32 × 32 patch incremental fine | 32.22 ms |
| global cold | 78.38 ms |
| global exact cache | 1.83 ms |
| global moved-start | 77.37 ms |
| wheel total | 1534.35 ms |
| wheel postprocess | 1501.47 ms |
| wheel raw → final | 50,790 → 158 |

三次 wheel total 为 `1576.07 / 1534.35 / 1519.31 ms`，postprocess 为
`1543.32 / 1501.47 / 1488.20 ms`，热点具有重复性。

### 足式边认证

640 × 640 台阶/缺口场景连续三次：

| 指标 | p50 或固定值 |
| --- | ---: |
| legged total | 2688.80 ms |
| legged postprocess | 52.97 ms |
| expanded / generated | 352,216 / 619,965 |
| evaluated transitions | 4,999,407 |
| legged raw → final | 49,211 → 5,071 |

三次 legged total 为 `2697.72 / 2624.56 / 2688.80 ms`。相同地图的 wheel p50 约 `0.77 ms`，说明
主要差异来自足式语义认证，而不是地图尺寸本身。

## 源码级瓶颈解释

### 1. Fine cell evaluator

`platform_elevation_physics.cpp` 中，每个输出格先扫描 hard footprint 邻域；若没有 blocked/UNKNOWN
提前返回，再扫描 `hard radius + preferred clearance` 的矩形邻域。`IntrinsicAt()` 虽避免重复计算
坡度/起伏，但缓存为 `std::map<GridIndex, ...>`，每次邻域访问仍做树查找。大片 FREE/risk 地形几乎都走
完整 clearance 扫描，所以比可早停的 dense rock 更慢。`fine_traversability_builder.cpp` 的全量分支还会
遍历所有 candidate tile cell；增量分支已经只遍历 influence halo。

### 2. Phase-aware path simplifier

`phase_aware_path_simplifier.cpp` 对每个 anchor 从整段末尾反向尝试 candidate，每次失败都重新做
supercover line-of-sight。长蛇形路径会产生大量“很远但被墙阻挡”的失败射线，因此 raw path 越长，
后处理增长越明显。

### 3. Legged directed-edge certification

`legged_local_planner.cpp` 在每个展开状态先尝试终点认证，再对八邻域逐边认证。终点即使远超任何运动基元
长度，也会进入 edge cache 和 `SupportsTranslation()` 的基元循环；邻居状态的 closed/现有 g 检查则在
认证之后。`SegmentCells()` 每条边创建 `std::vector` 并用 `std::find` 去重。上述常数成本乘以数十万
expanded state 后形成约 500 万次 transition evaluation。

### 4. Global route cache

`global_route_planner.cpp` 只有缓存起点、终点、profile 和 geometry 全部相同时才进入 reuse 检查。
因此 exact revision 和不影响 route influence 的新 revision 可复用，但机器人沿缓存路线移动一个 cell 后
必然重新运行搜索。搜索记录本身还是 `std::map<GridIndex, Record>`，在数十万展开时也有优化空间。

## 按收益和风险排序的优化方案

### P0-A：Fine 派生的数据布局优化

保持 `FineTraversabilityBuilder -> FineCellEvaluator -> snapshot` 职责和结果语义不变：

1. 在一次 Derive 内先将 intrinsic evaluation 放入按 tile 或局部 bounds 索引的稠密 scratch array，替换
   `std::map<GridIndex, ...>` 热路径；UNKNOWN、边界和 profile hash 合同不变。
2. 对固定 hard/clearance 半径预计算相对 cell offset，避免每个输出格重复 bounds、圆与 cell-area
   几何计算。
3. 若前两项仍不足，再评估“intrinsic classification + exact distance/inflation transform”两阶段实现；必须
   逐格对比旧实现，严格保持 cell-area 距离、地图边界、UNKNOWN 和 clearance cost。

验收：全部现有 builder 测试通过；六类场景逐格状态/代价等价；单格和 patch 的 updated halo 不扩大；
以 640 risk/UNKNOWN 的 full fine p50 为主指标，但不把本机毫秒值写成跨平台硬门槛。

### P0-B：简化前做线性折点压缩

只修改 `phase_aware_path_simplifier`：先在每个 phase 内线性移除共线、同方向的中间 raw vertex，再对折点
序列执行现有最远可见 greedy 和现有 LOS 安全认证。相邻折点仍沿原始已认证路径相连；任何 shortcut 仍须
通过原 `Allowed()`/supercover 检查。不要将简化逻辑移入 A* 或 coordinator。

验收：起终点与 phase 边界保持；每条输出段重新通过 LOS；取消/deadline 仍 fail closed；窄通道
50,790 点基准的 postprocess p50 显著下降，并记录最终点数变化。若路径质量下降，保留 raw-segment fallback，
不放宽安全条件。

### P0-C：足式认证的廉价前置门控

保持 `LeggedDirectedEdgeCache` 为唯一认证责任：

1. 构造 planner 时预计算可用平移基元的最大平面长度和合法 spin delta；终点距离超过能力上界时，在创建
   cache entry 和遍历基元前直接判定不可连接。
2. 邻居循环先构造 `next_state`，先跳过 closed；以几何距离作为认证代价下界，若不可能改善 g，再跳过
   expensive certificate。只有可能松弛的边才进入原完整认证。
3. 八邻域短边使用无堆分配的固定容量 supercover/visitor；长终点边仍走通用实现。不得删除方向性、
   step/gap、高程、start-prefix 或 terminal yaw 检查。

验收：所有 legged directed-edge/terminal/start-prefix 回归不变；六场景结果不变；640 step/gap 的
evaluated transitions 和 p50 同时下降。先看计数下降，再解释 wall-clock，避免只依赖偶然调度。

### P1：沿缓存路线的 moved-start 后缀复用

在 `GlobalRoutePlanner::Impl` 内扩展现有 cache：若 profile/geometry/goal 匹配，revision influence 仍有效，
且新 start cell 位于缓存 route 上，则从该 cell 返回安全后缀；不在路线、on-route revision 改变或指纹不匹配
时仍走现有完整搜索。该方案不改变 coordinator，也不需要引入 D* Lite。

验收：exact/off-route/on-route/moved-start 四阶段全部覆盖；moved-start 必须 `cache_reused=true` 且路径从
新实际起点开始；on-route 障碍仍禁止复用。之后再决定是否将 global 搜索的 `std::map` 换成 request-local
稠密/哈希记录。

### 暂不优化

- 目标选择 409,600 格扫描仅约 2.4 ms；
- 路径 revision 复检在现有复杂场景远低于 1 ms 量级；
- 不先并行化 fine tile，不先调整 heuristic weight，不放宽 UNKNOWN/障碍/足式高程语义；
- 不以降低地图分辨率冒充算法提速。0.2 m 和 0.1 m 应分别保留质量与性能证据。

## 优化实施前冻结对照

后续生产代码修改前，冻结 fresh build
`/tmp/lunar-complex-final.uyoh2c/build/lunar_incremental_navigation_core_complex_terrain_benchmark`
为 before 二进制，不再重建该目录。三个目标场景均使用 `640 × 640`、`0.1 m`、
`--iterations=3 --deadline-ms=600000` 串行复跑；原始 CSV 保存在仓库外
`/tmp/lunar-complex-perf-20260904/`。本轮优化采用同命令的 p50 作对照，至少提升 10% 且语义回归通过才保留。

| 优化目标/阶段 | before p50 (ms) | 关键语义或工作量 |
| --- | ---: | --- |
| risk/UNKNOWN full fine | 6675.13 | READY；409600 updated / 409600 examined |
| risk/UNKNOWN 单格 incremental fine | 3.829 | READY；293 / 1301 |
| risk/UNKNOWN 32×32 patch fine | 57.505 | READY；2432 / 4556 |
| 窄通道 wheel total / postprocess | 1495.68 / 1464.17 | PLAN_FOUND；50790 raw → 158 final |
| 窄通道 global moved-start | 78.548 | AVAILABLE；cache=false；63075 expanded |
| step/gap legged total / postprocess | 2666.73 / 52.911 | PLAN_FOUND；352216 expanded；4999407 transitions；49211 raw → 5071 final |

这些数值来自共享开发主机，作为同一时段、同一 workload 的相对基线，不是跨平台 SLA。额外执行的
`alternating_wall_maze` 数据不用于长路径简化的正式对照，因为该热点的既定代表场景是
`narrow_passages_and_dead_ends`。

## 逐项优化实测

### P0-A：fine intrinsic tile scratch cache（保留）

将一次 derive 内的 `std::map<GridIndex, IntrinsicTraversalEvaluation>` 改为懒分配的 tile array +
populated bitmap，并保留最近 tile 指针；`FineTraversabilityBuilder` 和 evaluator 职责、每格物理评估、
UNKNOWN/blocked/clearance 计算与增量 influence halo 均未改变。新增回归验证 520×16 稠密输入只使用
3 个 scratch tile，同时仍精确评估 8320 个唯一高程格。

同一冻结 before 二进制与当前 after 二进制在 640 risk/UNKNOWN 场景的三次结果为：

| 阶段 | before 三次 (ms) | after 三次 (ms) | p50 变化 | 工作量合同 |
| --- | --- | --- | ---: | --- |
| full fine | 6811.47 / 6675.13 / 6599.48 | 2246.62 / 2269.61 / 2252.26 | 6675.13 → 2252.26，-66.26%，2.96× | 409600 updated / 409600 examined，完全一致 |
| 单格 incremental fine | 3.829 / 3.781 / 3.844 | 2.195 / 2.192 / 2.213 | 3.829 → 2.195，-42.69%，1.75× | 293 / 1301，完全一致 |
| 32×32 patch fine | 57.505 / 56.623 / 57.932 | 42.204 / 42.272 / 42.839 | 57.505 → 42.272，-26.49%，1.36× | 2432 / 4556，完全一致 |

相关 fine builder 与 complex scenario CTest 为 2/2 通过，达到 10% 保留门槛。由于仅数据布局已经获得
显著收益，本轮不继续加入固定 stencil offset 或距离变换，避免扩大浮点边界与 cell-area 几何风险。

### P0-B：phase 内共线折点预压缩（保留）

在每个 phase 内先线性移除“共线且同方向”的中间 raw vertex，再对折点序列执行原有最远可见 greedy；
所有 shortcut 仍由原 supercover + `Allowed()` 检查认证，phase 边界、起终点、取消和 deadline 逻辑不变。
专门的 U 形阻挡回归保留安全转折，且同一 clock/cancel 探测接口的调用从优化前 986 次降至 224 次；
其中 201 次是为保证预处理仍可取消而逐 raw vertex 执行的线性检查。

640 窄通道/死胡同场景结果：

| 指标 | before 三次 (ms) | after 三次 (ms) | p50 变化 |
| --- | --- | --- | ---: |
| wheel total | 1501.67 / 1485.02 / 1495.68 | 42.389 / 42.249 / 42.513 | 1495.68 → 42.389，-97.17%，35.28× |
| wheel postprocess | 1469.24 / 1452.51 / 1464.17 | 8.347 / 8.420 / 8.468 | 1464.17 → 8.420，-99.42%，173.90× |

三次均为 `PLAN_FOUND`，expanded/generated 固定为 63075/63127，raw/final 固定为 50790/158；因此收益
来自消除重复 LOS candidate 扫描，不是减少搜索、降低分辨率或放宽安全条件。simplifier、wheel planner
和 complex scenario CTest 为 3/3 通过，达到保留门槛。

## Fresh 构建与回归

在仓库外新建 `/tmp/lunar-complex-final.uyoh2c/build`，从当前源码重新配置并构建 Jazzy
`RelWithDebInfo`。按包串行执行全部 core CTest：

```text
15/15 passed, 0 failed, 3.43 s
```

其中新增 `complex_terrain_scenarios_test` 和 `complex_terrain_benchmark_test` 分别通过；独立 benchmark
可执行文件也由该 fresh build 生成，并完成 320 全矩阵、640 窄通道三次、640 足式三次及 640
risk/UNKNOWN 定向复跑。两个新增 CTest 另以 `--repeat until-fail:3` 连续执行三次，均通过。
`git diff --check` 与新增文件尾随空白检查均无输出。

额外运行三个相关静态 Python 合同文件时为 `42 passed, 1 failed`。唯一失败来自既有
`test_isolation_contract.py` 仍断言 `config/` 只能含六个 YAML，但基线提交 `0211add` 已同时包含
`exploration_navigation.yaml` 与 `incremental_navigation_interfaces.yaml`。本分支相对该基线对 `config/`
和该测试的 diff 为空，因此未把旧合同修订混入本性能任务；该结果不计作本任务新增回归通过。

## 验证边界

本记录的数值来自共享开发主机，未固定 CPU 频率、未做实时调度或进程隔离，适合作为热点排序和前后
相对比较，不是 SLA。以下均为 `NOT_RUN`：

- ROS 2 Humble 原生/容器性能；
- Jetson AGX Orin 构建、延迟、内存、功耗和温度稳定性；
- DDS 跨机、真实 rosbag、真实 mapper/TF 连续 revision；
- ROS Action 正式规划成功、控制器和实车。

任何优化进入生产候选前，必须在相同复杂矩阵上重新记录 host 和 Orin p50/p95/max，并分别保留
首次全量、稀疏增量、局部 patch、moved-start 和 `NO_PATH` 的结果。
