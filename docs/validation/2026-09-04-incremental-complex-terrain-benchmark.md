# Incremental Navigation 复杂地形性能基准与优化结果

- 日期：2026-09-04 至 2026-09-05
- 分支：`feat/incremental-complex-terrain-benchmarks`
- 基线提交：`0211add55e2bfeb0d3808e041000ca24948056d2`
- 环境：本机 x86_64、ROS 2 Jazzy、GCC 13.3、`RelWithDebInfo`
- 范围：新增复杂地形测试与离线基准，并在既有 core 模块内实施四项局部优化；未修改 ROS 接口、
  launch、配置、coordinator 职责或控制器

## 结论

简单稀疏障碍地图确实掩盖了多个热点，而不是单一的 A*。在冻结 before 二进制和最终 fresh after
二进制上，以相同 `640 × 640`、`0.1 m`、三次串行运行的 p50 比较，四项优化均超过 10% 保留门槛：

| 优化 | 代表阶段 before → after p50 | 变化 | 决策 |
| --- | ---: | ---: | --- |
| fine tile scratch cache | full derive `6675.13 → 2213.65 ms` | `-66.84%`，`3.02×` | 保留 |
| phase 折点预压缩 + 逐段 LOS | wheel `1495.68 → 43.02 ms` | `-97.12%`，`34.77×` | 保留 |
| 足式认证前门控 | legged `2666.73 → 1047.20 ms` | `-60.73%`，`2.55×` | 保留 |
| moved-start 安全后缀复用 | global `78.548 → 1.040 ms` | `-98.68%`，`75.56×` | 保留 |

fine/wheel/legged 的状态、expanded/generated、raw/final path 点数和 fine 更新范围保持一致；global
moved-start 按预期由 `63075 expanded/cache=false` 变为 `0/cache=true`。复核期间发现并修复了两个安全
缺口：极小合法分辨率下的误共线会产生未认证 chord；独立地图可能复用相同数字 revision 的陈旧缓存。
最终实现对预压缩边重新做 LOS，对不同持久化 tile root 的 equal-revision snapshot 重新验完整 route
influence fingerprint。未引入 D* Lite、LPA*、多层新规划器或 coordinator 重构。

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

基线 `platform_elevation_physics.cpp` 中，每个输出格先扫描 hard footprint 邻域；若没有 blocked/UNKNOWN
提前返回，再扫描 `hard radius + preferred clearance` 的矩形邻域。`IntrinsicAt()` 虽避免重复计算
坡度/起伏，但缓存为 `std::map<GridIndex, ...>`，每次邻域访问仍做树查找。大片 FREE/risk 地形几乎都走
完整 clearance 扫描，所以比可早停的 dense rock 更慢。`fine_traversability_builder.cpp` 的全量分支还会
遍历所有 candidate tile cell；增量分支已经只遍历 influence halo。

### 2. Phase-aware path simplifier

`phase_aware_path_simplifier.cpp` 对每个 anchor 从整段末尾反向尝试 candidate，每次失败都重新做
supercover line-of-sight。长蛇形路径会产生大量“很远但被墙阻挡”的失败射线，因此 raw path 越长，
后处理增长越明显。

### 3. Legged directed-edge certification

基线 `legged_local_planner.cpp` 在每个展开状态先尝试终点认证，再对八邻域逐边认证。终点即使远超任何运动基元
长度，也会进入 edge cache 和 `SupportsTranslation()` 的基元循环；邻居状态的 closed/现有 g 检查则在
认证之后。`SegmentCells()` 每条边创建 `std::vector` 并用 `std::find` 去重。上述常数成本乘以数十万
expanded state 后形成约 500 万次 transition evaluation。

### 4. Global route cache

基线 `global_route_planner.cpp` 只有缓存起点、终点、profile 和 geometry 全部相同时才进入 reuse 检查。
因此 exact revision 和不影响 route influence 的新 revision 可复用，但机器人沿缓存路线移动一个 cell 后
必然重新运行搜索。搜索记录本身还是 `std::map<GridIndex, Record>`，在数十万展开时也有优化空间。

## 按收益和风险排序的实施方案

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
p50 明显下降，expanded/generated 与路径合同不变。`evaluated_transitions` 若因摘要改变统计粒度，必须
单独标注，不能把不同口径当作同一工作量直接比较。

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
| full fine | 6811.47 / 6675.13 / 6599.48 | 2213.65 / 2255.09 / 2208.51 | 6675.13 → 2213.65，-66.84%，3.02× | 409600 updated / 409600 examined，完全一致 |
| 单格 incremental fine | 3.829 / 3.781 / 3.844 | 2.118 / 2.171 / 2.159 | 3.829 → 2.159，-43.63%，1.77× | 293 / 1301，完全一致 |
| 32×32 patch fine | 57.505 / 56.623 / 57.932 | 41.794 / 43.276 / 41.955 | 57.505 → 41.955，-27.04%，1.37× | 2432 / 4556，完全一致 |

相关 fine builder 与 complex scenario CTest 为 2/2 通过，达到 10% 保留门槛。由于仅数据布局已经获得
显著收益，本轮不继续加入固定 stencil offset 或距离变换，避免扩大浮点边界与 cell-area 几何风险。

### P0-B：phase 内共线折点预压缩（保留）

在每个 phase 内先线性移除“共线且同方向”的中间 raw vertex，再对折点序列执行原有最远可见 greedy。
共线容差按 cross product 自身尺度计算，不再使用与地图尺度无关的绝对下限；压缩后的每条相邻边先经
原 supercover + `Allowed()` 认证，失败则退回 raw run，raw run 也不安全或发生取消/deadline 时返回空，
不发布部分路径。U 形阻挡和 `1e-8 m` 合法分辨率回归均保留安全转折；长折线回归还约束 LOS 工作为
线性预处理加少量候选探测。

640 窄通道/死胡同场景结果：

| 指标 | before 三次 (ms) | after 三次 (ms) | p50 变化 |
| --- | --- | --- | ---: |
| wheel total | 1501.67 / 1485.02 / 1495.68 | 43.015 / 43.516 / 42.375 | 1495.68 → 43.015，-97.12%，34.77× |
| wheel postprocess | 1469.24 / 1452.51 / 1464.17 | 10.663 / 11.000 / 10.706 | 1464.17 → 10.706，-99.27%，136.76× |

三次均为 `PLAN_FOUND`，expanded/generated 固定为 63075/63127，raw/final 固定为 50790/158；因此收益
来自消除重复 LOS candidate 扫描，不是减少搜索、降低分辨率或放宽安全条件。simplifier、wheel planner
和 complex scenario CTest 为 3/3 通过，达到保留门槛。

### P0-C：足式边认证前置门控（保留）

planner 构造时对不可变 capability 一次性建立合法平移上界和 spin delta 摘要；有序集合只与相邻候选
比较，在保持原容差去重语义的同时把最坏前处理从 `O(P²)` 限为 `O(P log P)`。每次 `Plan()` 只读取摘要，
远超运动基元能力的终点连接不再创建 edge-cache entry；邻居先检查 closed 与几何代价下界，只有可能改善
g 的边才进入原 `LeggedDirectedEdgeCache` 完整认证。短 segment 使用 16 格内联 buffer，超过容量才退化
为动态 vector。方向性、step/gap、高程、assumed start-prefix、连续终点和 terminal yaw 检查均保留。
30,000 个唯一 spin primitive 的回归证明简单位置请求不会再做每请求二次摘要扫描。基准按实际长期存活
planner 用法在迭代外构造 planner，因此本表是请求延迟，不包含一次性构造成本。另有 `-π/+π` 环绕、
非相邻近重复和输入顺序回归锁定 spin 摘要语义。

640 step/gap 场景结果：

| 指标 | before 三次 | after 三次 | p50/固定值变化 |
| --- | --- | --- | ---: |
| legged total (ms) | 2650.50 / 2669.71 / 2666.73 | 1066.10 / 1036.87 / 1047.20 | 2666.73 → 1047.20，-60.73%，2.55× |
| legged postprocess (ms) | 51.511 / 52.911 / 53.600 | 52.122 / 50.151 / 51.615 | 52.911 → 51.615，-2.45% |
| reported evaluated transitions | 4999407（每次） | 965377（每次） | 仅作 after 诊断，不作同口径倍率 |

三次均为 `PLAN_FOUND`，expanded/generated 固定为 352216/619965，raw/final 固定为 49211/5071。
新增长距离回归在优化前为 1252 次 transition evaluation，超过“可能松弛边”上界；优化后通过该上界。
需要注意，before 的 `evaluated_transitions` 包含逐 primitive 探测，after 的成功摘要构造发生在 planner
构造期，Plan 内该计数主要是 capability 查询和 spin 转移，二者不是同一工作单位。因此保留依据是
请求 wall-clock 降低 60.73%、expanded/generated 与路径完全一致，以及 25 项足式回归通过；不宣称
`4999407 → 965377` 本身代表精确 5.18 倍工作量下降。

### P1：moved-start 缓存路径后缀复用（保留）

缓存匹配 goal/profile/geometry 后，以可中断的线性查找确认新 start cell 位于缓存 route；同 revision 且
共享不可变 persistent tile root 时可直接复用，同 revision 但 root 不同、或 successor revision 时必须验证
完整 route influence fingerprint，匹配后才从该 cell 构造后缀。blocked start/goal 在任何缓存路径之前
拒绝。缓存路径之外的起点、跳跃 revision、profile/geometry/goal 变化和 route influence 不匹配仍进入
原完整搜索。

640 窄通道/死胡同场景的缓存四阶段对照：

| 阶段 | before p50 (ms) | after p50 (ms) | after 语义 |
| --- | ---: | ---: | --- |
| cold global | 78.991 | 79.388 | AVAILABLE；cache=false；63075 expanded |
| exact cache | 1.894 | 2.045 | AVAILABLE；cache=true；0 expanded |
| off-route revision update | 8.727 | 8.554 | AVAILABLE；cache=true；0 expanded |
| on-route revision update | 20.491 | 20.281 | NO_ROUTE；cache=false；31617 expanded |
| moved-start | 78.548 | 1.040 | AVAILABLE；cache=true；0 expanded；50789 points |

moved-start 目标阶段降低 98.68%，为 75.56×；三次 after 为 1.0337 / 1.0452 / 1.0396 ms。
新增单元回归同时验证路径从新连续起点开始、终点不变、后缀长度正确、偏离缓存 route 的起点不复用，
以及独立但 revision 相同的 snapshot 在 moved-start/route interior 被阻塞或 route terrain risk 改变时
绝不陈旧复用。冷启动和 exact cache 不是本项优化目标，表中如实保留其小幅波动。

## Fresh 构建与回归

冻结 before build 为 `/tmp/lunar-complex-final.uyoh2c/build`；最终在仓库外新建
`/tmp/lunar-complex-reviewed.RRAHfB/build`，从最终源码重新配置并构建 Jazzy `RelWithDebInfo`。
按包串行执行全部 core CTest：

```text
15/15 passed, 0 failed, 3.17 s
```

同一 fresh build 再以 `ctest --repeat until-fail:3 -j1` 串行复跑全部 15 个测试目标，三轮均通过，
总计 `9.49 s`；最终 `git diff --check` 无输出。

该 fresh build 的独立 benchmark 完成 320 全矩阵三次、640 窄通道三次、640 足式三次及 640
risk/UNKNOWN 三次定向复跑。320 矩阵中五个可达场景的 global 均为 `AVAILABLE`，wheel/legged/coordinator
均为 `PLAN_FOUND`；封闭目标三次均为 global `NO_ROUTE`、wheel/legged/coordinator `NO_PATH`，未出现
`TIMEOUT`、`CANCELED` 或空成功路径。最终原始 CSV 位于仓库外：

- `/tmp/lunar-complex-perf-20260904/final-reviewed-all-320.csv`
- `/tmp/lunar-complex-perf-20260904/final-reviewed-dead-ends.csv`
- `/tmp/lunar-complex-perf-20260904/final-reviewed-legged-step-gap.csv`
- `/tmp/lunar-complex-perf-20260904/final-reviewed-risk-unknown.csv`

仓库根目录运行全部 Python 静态合同为 `111 passed, 1 failed`。唯一失败来自既有
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
