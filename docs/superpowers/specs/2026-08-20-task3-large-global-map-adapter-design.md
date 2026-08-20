# 课题三大范围地图适配与有界全局规划图设计

## 1. 决策

课题三的完整地图作为只读、瓦片化的长期地图存储；本项目不把其完整 0.2 m 栅格地图发布或复制为一次规划输入。

每个 `ExplorationTask` 在 `map` 坐标系中显式提供矩形任务 ROI。`luna_t3_global_map_adapter` 只读取该 ROI 覆盖的 SQLite tile，并将其保守聚合为一张满足现有规划器地图金字塔契约的有界全局 `GridMap`。局部规划继续使用课题三 60 m x 60 m、0.2 m 的实时局部图。

本设计采用现有 `lunar-conservative-grid-aggregation/v1` 聚合合同，不修改规划核心的 `GridMap` 快照接口。

## 2. 范围与非目标

### 范围

- 适配课题三 SQLite 全局地图与 `/Car/T3/mapping/global_map_revision`。
- 以 `ExplorationTask` 的 ROI、任务 ID 和 revision 驱动全局图工作集。
- 构建和增量更新 `/environment/map_global`。
- 保留课题三 `/Car/T3/mapping/grid_map` 到 `/environment/map_local` 的独立局部适配边界。
- 提供可观察的地图 revision、层级、尺寸、tile 缓存与拒绝原因诊断。

### 非目标

- 不修改课题三 SQLite 数据、WAL 或 SHM 文件。
- 不把完整原始全局地图作为 ROS `GridMap` 发布。
- 不实现规划核心的原生按需瓦片搜索。
- 不接管底盘控制、路径跟踪或任务系统。
- 不用零值或虚构数据填补不能可靠派生的地图层。

## 3. 架构与数据流

```text
ExplorationTask (ROI, mission_id, revision)
                 |
Task3 SQLite global map + global_map_revision
                 |
        luna_t3_global_map_adapter
                 |
  /environment/map_global (bounded, conservative GridMap)
                 |
            lunar_planner_ros

/Car/T3/mapping/grid_map
                 |
         luna_t3_local_map_adapter
                 |
 /environment/map_local (60 m x 60 m, 0.2 m GridMap)
```

The Task3 SQLite payload is an outer zlib stream containing NumPy `.npz` arrays. Its decode, bounded cache, conservative aggregation, and ROS `GridMap` publication therefore stay in one small Python runtime adapter (`sqlite3`, `zlib`, `numpy`, `rclpy`), avoiding a copy-through-file or subprocess bridge. The C++ ROI/level and aggregation components remain tested reference implementations of the planner contract. `luna_t3_global_map_adapter` has four responsibilities only:

1. Freeze the active task ROI from an `ACTIVE` `ExplorationTask` revision.
2. Read and cache only SQLite tiles overlapping that ROI.
3. Select the finest admissible global map level and construct a conservative global planning map.
4. Refresh affected cached tiles when the global-map revision changes, then atomically replace the published global-map snapshot.

It does not perform local planning, publish chassis commands, or write to the Task3 map database.

## 4. ROI, map levels, and resource bounds

The active task ROI is the exact outer rectangle used for the global planning map. It does not slide with local motion. A task revision change creates a new ROI and invalidates the old task's global-map working set.

The base resolution is `r0 = 0.2 m`; allowed global scales are `[1, 2, 4, 8, 16, 20]`, giving 0.2, 0.4, 0.8, 1.6, 3.2, and 4.0 m. The adapter selects the smallest level that satisfies the planner's `target_axis_cells = 256`, `maximum_axis_cells = 4096`, and `maximum_cells = 1048576` rules.

| ROI side length | Expected finest global level | Nominal grid side |
| ---: | ---: | ---: |
| 100 m | 0.4 m (L1) | 250 |
| 200 m | 0.8 m (L2) | 250 |
| 300 m | 1.6 m (L3) | 188 |
| 500 m | 3.2 m (L4) | 157 |
| 1024 m | 4.0 m (L5) | 256 |

The supported single-task upper bound is approximately 1024 m per axis. An ROI that cannot fit at L5 is rejected as `GLOBAL_MAP_SCALE_UNSUPPORTED`; the adapter must not silently select a coarser, non-contract resolution. Task systems should split larger missions into revisioned task regions.

## 5. Conservative aggregation

The adapter must implement `lunar-conservative-grid-aggregation/v1` exactly:

| Layer | Parent-cell aggregation |
| --- | --- |
| `valid_mask` | AND over all children |
| `obstacle`, `forbidden` | OR over all children |
| `obstacle_height`, `observation_age_s`, `obstacle_variance` | maximum |
| `observation_quality`, `observation_count` | minimum |
| `elevation` | arithmetic mean only if every child is valid; otherwise parent invalid |
| `elevation_variance` | maximum child variance plus variance of valid child elevations |

An incomplete boundary block is invalid and forbidden. Conservative compression may reject a narrow valid passage; it may not average a physical obstacle or unknown area into traversable space. Final endpoint and traversal acceptance remain the responsibility of the L0 local map and rolling local planner.

## 6. Map layer adaptation

Task3 provides local-map occupancy, semantic ID, elevation, and roughness. The adapter produces the planner's required layers with these rules:

| Planner layer | Source or safe derivation |
| --- | --- |
| `elevation` | Task3 elevation |
| `valid_mask` | valid occupancy and elevation |
| `obstacle` | configured occupancy and semantic mapping |
| `obstacle_height` | configured conservative derivation; otherwise mark non-traversable |
| `forbidden` | task forbidden zones, configured semantic exclusions, and invalid regions |
| `elevation_variance`, `obstacle_variance` | approved conservative bounds, never zero-filled |
| `observation_count` | SQLite value when available; otherwise one for a current valid local observation |
| `observation_age_s` | snapshot age only when freshness applies to the cell; retained cells of unknown age are invalid or low quality |
| `observation_quality` | validity, freshness, and configured semantic confidence rule |

No missing field may be represented as free space, zero uncertainty, or fresh observation without source evidence.

## 7. Caching, revision consistency, and publication

Two bounded caches are required:

1. **Raw tile cache**: keyed by tile coordinate and SQLite/global-map revision; stores only active ROI tiles plus a fixed adjacent margin and evicts by LRU.
2. **Global planning snapshot cache**: keyed by `(mission_id, mission_revision, map_revision, aggregation_version)`; shared read-only by planning requests.

On task activation or task revision, the adapter opens a SQLite read-only transaction, reads all required ROI tiles, constructs a complete map snapshot, and publishes it atomically. On an unchanged global revision it performs no SQLite read and no global-map reconstruction.

On a new global revision, the adapter identifies affected ROI tiles, refreshes only those cache entries, rebuilds the bounded published map, and atomically swaps the snapshot. If the revision changes during one construction transaction, the intermediate result is discarded and rebuilt from a single revision. A planning request always uses exactly one global snapshot; it may never mix old and new tile content.

## 8. Runtime inputs, outputs, and failure behavior

Inputs:

- `/mission/exploration_task` for mission ID, revision, `ACTIVE` state, ROI, forbidden and science regions.
- `/Car/T3/mapping/global_map_revision` for global database change detection.
- Task3 `global_grid_map.sqlite3` opened read-only for tile reads.
- `/Car/T3/mapping/grid_map`, consumed by the separate local-map adapter rather than to reconstruct the global map.

Outputs:

- `/environment/map_global`: `grid_map_msgs/msg/GridMap`, `frame_id=map`, exact selected global level, bounded dimensions.
- Adapter diagnostics: task and map revision, selected level, dimensions, source tile count, cache hits/misses, refresh wall time, and last rejection reason.

Failure handling:

- unreadable SQLite, missing required tile, or transaction/revision inconsistency: no partial global map publication;
- ROI larger than L5 capacity: explicit task rejection;
- insufficient layer evidence: affected cells become invalid/forbidden;
- stale global or local map: no new planning request until a coherent snapshot exists;
- a previously authorized path may be executed only to its existing safe-stop boundary; it must not authorize a newly selected global candidate on stale data.

## 9. Verification

The implementation must demonstrate:

1. 100 m, 300 m, and 500 m ROIs select L1, L3, and L4 respectively.
2. A single L0 obstacle or forbidden cell propagates to every appropriate parent level.
3. An unchanged revision causes neither source tile reads nor global-map reconstruction.
4. A changed revision refreshes only the intersecting tile cache entries and produces a new atomic snapshot.
5. Planning requests retain one global revision throughout snapshot freeze and reject mixed revisions.
6. Missing tiles, failed read-only transactions, over-limit ROI, and incomplete layers fail closed.
7. The 60 m x 60 m L0 local map remains independent of global-map size and update frequency.

## 10. Integration order

1. Add the adapter's data model, ROI/level selection, and pure conservative aggregation tests.
2. Add read-only SQLite tile provider and revisioned bounded caches.
3. Add ROS Task3 revision/task subscriptions and canonical global-map publisher.
4. Add local-map layer adapter and coherence diagnostics.
5. Validate against a recorded Task3 map/pose/revision stream before enabling live planner requests.

The planner core, safety constraints, path execution boundary, and task-system ownership remain unchanged.
