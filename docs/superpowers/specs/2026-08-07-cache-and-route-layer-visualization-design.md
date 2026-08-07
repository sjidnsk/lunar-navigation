# 安全投影缓存、累计搜索统计与路线分层显示设计

日期：2026-08-07

状态：已按用户“前面的推荐修复也一并处理”授权冻结

## 1. 目标

在不增加搜索资源上限、不改变可行性结论的前提下，避免同一内容地图重复构建安全投影，并让用户在
RViz 中区分“全局路线已经找到”和“当前局部执行段已经完成认证”。同时把所有重试累计进耗时和
搜索计数，避免面板只显示最后一轮而无法解释实际等待。

## 2. 内容键缓存

主仓增加容量为 1 的不可变全局规划上下文缓存，缓存 `MapSnapshot + SafeProjection + terrain limits`。
键包含全局内容 generation、平台类型、platform id、capability version、地图几何和影响投影的安全配置。
命中时只复用不可变对象；任一键变化、取消或构建失败均不写缓存。缓存只改变重复计算，不改变搜索
Open、父链或输出排序。

轮式局部同一次请求内使用最远走廊的公共 `MapSnapshot/SafeProjection` 和一棵 ranked-frontier 搜索树；
不跨不同 `local_map_generation` 复用碰撞结果。若滚动请求的局部内容 generation 与完整投影键相同，
允许复用只读局部投影；键变化立即失效。

诊断新增但不修改消息 schema：

```text
hierarchical_global_projection_cache_hits
hierarchical_local_projection_cache_hits
hierarchical_local_search_runs
hierarchical_global_replans
planner_total_elapsed_s
```

既有 `hierarchical_local_attempts` 表示累计逻辑前沿尝试；expanded states、global/local elapsed 均跨条件
走廊重试累计。RViz 显示 `Frontiers / searches`、`Retries`、总耗时和轮式 2 秒结果。

## 3. 暂定全局路线观察器

保持冻结入口 `Planner::Plan(const PlannerInput&)` 不变，并新增不影响规划结果的可选观察入口。
地面全局 A* 成功且完成简化后，核心把不可变 `GlobalRoutePreview` 交给观察器；观察器异常必须被隔离，
不能改变规划结果。ROS server 使用它立即发布 transient-local
`/planning/provisional_route_markers`：

- namespace `provisional_global_route`；
- 青色、较细、半透明虚线/分段线；
- route id 和 request id 写入文本；
- 新 Goal 先 DELETE 旧暂定路线，再发布新路线；取消和平台切换清理 owned markers。

最终 `MotionReference` 仍通过已有 `/planning/certified_route_markers` 发布：认证局部执行段使用绿色粗实线；
飞跃式认证弹道和落区继续只出现在 certified topic。地面结果完成后保留当前请求的青色全局路线作为
上下文，绿色只表示可立即执行的局部段，不能把全局预览误称为已认证。

RViz 配置新增一个独立 MarkerArray display，名称为 `Provisional Global Route`。外部测试订阅两个 topic，
必须观察到 provisional ADD 的时间早于 certified ADD/Action terminal result，且 namespace、颜色和线宽不同。

## 4. 外部地图编码缓存

与地面滚动设计的中心保留区联动：全局 surface 点云在地图内容不变时只编码一次；局部 GridMap/hazard
在窗口不重心化时复用。每次发布仍更新 header stamp，内容 generation 由内容身份决定，不因纯时间戳
变化而递增。

## 5. 资格标准

- 同键连续规划第二次命中全局投影缓存，输出路线和 reason code 与无缓存完全一致；能力、安全配置或
  generation 改变时必须 miss。
- 轮式多个逻辑前沿只运行一棵局部树；条件走廊重试时计数和耗时累计，不归零。
- 观察器抛异常、订阅者缺失或 publisher 未激活均不影响 Action 结果。
- 进程测试验证暂定路线先出现、认证段后出现；新 Goal 不残留旧 marker；飞跃路线仍只显示认证结果。
- 所有性能测试为 Release，不设置 2 秒 runtime timeout；p95 门槛来自配套轮式设计。

不新增地图 Topic，不改变 `PlanMotion.action`、`PlannerDiagnostics.msg` 或 `MotionReference.msg`。
