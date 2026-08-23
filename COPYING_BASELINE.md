# Pure Planner 固定复制基线

- 复制日期：2026-08-22
- 固定提交：`e85aa5ff634cf06db8c5dd76cbd71b49e7902db8`
- 源路径：`ros2_ws/src/lunar_planner_core/`
- 目标路径：`pure_planner/ros2_ws/src/lunar_pure_planner_core/`

本目录仅由上述固定提交的 Git archive 中已跟踪文件机械复制而来；不包含源工作树的任何未提交修改，也不包含源包的 `build/`、`install/` 或 `log/` 产物。

复制时对路径和文件内容应用以下确定性重命名规则：

```text
lunar_planner_core -> lunar_pure_planner_core
lunar::planning    -> lunar::pure_planning
LUNAR_PLANNER_CORE -> LUNAR_PURE_PLANNER_CORE
```

## 三平台算法输出基线

`tests/parity/expected/{wheel,legged,hopper}.json` 只能由上述固定提交的
`lunar_planner_core` probe 生成。不得由 pure core、手写结果或其他提交更新。
parity probe 是不安装的离线测试工具；旧核心只允许作为单独的 baseline 二进制依赖，
不会进入 `pure_planner/ros2_ws/src` 的生产依赖，也不会与 pure core 链接到同一进程。

基线 reference worktree 必须保持只读、detached 且精确指向固定提交：

```bash
PURE_BASELINE_SHA=e85aa5ff634cf06db8c5dd76cbd71b49e7902db8
PURE_BASELINE_WORKTREE=/home/kai/WS/lunar-navigation-orin-pure-baseline
git worktree add --detach "$PURE_BASELINE_WORKTREE" "$PURE_BASELINE_SHA"
test "$(git -C "$PURE_BASELINE_WORKTREE" rev-parse HEAD)" = "$PURE_BASELINE_SHA"
test -z "$(git -C "$PURE_BASELINE_WORKTREE" status --porcelain)"
```

在 Ubuntu 22.04 ROS 2 Humble 容器内构建旧核心与 baseline probe；源码只读挂载，
全部构建和运行产物写到仓库外：

```bash
PURE_ARTIFACT_ROOT=/home/kai/CodexDownloads/lunar_navigation/pure_planner
docker run --rm \
  --volume /home/kai/WS/lunar-navigation-orin-pure-planner:/workspace:ro \
  --volume "$PURE_BASELINE_WORKTREE":/baseline:ro \
  --volume "$PURE_ARTIFACT_ROOT":/artifacts \
  --workdir /workspace \
  osrf/ros:humble-desktop-full-jammy \
  bash -lc 'set -eo pipefail
    apt-get update
    apt-get install -y nlohmann-json3-dev
    source /opt/ros/humble/setup.bash
    test "$ROS_DISTRO" = humble
    mkdir -p /artifacts/parity/build-baseline-core \
      /artifacts/parity/install-baseline-core \
      /artifacts/parity/log-baseline-core \
      /artifacts/parity/baseline
    colcon --log-base /artifacts/parity/log-baseline-core build \
      --base-paths /baseline/ros2_ws/src \
      --build-base /artifacts/parity/build-baseline-core \
      --install-base /artifacts/parity/install-baseline-core \
      --packages-select lunar_planner_core \
      --cmake-args -DBUILD_TESTING=OFF
    cmake -S /workspace/pure_planner/tests/parity \
      -B /artifacts/parity/build-baseline \
      -DPLANNER_VARIANT=old \
      -DPROBE_OUTPUT_DIR=/artifacts/parity/baseline \
      -DCMAKE_PREFIX_PATH=/artifacts/parity/install-baseline-core
    cmake --build /artifacts/parity/build-baseline --parallel
    ctest --test-dir /artifacts/parity/build-baseline --output-on-failure'
```

审核三次输出的字段和值及三个相同 SHA-256 后，才可从 baseline probe 的第一次输出
确定性生成仓库内 expected JSON：

```bash
for platform in wheel legged hopper; do
  cmp "$PURE_ARTIFACT_ROOT/parity/baseline/$platform.run-1.json" \
      "$PURE_ARTIFACT_ROOT/parity/baseline/$platform.run-2.json"
  cmp "$PURE_ARTIFACT_ROOT/parity/baseline/$platform.run-1.json" \
      "$PURE_ARTIFACT_ROOT/parity/baseline/$platform.run-3.json"
  python3 -m json.tool --sort-keys --indent 2 \
    "$PURE_ARTIFACT_ROOT/parity/baseline/$platform.run-1.json" \
    "pure_planner/tests/parity/expected/$platform.json"
done
```
