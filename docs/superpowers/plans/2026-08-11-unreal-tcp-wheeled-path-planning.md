# Unreal TCP Wheeled Path Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the repository-owned ROS 2 Humble side of `lunar-unreal-tcp/v1` so one WHEELED robot in Unreal can stream observed elevation and state, receive existing `MotionReference` trajectories, and execute rolling `PlanMotion` requests from one ROS `PoseStamped` goal.

**Architecture:** Add three focused C++20 Lifecycle packages. `lunar_unreal_tcp_bridge` owns wire framing, strict metadata, session safety, coordinate conversion, TCP I/O, ROS state/map publication, and trajectory/feedback translation. `lunar_observed_map` owns sparse observed-only L0 fusion and planner-ready local/global GridMaps; `lunar_goal_coordinator` owns the single-goal rolling state machine and is the only `/lunar/motion_reference` publisher in this launch graph. A repository-local fake Unreal server verifies the complete ROS side; the real Unreal Runtime plugin remains in its external Git root.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, C++20, `rclcpp_lifecycle`, POSIX TCP, nlohmann-json 3.10, Eigen3, GridMap, GoogleTest, `launch_testing`, Python 3 fake server.

## Global Constraints

- Work only in branch `feature/unreal-tcp-wheeled-path-planning` at `/mnt/data/WS/.lunar-navigation-worktrees/unreal-tcp-wheeled-path-planning`.
- Source `/opt/ros/humble/setup.bash` before ROS commands and keep build/install/log under `/home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning`.
- Preserve `PlanMotion.action`, `MotionReference.msg`, and all names in `lunar-external-interfaces/v5` unchanged.
- Support only one Unreal process, one active TCP client session, one WHEELED robot, and one explicit `/goal_pose`.
- Never start exploration policy, PPO, training, LEGGED, HOPPER, motor torque, or wheel-speed control in this launch graph.
- TCP online values remain Unreal-native centimeters; all ROS-facing positions and elevations use meters.
- Only sensor-observed cells may enter map products; invalid input cannot erase valid history and Landscape truth cannot be read.
- Wire framing is `lunar-unreal-tcp/v1`: fixed 48-byte little-endian header, strict UTF-8 JSON object, optional binary payload, CRC-32/ISO-HDLC, 65,536-byte metadata limit, and 8-MiB body limit.
- Session sequence starts at 1 and increases strictly; stale session, malformed frame, heartbeat loss, stale state/map, or mismatched plan fails closed to HOLD.
- The external Unreal project and its `Content`, `Binaries`, `Intermediate`, `Saved`, and `DerivedDataCache` directories never enter this repository.

---

### Task 1: Wire protocol, strict metadata, and golden vectors

**Files:**
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/package.xml`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/protocol.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/metadata_validation.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/protocol.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/metadata_validation.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/protocol_test.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/metadata_validation_test.cpp`
- Create: `tests/fixtures/unreal_tcp/protocol_v1_vectors.json`

**Interfaces:**
- Produces: `enum class MessageType : std::uint16_t`, `FrameHeader`, `Frame`, `ProtocolError`, `EncodeFrame(const Frame&)`, `DecodeFrame(std::span<const std::byte>)`, `Crc32(std::span<const std::byte>)`, and `ValidateMetadata(MessageType, const nlohmann::json&)`.
- Invariant: encoding writes every integer explicitly in little-endian order and never serializes a native struct.
- Invariant: decoding rejects duplicate JSON keys, unknown fields, non-finite values, invalid flags/reserved bytes, CRC mismatch, size overflow, and a total body above 8 MiB.

- [ ] **Step 1: Create package scaffolding, literal golden vectors, and failing codec tests**

  Add GoogleTests named `EncodesGoldenHelloFrame`, `DecodesGoldenRobotStateFrame`, `RejectsHeaderAndBodyCrcMismatch`, `RejectsUnknownFlagsAndNonzeroReserved`, `RejectsMetadataAndBodyLimits`, and `RejectsDuplicateJsonKeys`. The fixture JSON stores literal hexadecimal frames and decoded literal fields; derive CRC literals independently with Python `zlib.crc32`, never through the C++ codec.

- [ ] **Step 2: Run the protocol tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/baseline/runtime/install/setup.bash
  colcon --log-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/log build --base-paths ros2_ws/src --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
  ```

  Expected: link or compile failure because `EncodeFrame`, `DecodeFrame`, and validation functions have no implementation.

- [ ] **Step 3: Implement the minimal frame codec and strict JSON validator**

  Use these public constants and result shape:

  ```cpp
  inline constexpr std::size_t kHeaderSize = 48U;
  inline constexpr std::size_t kMaximumMetadataBytes = 65'536U;
  inline constexpr std::size_t kMaximumBodyBytes = 8U * 1024U * 1024U;
  inline constexpr std::array<std::byte, 4> kMagic{
      std::byte{'L'}, std::byte{'N'}, std::byte{'T'}, std::byte{'1'}};

  struct DecodeResult final {
    std::optional<Frame> frame;
    std::optional<ProtocolError> error;
    [[nodiscard]] bool ok() const noexcept { return frame.has_value(); }
  };
  ```

  Implement CRC-32/ISO-HDLC with polynomial `0xEDB88320`, initial/final XOR `0xffffffff`, and validate header CRC with bytes 40-43 zeroed. Use a nlohmann SAX pass with an object-key stack to reject duplicate keys before normal parsing. Validate exact allowed and required top-level keys for all nine v1 message types.

- [ ] **Step 4: Run the package tests and verify GREEN**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon --log-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/test-log test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --event-handlers console_direct+
  colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --verbose
  ```

  Expected: all Task 1 tests pass with zero failures.

- [ ] **Step 5: Commit the independently usable protocol library**

  ```bash
  git add ros2_ws/src/lunar_unreal_tcp_bridge tests/fixtures/unreal_tcp/protocol_v1_vectors.json
  git commit -m "feat(unreal-tcp): add protocol v1 codec"
  ```

### Task 2: Stream decoder, priority queue, and session protocol

**Files:**
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/stream_decoder.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/outbound_queue.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/session_protocol.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/stream_decoder.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/outbound_queue.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/session_protocol.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/stream_decoder_test.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/outbound_queue_test.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/session_protocol_test.cpp`
- Modify: `ros2_ws/src/lunar_unreal_tcp_bridge/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1 `Frame`, `FrameHeader`, `DecodeFrame`, and metadata validation.
- Produces: `StreamDecoder::Push(std::span<const std::byte>)`, `OutboundQueue::Push(Frame)`, `OutboundQueue::Pop()`, and `SessionProtocol::{BuildHello, AcceptHelloAck, AcceptIncoming, Reset}`.
- Invariant: control/reference/feedback/error cannot be silently dropped; state and map each occupy a replaceable latest-only slot.

- [ ] **Step 1: Write failing stream and queue tests**

  Add tests `DecodesEveryByteSplitOfGoldenFrame`, `DecodesCoalescedFramesAndRetainsHalfFrame`, `RejectsMaliciousLengthBeforeAllocation`, `ReplacesOnlyLatestStateAndMap`, `PreservesControlReferenceFeedbackOrdering`, and `ReportsHighPriorityQueueExhaustion`.

- [ ] **Step 2: Run focused tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R 'stream_decoder|outbound_queue' --output-on-failure
  ```

  Expected: failure because stream and queue types do not exist.

- [ ] **Step 3: Implement bounded incremental decode and priority scheduling**

  Keep at most `kHeaderSize + kMaximumBodyBytes` receive bytes. Parse the advertised body size only after magic/version/header-size and header CRC validate. Use four priority levels from the design; store state/map in `std::optional<Frame>` slots and use bounded `std::deque<Frame>` for non-droppable classes.

- [ ] **Step 4: Write failing session tests**

  Add tests `HandshakeFreezesNewSessionAndCalibration`, `RequiresSequenceOneThenStrictIncrease`, `AllowsForwardGapAfterLatestOnlyDrop`, `RejectsWrongSessionAndSimulationTimeRollback`, `ReconnectInvalidatesOldSession`, and `HeartbeatExpiryRequiresHold` using a fake steady clock.

- [ ] **Step 5: Run focused session test and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R session_protocol --output-on-failure
  ```

  Expected: failure because `SessionProtocol` is not implemented.

- [ ] **Step 6: Implement the session state machine**

  Expose states `DISCONNECTED`, `HANDSHAKING`, `SYNCING`, `READY`, `EXECUTING`, and `HOLD`. `BuildHello` emits sequence 1; `AcceptHelloAck` freezes UUID session, scene/robot identity, coordinate convention, calibration matrices/hash, rates, and body limit. `AcceptIncoming` rejects duplicate/backward sequence, wrong session, time rollback, and invalid plan identity, but permits a forward sequence gap.

- [ ] **Step 7: Run all bridge-library tests and commit**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --event-handlers console_direct+
  colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --verbose
  git add ros2_ws/src/lunar_unreal_tcp_bridge
  git commit -m "feat(unreal-tcp): add bounded session stream"
  ```

### Task 3: Coordinate, state, map, and trajectory conversion

**Files:**
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/coordinate_transform.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/ros_conversions.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/coordinate_transform.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/ros_conversions.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/coordinate_transform_test.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/ros_conversions_test.cpp`
- Modify: `ros2_ws/src/lunar_unreal_tcp_bridge/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_unreal_tcp_bridge/package.xml`

**Interfaces:**
- Consumes: validated `ROBOT_STATE`, `LOCAL_ELEVATION_MAP`, and `MOTION_REFERENCE` metadata/payload.
- Produces: `CoordinateTransform`, `ConvertRobotState`, `ConvertObservedElevation`, `ConvertMotionReference`, and `ConvertExecutionFeedback`.
- Invariant: one frozen basis/matrix handles positions, rotations, linear/angular velocities, grid origin/basis, base-link/base-footprint calibration, and inverse trajectory conversion.

- [ ] **Step 1: Write failing coordinate golden tests**

  Add tests for Unreal origin, each unit axis, arbitrary yaw, negative Landscape coordinates `(-205160, -511559, 50)` cm, quaternion basis change, angular velocity pseudovector behavior, and ROS-Unreal-ROS round trip. The expected values are literal hand-derived vectors/matrices.

- [ ] **Step 2: Run the coordinate tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R coordinate_transform --output-on-failure
  ```

  Expected: failure because coordinate conversion is absent.

- [ ] **Step 3: Implement full basis and rigid calibration conversion**

  Store a 3x3 invertible basis `B`, translation in meters, and the two rigid calibrations. Convert orientation with `R_ros = B * R_unreal * B.inverse()` and normalize the resulting quaternion. Convert angular velocity with determinant-aware pseudovector handling. Reject non-rigid/non-finite matrices and a handshake calibration hash mismatch.

- [ ] **Step 4: Write failing ROS conversion tests**

  Add tests `RobotStateProducesExternalAndWheeledOdometryFromOneSnapshot`, `ObservedMapUsesCellZeroAndUvAxesNotLandscapeOrigin`, `ObservedMapRejectsBadBitsetAndNonfiniteValidElevation`, `MotionReferenceEncodes112ByteStrictlyIncreasingRecords`, `MotionReferenceRejectsNonWheeledOrWrongCalibration`, and `ExecutionFeedbackMapsStableStatesAndReasons`.

- [ ] **Step 5: Run conversion tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R ros_conversions --output-on-failure
  ```

  Expected: failure because ROS conversions are absent.

- [ ] **Step 6: Implement ROS conversions and verify all package tests**

  Decode exactly 102,400 `float32_le` elevation samples plus 12,800 `bitset_lsb0` bytes. Publish raw observed GridMap with only `elevation` and `valid_mask`. Encode trajectory records explicitly as one `int64` and thirteen `float64` values, total 112 bytes per point; `time_from_start_ns` begins at zero and increases strictly.

- [ ] **Step 7: Commit coordinate and message conversion**

  ```bash
  git add ros2_ws/src/lunar_unreal_tcp_bridge
  git commit -m "feat(unreal-tcp): convert Unreal state and references"
  ```

### Task 4: TCP client and Lifecycle bridge node

**Files:**
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/tcp_client.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/include/lunar_unreal_tcp_bridge/bridge_node.hpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/tcp_client.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/bridge_node.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/src/main.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/config/bridge.yaml`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/tcp_client_test.cpp`
- Create: `ros2_ws/src/lunar_unreal_tcp_bridge/test/bridge_node_test.cpp`
- Modify: `ros2_ws/src/lunar_unreal_tcp_bridge/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_unreal_tcp_bridge/package.xml`

**Interfaces:**
- Consumes: Task 1-3 protocol/session/conversion library and `/lunar/motion_reference`.
- Produces: `/clock`, `/lunar/unreal/observed_elevation`, `/localization/odometry`, `/lunar/unreal/wheeled_odometry`, `/localization/status`, `/tf`, `/tf_static`, `/execution/motion_feedback`, services `/lunar/unreal/{start,hold,resume,reset_session}`, and `/lunar/unreal/status`.
- Produces: dependency-injectable `SessionTransport` so node tests exercise the real node without mocking ROS publishers.

- [ ] **Step 1: Write failing socket and reconnect tests**

  Use a loopback test server and add `ConnectsWithTcpNoDelayAndKeepalive`, `HandlesFragmentedAndCoalescedFrames`, `ReconnectBackoffIsBoundedFromHalfToFiveSeconds`, `DisconnectStopsWriterAndInvalidatesSession`, and `HighPriorityOverflowSendsHoldThenDisconnects`.

- [ ] **Step 2: Run socket tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R tcp_client --output-on-failure
  ```

  Expected: failure because `TcpClient` is absent.

- [ ] **Step 3: Implement one-reader/one-writer persistent TCP client**

  Use `std::jthread` and stop tokens, `getaddrinfo`, `TCP_NODELAY`, `SO_KEEPALIVE`, bounded reconnect, exact send loops, and poll-based interruptible receive. Never invoke ROS publishers from the I/O thread; deliver immutable frames through callbacks/queues.

- [ ] **Step 4: Write failing Lifecycle bridge tests**

  Add tests `ConfigureDeclaresFrozenParameters`, `ActivateStartsTransportAndDeactivateHolds`, `HandshakeRequiresStateAndMapBeforeReady`, `PublishesClockOdometryTfAndObservedMap`, `ForwardsOnlyValidWheeledReference`, `MatchingFeedbackPublishesExternalContract`, and `HeartbeatOrFreshnessExpiryFailsClosed`.

- [ ] **Step 5: Run bridge-node tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R bridge_node --output-on-failure
  ```

  Expected: failure because the Lifecycle node is absent.

- [ ] **Step 6: Implement bridge node, services, diagnostics, and watchdog**

  Network callbacks enqueue events; a mutually exclusive ROS callback group drains them. Use Unreal simulation time for message stamps and `/clock`, and `steady_clock` for 3-second heartbeat plus 1-second state/map freshness. Any forced-HOLD condition publishes a diagnostic reason, sends CONTROL/HOLD when the socket is usable, and invalidates the active plan/session as specified.

- [ ] **Step 7: Build, run all bridge tests, and commit**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_unreal_tcp_bridge --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --event-handlers console_direct+
  colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --verbose
  git add ros2_ws/src/lunar_unreal_tcp_bridge
  git commit -m "feat(unreal-tcp): add lifecycle ROS bridge"
  ```

### Task 5: Sparse observed-only L0 fusion and obstacle semantics

**Files:**
- Create: `ros2_ws/src/lunar_observed_map/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_observed_map/package.xml`
- Create: `ros2_ws/src/lunar_observed_map/include/lunar_observed_map/map_types.hpp`
- Create: `ros2_ws/src/lunar_observed_map/include/lunar_observed_map/sparse_observed_map.hpp`
- Create: `ros2_ws/src/lunar_observed_map/include/lunar_observed_map/obstacle_classifier.hpp`
- Create: `ros2_ws/src/lunar_observed_map/src/sparse_observed_map.cpp`
- Create: `ros2_ws/src/lunar_observed_map/src/obstacle_classifier.cpp`
- Create: `ros2_ws/src/lunar_observed_map/test/sparse_observed_map_test.cpp`
- Create: `ros2_ws/src/lunar_observed_map/test/obstacle_classifier_test.cpp`

**Interfaces:**
- Produces: `ObservedPatch`, `CellEvidence`, `TileKey`, `SparseObservedMap::Fuse`, `SparseObservedMap::ResetSession`, and `ObstacleClassifier::Classify`.
- Invariant: L0 is 0.2 m, tile is 256x256, session budget is 512 allocated tiles, and negative coordinates use mathematical floor division.

- [ ] **Step 1: Write failing sparse fusion tests**

  Add `AllocatesOnlyForValidObservation`, `MapsPositiveNegativeAndBoundaryCellsToTiles`, `InvalidPatchCannotEraseHistory`, `WelfordMeanVarianceCountAndAgeAreExact`, `SessionResetDropsAllOldSceneEvidence`, and `RefusesTheFiveHundredThirteenthTileWithoutEviction`.

- [ ] **Step 2: Run sparse-map tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_observed_map --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
  ```

  Expected: compile/link failure because sparse map behavior is absent.

- [ ] **Step 3: Implement sparse tiles and online evidence fusion**

  Each allocated tile owns exactly 65,536 `CellEvidence` entries. Apply Welford online mean/M2, monotonic observation count, and last simulation timestamp only to valid finite observations. Preflight all tile allocations for one patch so a budget failure is atomic.

- [ ] **Step 4: Write failing obstacle classification tests**

  Add `ClassifiesUnknownFreeAndOccupiedIndependently`, `ThresholdIsFreeAtPointTwoAndOccupiedAbovePointTwo`, `DetectsNegativeStepAsObstacleWithConservativeHeight`, `UsesObservedNeighborsOnlyForSupport`, and `ForbiddenRemainsIndependentOfUnknownAndObstacle`.

- [ ] **Step 5: Run classifier tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_observed_map --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R obstacle_classifier --output-on-failure
  ```

  Expected: failure because classification is absent.

- [ ] **Step 6: Implement 0.20 m WHEELED relief classification and commit**

  Estimate support as the median of finite valid 3x3 neighbors. Mark occupied when positive residual or any observed adjacent discontinuity exceeds `0.20 m`; a negative step uses at least one L0 resolution as conservative `obstacle_height`. Do not perform vehicle-footprint inflation.

- [ ] **Step 7: Run map-core tests and commit**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_observed_map --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --event-handlers console_direct+
  colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --verbose
  git add ros2_ws/src/lunar_observed_map
  git commit -m "feat(observed-map): fuse sparse observed elevation"
  ```

### Task 6: Planner GridMap products and observed-map Lifecycle node

**Files:**
- Create: `ros2_ws/src/lunar_observed_map/include/lunar_observed_map/grid_map_products.hpp`
- Create: `ros2_ws/src/lunar_observed_map/include/lunar_observed_map/observed_map_node.hpp`
- Create: `ros2_ws/src/lunar_observed_map/src/grid_map_products.cpp`
- Create: `ros2_ws/src/lunar_observed_map/src/observed_map_node.cpp`
- Create: `ros2_ws/src/lunar_observed_map/src/main.cpp`
- Create: `ros2_ws/src/lunar_observed_map/config/observed_map.yaml`
- Create: `ros2_ws/src/lunar_observed_map/test/grid_map_products_test.cpp`
- Create: `ros2_ws/src/lunar_observed_map/test/observed_map_node_test.cpp`
- Modify: `ros2_ws/src/lunar_observed_map/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_observed_map/package.xml`

**Interfaces:**
- Consumes: `/lunar/unreal/observed_elevation`, `/lunar/unreal/wheeled_odometry`, `/goal_pose`, and session reset indication.
- Produces: `/environment/map_local` in `odom` and `/environment/map_global` in `map`, each with the ten frozen layers.
- Invariant: local map is 64 m x 64 m at 0.2 m; active global map chooses the smallest admissible factor from `{1,2,4,8,16,20}` within cell/axis/1,024 m limits.

- [ ] **Step 1: Write failing GridMap product tests**

  Add `BuildsThreeHundredTwentySquareLocalMapWithTenLayers`, `NeverEmitsUnobservedTruth`, `CropsAcrossSparseTileBoundaries`, `SelectsSmallestAdmissibleGlobalLevel`, `AppliesConservativeAndOrMaxMinAggregation`, `IncludesRobotGoalAndObservedCorridor`, and `RejectsGoalOutsideObservedConnectedEvidence`.

- [ ] **Step 2: Run product tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_observed_map --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R grid_map_products --output-on-failure
  ```

  Expected: failure because product builder is absent.

- [ ] **Step 3: Implement ten-layer products and conservative pyramid**

  Build `std_msgs/Float32MultiArray` in the exact GridMap circular-buffer layout accepted by `lunar_planner_ros::GridMapAdapter`. Aggregation uses valid AND, obstacle/forbidden OR, maximum obstacle height/age/variance, minimum quality/count, and mean elevation over valid children.

- [ ] **Step 4: Write failing Lifecycle node tests**

  Add `ConfigureRequiresFrozenMapParameters`, `ActivatePublishesInitialRobotCenteredGlobalMap`, `ObservedPatchAndOdometryProduceSameGenerationProducts`, `GoalStampForcesContainingGlobalGenerationBeforePlanning`, `TileBudgetErrorKeepsPreviousProducts`, and `ResetSessionClearsProductsAndReturnsToSyncing`.

- [ ] **Step 5: Run node tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_observed_map --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R observed_map_node --output-on-failure
  ```

  Expected: failure because node is absent.

- [ ] **Step 6: Implement Lifecycle node and verify planner adapter compatibility**

  Subscribe with reliable depth 1 for observed map and transient-local goal; publish local/global with reliable transient-local depth 1. Stamp products from the common simulation snapshot, reject state/map skew above 0.2 s, and run each generated message through the real planner `GridMapAdapter` in tests.

- [ ] **Step 7: Run package tests and commit**

  ```bash
  git add ros2_ws/src/lunar_observed_map
  git commit -m "feat(observed-map): publish planner map products"
  ```

### Task 7: Single-goal rolling PlanMotion coordinator

**Files:**
- Create: `ros2_ws/src/lunar_goal_coordinator/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_goal_coordinator/package.xml`
- Create: `ros2_ws/src/lunar_goal_coordinator/include/lunar_goal_coordinator/goal_state_machine.hpp`
- Create: `ros2_ws/src/lunar_goal_coordinator/include/lunar_goal_coordinator/goal_coordinator_node.hpp`
- Create: `ros2_ws/src/lunar_goal_coordinator/src/goal_state_machine.cpp`
- Create: `ros2_ws/src/lunar_goal_coordinator/src/goal_coordinator_node.cpp`
- Create: `ros2_ws/src/lunar_goal_coordinator/src/main.cpp`
- Create: `ros2_ws/src/lunar_goal_coordinator/config/goal_coordinator.yaml`
- Create: `ros2_ws/src/lunar_goal_coordinator/test/goal_state_machine_test.cpp`
- Create: `ros2_ws/src/lunar_goal_coordinator/test/goal_coordinator_node_test.cpp`

**Interfaces:**
- Consumes: `/goal_pose`, current map/odometry/TF, `/execution/motion_feedback`, and `/plan_motion` Action.
- Produces: minimal `/mission/exploration_task`, the only `/lunar/motion_reference`, `/lunar/path_planning/status`, `/lunar/path_planning/cancel`, and HOLD through `/lunar/unreal/hold`.
- Invariant: no implicit goal replacement; only a successful WHEELED result with `ACTIVATE_NEW_REFERENCE` and `has_reference=true` is published.

- [ ] **Step 1: Write failing pure state-machine tests**

  Add `AcceptsFiniteMapGoalAndNormalizesQuaternion`, `RejectsWrongFrameUnknownObstacleForbiddenOrDisconnectedGoal`, `RejectsImplicitReplacementUntilCanceled`, `PublishesOnlyActivateNewWheeledReference`, `SegmentCompleteWaitsForStrictlyNewerStateAndMap`, `RollsAgainOutsideTolerance`, `SucceedsInsideHalfMeterAndFifteenDegrees`, and `PlannerOrExecutorFailureEntersHoldWithStableReason`.

- [ ] **Step 2: Run state-machine tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_goal_coordinator --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
  ```

  Expected: compile/link failure because coordinator core is absent.

- [ ] **Step 3: Implement deterministic rolling state machine**

  Generate mission ID from TCP session ID plus target UUID, revision from 1 upward, point goal with 0.5 m tolerance and default 15-degree yaw tolerance, empty science regions, and ROI equal to active global map. Model explicit events rather than calling ROS inside the core.

- [ ] **Step 4: Write failing coordinator-node tests**

  Add `GoalWaitsForContainingGlobalMapGeneration`, `SendsPlanMotionWithReplaceFalse`, `PublishesMinimalMissionBeforeActionGoal`, `PublishesReturnedReferenceOnce`, `MatchingFeedbackAndNewSnapshotTriggerSecondPlan`, `CancelWaitsForCanceledThenAcceptsNewGoal`, and `LaunchGraphHasSingleMotionReferencePublisher`.

- [ ] **Step 5: Run node tests and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  colcon test --packages-select lunar_goal_coordinator --build-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/build --install-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install --ctest-args -R goal_coordinator_node --output-on-failure
  ```

  Expected: failure because ROS node is absent.

- [ ] **Step 6: Implement Lifecycle coordinator and Action client**

  Use one mutually-exclusive callback group for state transitions and an Action callback group for `/plan_motion`. Validate fresh same-generation inputs before sending, set `replace_active_request=false`, require result request/mission/revision identity, and publish the reference with reliable volatile depth 1.

- [ ] **Step 7: Run coordinator tests and commit**

  ```bash
  git add ros2_ws/src/lunar_goal_coordinator
  git commit -m "feat(goal): add rolling PlanMotion coordinator"
  ```

### Task 8: Launch graph, fake Unreal server, and ROS integration

**Files:**
- Create: `ros2_ws/src/lunar_navigation_config/config/unreal_tcp_wheeled_path_planning.yaml`
- Create: `ros2_ws/src/lunar_navigation_config/launch/unreal_tcp_wheeled_path_planning.launch.py`
- Create: `tests/integration/unreal_tcp/__init__.py`
- Create: `tests/integration/unreal_tcp/fake_unreal_server.py`
- Create: `tests/integration/unreal_tcp/test_unreal_tcp_stack.launch.py`
- Create: `tests/integration/unreal_tcp/test_launch_contract.py`
- Modify: `ros2_ws/src/lunar_navigation_config/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_navigation_config/package.xml`
- Modify: `scripts/build_runtime.sh`

**Interfaces:**
- Consumes: all three new packages and existing `lunar_planner_ros`.
- Produces: one launch graph containing bridge, observed map, goal coordinator, and planner; planner remaps `/localization/odometry` to `/lunar/unreal/wheeled_odometry`.
- Invariant: graph contains no `lunar_exploration_policy`, `lunar_interface_v1_policy`, PPO, or training process.

- [ ] **Step 1: Write failing executable launch-contract test**

  Import the real launch file, resolve actions, and assert exact package/executable/remap identities, `use_sim_time=true`, one motion-reference publisher owner, and absence of exploration/training nodes. The failure must be missing launch artifact, not a source-text grep.

- [ ] **Step 2: Run launch-contract test and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  python3 -m pytest -q tests/integration/unreal_tcp/test_launch_contract.py
  ```

  Expected: failure because the launch file does not exist.

- [ ] **Step 3: Implement configuration and launch graph**

  Install `launch/`, add new package exec dependencies, and make `scripts/build_runtime.sh` include all packages automatically while continuing to place artifacts under the caller-supplied external output root.

- [ ] **Step 4: Write fake server and failing end-to-end launch test**

  The fake server implements real golden codec behavior, HELLO/ACK, 20 Hz state, 5 Hz observed map, heartbeat, CONTROL acknowledgments, trajectory ACCEPTED/EXECUTING/SEGMENT_COMPLETE, intentional wrong-session/stale/skew injections, and disconnect. Test cases are `HandshakePublishesClockAndPlannerMaps`, `GoalProducesWheeledReferenceAndMatchingFeedback`, `SegmentCompleteAndNewSnapshotTriggerSecondPlan`, `WrongSessionStaleSkewAndDisconnectFailClosed`, and `ReconnectDoesNotReplayOldReference`.

- [ ] **Step 5: Run end-to-end test and verify RED**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  python3 -m pytest -q tests/integration/unreal_tcp/test_unreal_tcp_stack.launch.py
  ```

  Expected: failure at the first unimplemented integration behavior.

- [ ] **Step 6: Complete only the integration wiring required to make GREEN**

  Fix package manifests, launch transitions, topic QoS/remaps, and fake-server orchestration without changing frozen messages or adding exploration behavior.

- [ ] **Step 7: Run integration suite and commit**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/implementation/install/setup.bash
  python3 -m pytest -q tests/integration/unreal_tcp
  git add ros2_ws/src/lunar_navigation_config scripts/build_runtime.sh tests/integration/unreal_tcp
  git commit -m "test(unreal-tcp): add fake server integration stack"
  ```

### Task 9: Protocol/deployment handoff, regression, and qualification boundary

**Files:**
- Create: `docs/interfaces/lunar-unreal-tcp-v1.md`
- Create: `docs/deployment/unreal-tcp-wheeled-path-planning.md`
- Create: `docs/validation/unreal-tcp-wheeled-path-planning-qualification.md`
- Create: `scripts/run_unreal_tcp_regression.sh`
- Modify: `docs/deployment/interface-v1-quickstart.md`
- Modify: `README.md`

**Interfaces:**
- Documents: exact header offsets, message schemas, binary layouts, CRC, coordinate/calibration rules, Windows firewall/listen configuration, ROS client commands, diagnostics, HOLD behavior, golden vectors, and external Unreal plugin obligations.
- Qualification boundary: repository fake-server tests may qualify `ROS-side simulated-ready`; only external Unreal Engine 5.0.1 plugin build, three functional cases, network-disconnect case, and 30-minute two-machine run can qualify the full system.

- [ ] **Step 1: Add a reproducible regression script**

  The executable script accepts one absolute output directory, rejects repository-contained paths, sources ROS Humble, builds to external build/install/log directories, runs all three new package tests, integration tests, affected planner/interface-v1 regressions, and repository boundary checks.

- [ ] **Step 2: Run the script against a fresh external output root**

  ```bash
  ./scripts/run_unreal_tcp_regression.sh /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/final
  ```

  Expected: build and all automated repository tests pass with zero failures.

- [ ] **Step 3: Write protocol, deployment, and qualification documents**

  Include copyable Ubuntu commands and expected evidence. State `AGX plugin version=UNKNOWN` is transportable but never silently upgraded to a known version. Mark real Windows plugin build, AGX executor behavior, LAN firewall, three physical scenarios, disconnect HOLD within 3 seconds, and 30-minute stability as externally pending until recorded evidence exists.

- [ ] **Step 4: Run UTF-8, placeholder, boundary, and diff checks**

  ```bash
  python3 -c "from pathlib import Path; [p.read_text(encoding='utf-8') for p in Path('docs').rglob('*.md')]"
  python3 tools/check_repository_boundaries.py .
  python3 -m pytest -q tests/foundation/test_repository_boundaries.py
  git diff --check
  ```

  Expected: all commands exit zero.

- [ ] **Step 5: Run affected regression suites from the fresh install**

  ```bash
  source /opt/ros/humble/setup.bash
  source /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/final/install/setup.bash
  python3 -m pytest -q tests/foundation tests/ros tests/integration/unreal_tcp ros2_ws/src/lunar_external_adapter/test ros2_ws/src/lunar_exploration_policy/test
  colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/final/build --verbose
  ```

  Expected: zero failed tests; exploration tests remain regression-only and the Unreal launch graph does not run that policy.

- [ ] **Step 6: Review the final diff against every design completion criterion**

  Record each criterion as repository-verified, external-Unreal-pending, or two-machine-pending in the qualification document. Do not describe the system as fully complete while either external category remains pending.

- [ ] **Step 7: Commit documentation and qualification evidence**

  ```bash
  git add docs README.md scripts/run_unreal_tcp_regression.sh
  git commit -m "docs(unreal-tcp): add deployment and qualification handoff"
  ```

## Plan self-review record

- Spec coverage: Tasks 1-4 cover protocol, session, safety, coordinates, bridge, and ROS contracts; Tasks 5-6 cover observed-only maps; Task 7 covers single-goal rolling PlanMotion; Task 8 covers launch/fake server integration; Task 9 covers deployment, regression, and explicit external qualification gates.
- Repository boundary: the Unreal Runtime plugin is intentionally not copied into this Git root; its contract and golden vectors are the repository deliverable.
- Type consistency: `Frame`, `SessionProtocol`, `CoordinateTransform`, `ObservedPatch`, `SparseObservedMap`, and coordinator event names have one producing task and all consumers point backward to it.
- Scope consistency: no task changes frozen ROS interfaces or adds exploration/training behavior.
- Placeholder scan: the plan contains no deferred implementation marker; external-only evidence is named as a qualification gate rather than an implementation omission.
