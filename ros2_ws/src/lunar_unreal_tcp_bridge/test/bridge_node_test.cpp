#include "lunar_unreal_tcp_bridge/bridge_node.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <lunar_navigation_msgs/msg/motion_execution_feedback.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <gtest/gtest.h>

namespace lunar::unreal_tcp {
namespace {

using namespace std::chrono_literals;
constexpr char kSessionId[] = "123e4567-e89b-12d3-a456-426614174000";
constexpr char kCalibrationHash[] = "sha256:bridge-test";
constexpr std::size_t kMapCellCount = 320U * 320U;

class FakeTransport final : public SessionTransport {
 public:
  void SetCallbacks(TransportCallbacks callbacks) override {
    std::scoped_lock lock{mutex_};
    callbacks_ = std::move(callbacks);
  }

  void Start() override { ++start_count; }

  void Stop() noexcept override {
    ++stop_count;
    connected_ = false;
  }

  QueuePushResult Send(
      Frame frame, std::optional<Frame>) override {
    std::scoped_lock lock{mutex_};
    sent_.push_back(std::move(frame));
    return QueuePushResult{
        .accepted = true,
        .replaced_latest = false,
        .fatal = false,
        .reason_code = "ACCEPTED",
    };
  }

  [[nodiscard]] bool connected() const noexcept override {
    return connected_;
  }

  void Connect() {
    connected_ = true;
    auto callbacks = Callbacks();
    if (callbacks.on_connected) {
      callbacks.on_connected();
    }
  }

  void Inject(Frame frame) {
    auto callbacks = Callbacks();
    if (callbacks.on_frame) {
      callbacks.on_frame(std::move(frame));
    }
  }

  void Disconnect(std::string reason) {
    connected_ = false;
    auto callbacks = Callbacks();
    if (callbacks.on_disconnected) {
      callbacks.on_disconnected(std::move(reason));
    }
  }

  [[nodiscard]] std::vector<Frame> Sent() const {
    std::scoped_lock lock{mutex_};
    return sent_;
  }

  std::size_t start_count{};
  std::size_t stop_count{};

 private:
  [[nodiscard]] TransportCallbacks Callbacks() const {
    std::scoped_lock lock{mutex_};
    return callbacks_;
  }

  mutable std::mutex mutex_;
  TransportCallbacks callbacks_;
  std::vector<Frame> sent_;
  bool connected_{};
};

std::vector<double> Covariance(const double diagonal) {
  std::vector<double> result(36U, 0.0);
  for (std::size_t index = 0U; index < 6U; ++index) {
    result[index * 6U + index] = diagonal;
  }
  return result;
}

rclcpp::NodeOptions TestOptions() {
  return rclcpp::NodeOptions{}.parameter_overrides({
      rclcpp::Parameter{"calibration_hash", kCalibrationHash},
      rclcpp::Parameter{"simulation_pose_covariance", Covariance(0.1)},
      rclcpp::Parameter{"simulation_twist_covariance", Covariance(0.2)},
  });
}

struct TestBridge final {
  std::chrono::steady_clock::time_point now{};
  std::shared_ptr<FakeTransport> transport{std::make_shared<FakeTransport>()};
  std::shared_ptr<BridgeNode> node;

  TestBridge() {
    node = std::make_shared<BridgeNode>(
        TestOptions(), BridgeNodeDependencies{
            .transport = transport,
            .steady_now = [this] { return now; },
        });
  }
};

std::vector<double> Identity() {
  return {
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
  };
}

Frame HelloAck() {
  Frame frame;
  frame.header.message_type = MessageType::kHelloAck;
  frame.header.sequence = 1U;
  frame.header.simulation_time_ns = 1'000'000'000LL;
  frame.metadata = {
      {"session_id", kSessionId}, {"scene_id", "unreal-scene"},
      {"robot_id", "wheel-1"}, {"engine_version", "5.0.1"},
      {"agx_plugin_version", "UNKNOWN"},
      {"coordinate_convention", "UE_NATIVE"},
      {"length_unit_to_m", 0.01}, {"handedness", "LEFT"},
      {"up_axis", "Z"}, {"T_base_link_from_unreal_root", Identity()},
      {"T_base_footprint_from_base_link", Identity()},
      {"local_map_width", 320}, {"local_map_height", 320},
      {"resolution_native_cm", 20.0}, {"server_nonce", "server"},
      {"maximum_body_bytes", 8'388'608},
      {"calibration_hash", kCalibrationHash},
  };
  return frame;
}

Frame RobotState(
    const std::uint64_t sequence = 2U,
    const std::int64_t simulation_time = 2'000'000'000LL) {
  Frame frame;
  frame.header.message_type = MessageType::kRobotState;
  frame.header.sequence = sequence;
  frame.header.simulation_time_ns = simulation_time;
  frame.metadata = {
      {"session_id", kSessionId},
      {"position_cm", {100.0, 200.0, 300.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
      {"linear_velocity_cmps", {0.0, 0.0, 0.0}},
      {"angular_velocity_radps", {0.0, 0.0, 0.0}},
      {"control_state", "READY"},
      {"simulation_covariance_profile", "deterministic"},
  };
  return frame;
}

void AppendFloat(std::vector<std::byte>& output, const float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  for (std::size_t offset = 0U; offset < 4U; ++offset) {
    output.push_back(static_cast<std::byte>((bits >> (offset * 8U)) & 0xffU));
  }
}

Frame LocalMap(
    const std::uint64_t sequence = 3U,
    const std::int64_t simulation_time = 2'100'000'000LL) {
  Frame frame;
  frame.header.message_type = MessageType::kLocalElevationMap;
  frame.header.sequence = sequence;
  frame.header.simulation_time_ns = simulation_time;
  const nlohmann::json pose{
      {"position_cm", {0.0, 0.0, 0.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
  };
  frame.metadata = {
      {"session_id", kSessionId}, {"width", 320}, {"height", 320},
      {"resolution_cm", 20.0},
      {"cell_zero_center_world_cm", {0.0, 0.0, 0.0}},
      {"u_axis_world", {1.0, 0.0, 0.0}},
      {"v_axis_world", {0.0, 1.0, 0.0}},
      {"sensor_pose_world", pose}, {"robot_pose_world", pose},
      {"elevation_encoding", "float32_le"},
      {"valid_encoding", "bitset_lsb0"},
  };
  frame.payload.reserve(kMapCellCount * 4U + kMapCellCount / 8U);
  for (std::size_t index = 0U; index < kMapCellCount; ++index) {
    AppendFloat(frame.payload, 100.0F);
  }
  frame.payload.insert(
      frame.payload.end(), kMapCellCount / 8U, std::byte{0xff});
  return frame;
}

lunar_planning_msgs::msg::MotionReference Reference(
    const std::uint8_t platform =
        lunar_planning_msgs::msg::MotionReference::WHEELED) {
  lunar_planning_msgs::msg::MotionReference message;
  message.header.frame_id = "map";
  message.header.stamp.sec = 2;
  message.plan_id = "wheel-plan";
  message.platform_type = platform;
  message.input_time.sec = 2;
  message.trajectory.header.frame_id = "odom";
  trajectory_msgs::msg::MultiDOFJointTrajectoryPoint first;
  geometry_msgs::msg::Transform pose;
  pose.rotation.w = 1.0;
  first.transforms.push_back(pose);
  first.velocities.emplace_back();
  auto second = first;
  second.time_from_start.sec = 1;
  message.trajectory.points = {first, second};
  return message;
}

Frame Feedback() {
  Frame frame;
  frame.header.message_type = MessageType::kExecutionFeedback;
  frame.header.sequence = 4U;
  frame.header.simulation_time_ns = 2'200'000'000LL;
  frame.metadata = {
      {"session_id", kSessionId}, {"platform_type", "WHEELED"},
      {"plan_id", "wheel-plan"}, {"segment_id", "wheel-plan"},
      {"sequence", 1U}, {"state", "ACCEPTED"}, {"reason_code", ""},
  };
  return frame;
}

void ConfigureAndActivate(TestBridge& bridge) {
  ASSERT_EQ(
      bridge.node->configure().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(
      bridge.node->activate().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
}

void Ready(TestBridge& bridge) {
  ConfigureAndActivate(bridge);
  bridge.transport->Connect();
  bridge.node->DrainEventsForTesting();
  bridge.transport->Inject(HelloAck());
  bridge.node->DrainEventsForTesting();
  ASSERT_EQ(bridge.node->session_state_for_testing(), SessionState::kSyncing);
  bridge.transport->Inject(RobotState());
  bridge.node->DrainEventsForTesting();
  ASSERT_EQ(bridge.node->session_state_for_testing(), SessionState::kSyncing);
  bridge.transport->Inject(LocalMap());
  bridge.node->DrainEventsForTesting();
  ASSERT_EQ(bridge.node->session_state_for_testing(), SessionState::kReady);
}

template<typename Predicate>
bool SpinUntil(
    rclcpp::executors::SingleThreadedExecutor& executor,
    Predicate predicate,
    const std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some();
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  executor.spin_some();
  return predicate();
}

class RosContext final : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 1;
      char name[] = "bridge_node_test";
      char* argv[] = {name, nullptr};
      rclcpp::init(argc, argv);
    }
  }

  void TearDown() override {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

const auto* const kRosContext =
    ::testing::AddGlobalTestEnvironment(new RosContext);

TEST(BridgeNode, ConfigureDeclaresFrozenParameters) {
  (void)kRosContext;
  TestBridge bridge;
  EXPECT_EQ(
      bridge.node->configure().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_TRUE(bridge.node->configured_for_testing());
  EXPECT_TRUE(bridge.node->has_parameter("server_host"));
  EXPECT_TRUE(bridge.node->has_parameter("basis_map_from_unreal"));
  EXPECT_TRUE(bridge.node->has_parameter("calibration_hash"));
  EXPECT_EQ(
      bridge.node->get_parameter("calibration_hash").as_string(),
      kCalibrationHash);
  const auto update = bridge.node->set_parameter(
      rclcpp::Parameter{"length_unit_to_m", 0.02});
  EXPECT_FALSE(update.successful);
  EXPECT_EQ(update.reason, "BRIDGE_PARAMETERS_FROZEN_AFTER_CONFIGURE");
  EXPECT_DOUBLE_EQ(
      bridge.node->get_parameter("length_unit_to_m").as_double(), 0.01);
}

TEST(BridgeNode, ActivateStartsTransportAndDeactivateHolds) {
  TestBridge bridge;
  Ready(bridge);
  EXPECT_EQ(bridge.transport->start_count, 1U);

  EXPECT_EQ(
      bridge.node->deactivate().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(bridge.transport->stop_count, 1U);
  const auto sent = bridge.transport->Sent();
  ASSERT_FALSE(sent.empty());
  EXPECT_EQ(sent.back().header.message_type, MessageType::kControl);
  EXPECT_EQ(sent.back().metadata.at("command"), "HOLD");
}

TEST(BridgeNode, HandshakeRequiresStateAndMapBeforeReady) {
  TestBridge bridge;
  ConfigureAndActivate(bridge);
  bridge.transport->Connect();
  bridge.node->DrainEventsForTesting();
  const auto sent = bridge.transport->Sent();
  ASSERT_FALSE(sent.empty());
  EXPECT_EQ(sent.front().header.message_type, MessageType::kHello);

  bridge.transport->Inject(HelloAck());
  bridge.node->DrainEventsForTesting();
  EXPECT_EQ(bridge.node->session_state_for_testing(), SessionState::kSyncing);
  bridge.transport->Inject(RobotState());
  bridge.node->DrainEventsForTesting();
  EXPECT_EQ(bridge.node->session_state_for_testing(), SessionState::kSyncing);
  bridge.transport->Inject(LocalMap());
  bridge.node->DrainEventsForTesting();
  EXPECT_EQ(bridge.node->session_state_for_testing(), SessionState::kReady);
}

TEST(BridgeNode, PublishesClockOdometryTfAndObservedMap) {
  TestBridge bridge;
  Ready(bridge);
  auto observer = std::make_shared<rclcpp::Node>("bridge_publication_observer");
  std::optional<rosgraph_msgs::msg::Clock> clock;
  std::optional<nav_msgs::msg::Odometry> odometry;
  std::optional<grid_map_msgs::msg::GridMap> map;
  std::optional<tf2_msgs::msg::TFMessage> transforms;
  const auto clock_sub = observer->create_subscription<rosgraph_msgs::msg::Clock>(
      "/clock", rclcpp::QoS{10}.best_effort(),
      [&](const rosgraph_msgs::msg::Clock& value) { clock = value; });
  const auto odom_sub = observer->create_subscription<nav_msgs::msg::Odometry>(
      "/localization/odometry", 10,
      [&](const nav_msgs::msg::Odometry& value) { odometry = value; });
  const auto map_sub = observer->create_subscription<grid_map_msgs::msg::GridMap>(
      "/lunar/unreal/observed_elevation", 10,
      [&](const grid_map_msgs::msg::GridMap& value) { map = value; });
  const auto tf_sub = observer->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", 10,
      [&](const tf2_msgs::msg::TFMessage& value) { transforms = value; });
  (void)clock_sub;
  (void)odom_sub;
  (void)map_sub;
  (void)tf_sub;
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(observer);
  executor.add_node(bridge.node->get_node_base_interface());
  (void)SpinUntil(executor, [&] {
    return bridge.node->count_publishers("/clock") > 0U;
  }, 200ms);

  bridge.transport->Inject(RobotState(4U, 3'000'000'000LL));
  bridge.node->DrainEventsForTesting();
  bridge.transport->Inject(LocalMap(5U, 3'100'000'000LL));
  bridge.node->DrainEventsForTesting();
  ASSERT_TRUE(SpinUntil(executor, [&] {
    return clock.has_value() && odometry.has_value() && map.has_value() &&
        transforms.has_value();
  }));
  EXPECT_EQ(clock->clock.sec, 3);
  EXPECT_EQ(odometry->child_frame_id, "base_link");
  EXPECT_EQ(map->layers.size(), 2U);
  EXPECT_EQ(transforms->transforms.size(), 2U);
  executor.remove_node(bridge.node->get_node_base_interface());
  executor.remove_node(observer);
}

TEST(BridgeNode, ForwardsOnlyValidWheeledReference) {
  TestBridge bridge;
  Ready(bridge);
  const std::size_t before = bridge.transport->Sent().size();
  bridge.node->ReceiveMotionReferenceForTesting(Reference());
  const auto after_valid = bridge.transport->Sent();
  ASSERT_GT(after_valid.size(), before);
  EXPECT_EQ(
      after_valid.back().header.message_type,
      MessageType::kMotionReference);

  bridge.node->ReceiveMotionReferenceForTesting(
      Reference(lunar_planning_msgs::msg::MotionReference::LEGGED));
  const auto after_invalid = bridge.transport->Sent();
  EXPECT_EQ(
      std::count_if(
          after_invalid.begin(), after_invalid.end(),
          [](const Frame& frame) {
            return frame.header.message_type == MessageType::kMotionReference;
          }),
      1);
  EXPECT_EQ(after_invalid.back().header.message_type, MessageType::kControl);
  EXPECT_EQ(after_invalid.back().metadata.at("command"), "HOLD");
}

TEST(BridgeNode, MatchingFeedbackPublishesExternalContract) {
  TestBridge bridge;
  Ready(bridge);
  bridge.node->ReceiveMotionReferenceForTesting(Reference());
  auto observer = std::make_shared<rclcpp::Node>("bridge_feedback_observer");
  std::optional<lunar_navigation_msgs::msg::MotionExecutionFeedback> feedback;
  const auto subscription = observer->create_subscription<
      lunar_navigation_msgs::msg::MotionExecutionFeedback>(
      "/execution/motion_feedback", 10,
      [&](const lunar_navigation_msgs::msg::MotionExecutionFeedback& value) {
        feedback = value;
      });
  (void)subscription;
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(observer);
  executor.add_node(bridge.node->get_node_base_interface());
  executor.spin_some();
  bridge.transport->Inject(Feedback());
  bridge.node->DrainEventsForTesting();
  ASSERT_TRUE(SpinUntil(executor, [&] { return feedback.has_value(); }));
  EXPECT_EQ(feedback->platform_type, feedback->WHEELED);
  EXPECT_EQ(feedback->plan_id, "wheel-plan");
  EXPECT_EQ(feedback->segment_id, "wheel-plan");
  EXPECT_EQ(feedback->header.frame_id, "base_footprint");
  executor.remove_node(bridge.node->get_node_base_interface());
  executor.remove_node(observer);
}

TEST(BridgeNode, HeartbeatOrFreshnessExpiryFailsClosed) {
  TestBridge bridge;
  Ready(bridge);
  bridge.now += 1001ms;
  bridge.node->CheckWatchdogsForTesting();
  EXPECT_EQ(bridge.node->session_state_for_testing(), SessionState::kHold);
  EXPECT_EQ(bridge.node->last_reason_for_testing(), "STATE_OR_MAP_STALE");
  const auto sent = bridge.transport->Sent();
  ASSERT_FALSE(sent.empty());
  EXPECT_EQ(sent.back().header.message_type, MessageType::kControl);
  EXPECT_EQ(sent.back().metadata.at("command"), "HOLD");
}

TEST(BridgeNode, MissingInitialStateOrMapExpiresFromHandshake) {
  TestBridge bridge;
  ConfigureAndActivate(bridge);
  bridge.transport->Connect();
  bridge.node->DrainEventsForTesting();
  bridge.transport->Inject(HelloAck());
  bridge.node->DrainEventsForTesting();

  bridge.now += 1001ms;
  bridge.node->CheckWatchdogsForTesting();

  EXPECT_EQ(bridge.node->session_state_for_testing(), SessionState::kHold);
  EXPECT_EQ(bridge.node->last_reason_for_testing(), "STATE_OR_MAP_STALE");
}

TEST(BridgeNode, ReliableInboundQueueOverflowFailsClosed) {
  TestBridge bridge;
  Ready(bridge);
  for (std::uint64_t sequence = 4U; sequence < 71U; ++sequence) {
    Frame feedback = Feedback();
    feedback.header.sequence = sequence;
    feedback.metadata["sequence"] = sequence - 3U;
    bridge.transport->Inject(std::move(feedback));
  }

  bridge.node->DrainEventsForTesting();

  EXPECT_EQ(bridge.node->session_state_for_testing(), SessionState::kHold);
  EXPECT_EQ(
      bridge.node->last_reason_for_testing(),
      "INBOUND_RELIABLE_QUEUE_EXHAUSTED");
}

}  // namespace
}  // namespace lunar::unreal_tcp
