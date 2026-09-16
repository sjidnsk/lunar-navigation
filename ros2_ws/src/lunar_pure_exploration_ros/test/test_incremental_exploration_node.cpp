#include "lunar_pure_exploration_ros/incremental_exploration_node.hpp"

#include <algorithm>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using Action = lunar_planning_msgs::action::NavigateToPose;
using ServerGoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;
using Task = lunar_pure_exploration_msgs::msg::PureExplorationTask;
using namespace std::chrono_literals;

class RosEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 0;
      rclcpp::init(argc, nullptr);
    }
  }
  void TearDown() override {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

const auto* const kRosEnvironment =
    ::testing::AddGlobalTestEnvironment(new RosEnvironment{});

template <typename Predicate>
bool WaitFor(Predicate&& predicate,
             const std::chrono::milliseconds timeout = 4s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

geometry_msgs::msg::Quaternion YawQuaternion(double yaw) {
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.z = std::sin(yaw / 2.0);
  quaternion.w = std::cos(yaw / 2.0);
  return quaternion;
}

Task StartTask(const std::string& task_id = "incremental-task") {
  Task task;
  task.header.frame_id = "map";
  task.task_id = task_id;
  task.command = Task::START;
  for (const auto [x, y] :
       {std::pair{0.0F, 0.0F}, std::pair{12.0F, 0.0F},
        std::pair{12.0F, 12.0F}, std::pair{0.0F, 12.0F}}) {
    geometry_msgs::msg::Point32 point;
    point.x = x;
    point.y = y;
    task.boundary.points.push_back(point);
  }
  return task;
}

Task Command(std::uint8_t command) {
  Task task;
  task.header.frame_id = "map";
  task.task_id = "incremental-task";
  task.command = command;
  return task;
}

nav_msgs::msg::OccupancyGrid ExplorationMap(bool broad_frontier = true) {
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.info.width = 12U;
  map.info.height = 12U;
  map.info.resolution = 1.0F;
  map.info.origin.orientation.w = 1.0;
  map.data.assign(144U, 0);
  map.data.front() = 100;
  if (broad_frontier) {
    for (std::uint32_t y = 0U; y < map.info.height; ++y) {
      for (std::uint32_t x = 7U; x < map.info.width; ++x) {
        map.data[static_cast<std::size_t>(y) * map.info.width + x] = -1;
      }
    }
  } else {
    map.data[6U * map.info.width + 7U] = -1;
  }
  return map;
}

nav_msgs::msg::OccupancyGrid ExplorationMapWithUnknownRobotCell() {
  auto map = ExplorationMap(false);
  constexpr std::uint32_t kRobotCellX = 2U;
  constexpr std::uint32_t kRobotCellY = 2U;
  map.data.at(static_cast<std::size_t>(kRobotCellY) * map.info.width +
              kRobotCellX) = -1;
  return map;
}

nav_msgs::msg::OccupancyGrid FullyOccupiedExplorationMap() {
  auto map = ExplorationMap(false);
  std::fill(map.data.begin(), map.data.end(), std::int8_t{100});
  return map;
}

nav_msgs::msg::OccupancyGrid FinerExplorationMap() {
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.info.width = 24U;
  map.info.height = 24U;
  map.info.resolution = 0.5F;
  map.info.origin.orientation.w = 1.0;
  map.data.assign(576U, 0);
  for (std::uint32_t y = 0U; y < map.info.height; ++y) {
    for (std::uint32_t x = 14U; x < map.info.width; ++x) {
      map.data[static_cast<std::size_t>(y) * map.info.width + x] = -1;
    }
  }
  return map;
}

nav_msgs::msg::Odometry Odometry() {
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.x = 2.5;
  odometry.pose.pose.position.y = 2.5;
  odometry.pose.pose.orientation.w = 1.0;
  return odometry;
}

tf2_msgs::msg::TFMessage MapFromOdom() {
  tf2_msgs::msg::TFMessage transforms;
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.rotation.w = 1.0;
  transforms.transforms.push_back(transform);
  return transforms;
}

bool SameNavigationPose(const Action::Goal& left, const Action::Goal& right) {
  return left.target_x_m == right.target_x_m &&
         left.target_y_m == right.target_y_m &&
         left.target_yaw_rad == right.target_yaw_rad;
}

class FakeNavigationServer final {
 public:
  FakeNavigationServer(rclcpp::Node& node, const std::string& action_name) {
    server_ = rclcpp_action::create_server<Action>(
        &node, action_name,
        [this](const rclcpp_action::GoalUUID&,
               const std::shared_ptr<const Action::Goal> goal) {
          {
            std::scoped_lock lock{mutex_};
            goals_.push_back(*goal);
          }
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<ServerGoalHandle>) {
          ++cancel_count_;
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          std::scoped_lock lock{mutex_};
          active_ = handle;
        });
  }

  std::size_t goal_count() const {
    std::scoped_lock lock{mutex_};
    return goals_.size();
  }
  std::size_t cancel_count() const { return cancel_count_.load(); }
  Action::Goal goal(std::size_t index) const {
    std::scoped_lock lock{mutex_};
    return goals_.at(index);
  }

  void PublishFeedback(std::uint8_t state, const std::string& reason) {
    ASSERT_TRUE(WaitFor([this] {
      std::scoped_lock lock{mutex_};
      return active_ != nullptr;
    }));
    std::shared_ptr<ServerGoalHandle> active;
    {
      std::scoped_lock lock{mutex_};
      active = active_;
    }
    auto feedback = std::make_shared<Action::Feedback>();
    feedback->session_state = state;
    feedback->reason_code = reason;
    active->publish_feedback(feedback);
  }

  void Finish(std::uint8_t outcome, const std::string& reason_code) {
    ASSERT_TRUE(WaitFor([this] {
      std::scoped_lock lock{mutex_};
      return active_ != nullptr;
    }));
    std::shared_ptr<ServerGoalHandle> active;
    {
      std::scoped_lock lock{mutex_};
      active = std::move(active_);
    }
    ASSERT_NE(active, nullptr);
    auto result = std::make_shared<Action::Result>();
    result->outcome = outcome;
    result->reason_code = reason_code;
    if (outcome == Action::Result::GOAL_REACHED) {
      active->succeed(result);
    } else if (outcome == Action::Result::CANCELED) {
      active->canceled(result);
    } else {
      active->abort(result);
    }
  }

 private:
  mutable std::mutex mutex_;
  std::vector<Action::Goal> goals_;
  std::shared_ptr<ServerGoalHandle> active_;
  std::atomic<std::size_t> cancel_count_{0U};
  rclcpp_action::Server<Action>::SharedPtr server_;
};

class IncrementalExplorationNodeTest : public ::testing::Test {
 protected:
  virtual double MapWaitTimeoutSeconds() const { return -1.0; }
  virtual bool UseSimTime() const { return false; }
  virtual void ConfigureParameters(IncrementalExplorationNodeParameters&) {}

  void SetUp() override {
    const auto id = next_id_.fetch_add(1U);
    const std::string prefix = "/incremental_test_" + std::to_string(id);
    parameters_ = Parameters(prefix);
    ConfigureParameters(parameters_);
    server_node_ = std::make_shared<rclcpp::Node>(
        "incremental_navigation_server_" + std::to_string(id));
    observer_ = std::make_shared<rclcpp::Node>(
        "incremental_exploration_observer_" + std::to_string(id));
    server_ = std::make_unique<FakeNavigationServer>(
        *server_node_, parameters_.navigation_action);
    std::vector<rclcpp::Parameter> node_parameters{{"use_sim_time", UseSimTime()}};
    if (MapWaitTimeoutSeconds() >= 0.0) {
      node_parameters.emplace_back("navigation_map_wait_timeout_s",
                                   MapWaitTimeoutSeconds());
    }
    node_ = std::make_shared<IncrementalExplorationNode>(
        parameters_, rclcpp::NodeOptions{}.parameter_overrides(node_parameters));

    task_publisher_ = observer_->create_publisher<Task>(
        parameters_.task_topic, rclcpp::QoS{10}.reliable());
    map_publisher_ = observer_->create_publisher<nav_msgs::msg::OccupancyGrid>(
        parameters_.exploration_map_topic,
        rclcpp::QoS{1}.reliable().transient_local());
    odometry_publisher_ = observer_->create_publisher<nav_msgs::msg::Odometry>(
        parameters_.odometry_topic, rclcpp::QoS{1}.reliable());
    tf_publisher_ = observer_->create_publisher<tf2_msgs::msg::TFMessage>(
        parameters_.tf_topic, rclcpp::QoS{10}.reliable());
    status_subscription_ = observer_->create_subscription<Status>(
        parameters_.status_topic, rclcpp::QoS{10}.reliable().transient_local(),
        [this](Status::SharedPtr status) {
          std::scoped_lock lock{status_mutex_};
          last_status_ = *status;
          statuses_.push_back(*status);
        });
    diagnostics_subscription_ = observer_->create_subscription<
        diagnostic_msgs::msg::DiagnosticArray>(
        parameters_.diagnostics_topic, rclcpp::QoS{1}.reliable().transient_local(),
        [this](diagnostic_msgs::msg::DiagnosticArray::SharedPtr diagnostics) {
          for (const auto& status : diagnostics->status) {
            std::uint64_t sequence = 0U;
            for (const auto& value : status.values) {
              if (value.key == "exploration_map_sequence") {
                sequence = std::stoull(value.value);
              } else if (value.key == "successful_candidate_count") {
                successful_candidate_count_.store(std::stoull(value.value));
              } else if (value.key == "recent_observation_memory_count") {
                observation_memory_count_.store(std::stoull(value.value));
              } else if (value.key == "recent_observation_memory_patch_cells") {
                observation_patch_cells_.store(std::stoull(value.value));
              } else if (value.key == "recent_observation_memory_eviction_count") {
                observation_evictions_.store(std::stoull(value.value));
              } else if (value.key == "recent_observation_memory_invalidated_count") {
                observation_invalidations_.store(std::stoull(value.value));
              } else if (value.key == "eligible_candidate_count") {
                eligible_candidate_count_.store(std::stoull(value.value));
              } else if (value.key == "suppressed_candidate_count") {
                suppressed_candidate_count_.store(std::stoull(value.value));
              }
            }
            map_sequence_.store(sequence);
          }
        });
    task_map_subscription_ = observer_->create_subscription<
        visualization_msgs::msg::MarkerArray>(
        parameters_.task_map_markers_topic,
        rclcpp::QoS{1}.reliable().transient_local(),
        [this](visualization_msgs::msg::MarkerArray::SharedPtr markers) {
          std::scoped_lock lock{task_map_mutex_};
          last_task_map_ = *markers;
          ++task_map_message_count_;
        });
    frontiers_subscription_ = observer_->create_subscription<
        visualization_msgs::msg::MarkerArray>(
        parameters_.frontiers_topic, rclcpp::QoS{1}.reliable().transient_local(),
        [this](visualization_msgs::msg::MarkerArray::SharedPtr markers) {
          std::scoped_lock lock{task_map_mutex_};
          frontier_messages_.push_back(*markers);
        });

    executor_.add_node(server_node_);
    executor_.add_node(observer_);
    executor_.add_node(node_);
    spin_ = std::jthread([this] { executor_.spin(); });
    ASSERT_TRUE(WaitFor([this] {
      return task_publisher_->get_subscription_count() == 1U &&
             map_publisher_->get_subscription_count() == 1U &&
             odometry_publisher_->get_subscription_count() == 1U &&
             tf_publisher_->get_subscription_count() == 1U;
    }));
  }

  void TearDown() override {
    executor_.cancel();
    spin_.join();
    executor_.remove_node(node_);
    executor_.remove_node(observer_);
    executor_.remove_node(server_node_);
    node_.reset();
    server_.reset();
  }

  static IncrementalExplorationNodeParameters Parameters(
      const std::string& prefix) {
    return {
        .platform = {.platform_id = "test-wheel",
                     .platform_type = "WHEELED",
                     .base_frame_id = "base_link",
                     .footprint_vertices = {{-0.2, -0.2}, {0.2, -0.2},
                                            {0.2, 0.2}, {-0.2, 0.2}},
                     .minimum_clearance_m = 0.0},
        .candidate_parameters =
            {{-std::numbers::pi / 4.0, -std::numbers::pi / 8.0, 0.0,
              std::numbers::pi / 8.0, std::numbers::pi / 4.0}},
        .candidate_limits = {100000U, 10000U, 1000000U},
        .task_raster_limits = {100000U},
        .sensor_model = {5.0, std::numbers::pi / 2.0},
        .information_gain_limits = {1000000U},
        .score_weights = {},
        .failure_memory_limits = {100U, 10000U, 100000U},
        .minimum_frontier_length_m = 1.0,
        .coverage_target = 1.0,
        .exploration_map_topic = prefix + "/exploration_map",
        .odometry_topic = prefix + "/odometry",
        .tf_topic = prefix + "/tf",
        .task_topic = prefix + "/task",
        .navigation_action = prefix + "/navigate_to_pose",
        .status_topic = prefix + "/status",
        .task_boundary_topic = prefix + "/task_boundary",
        .task_map_markers_topic = prefix + "/task_map_markers",
        .current_goal_topic = prefix + "/current_goal",
        .frontiers_topic = prefix + "/frontiers",
        .diagnostics_topic = prefix + "/diagnostics",
    };
  }

  std::optional<Status> LastStatus() const {
    std::scoped_lock lock{status_mutex_};
    return last_status_;
  }

  std::size_t StatusCount() const {
    std::scoped_lock lock{status_mutex_};
    return statuses_.size();
  }

  bool HasStatusReasonPrefixSince(const std::size_t first,
                                  const std::string_view prefix) const {
    std::scoped_lock lock{status_mutex_};
    return std::any_of(statuses_.begin() + std::min(first, statuses_.size()),
                       statuses_.end(), [prefix](const Status& status) {
                         return status.reason_code.rfind(prefix, 0U) == 0U;
                       });
  }

  std::optional<visualization_msgs::msg::MarkerArray> LastTaskMap() const {
    std::scoped_lock lock{task_map_mutex_};
    return last_task_map_;
  }

  std::size_t TaskMapMessageCount() const {
    std::scoped_lock lock{task_map_mutex_};
    return task_map_message_count_;
  }

  std::vector<visualization_msgs::msg::MarkerArray> FrontierMessages() const {
    std::scoped_lock lock{task_map_mutex_};
    return frontier_messages_;
  }

  void PublishPoseAndMap(bool broad_frontier = true) {
    map_publisher_->publish(ExplorationMap(broad_frontier));
    odometry_publisher_->publish(Odometry());
    tf_publisher_->publish(MapFromOdom());
  }

  void Start(bool broad_frontier = true) {
    PublishPoseAndMap(broad_frontier);
    task_publisher_->publish(StartTask());
    ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 1U; }));
    ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 1U; }));
  }

  std::size_t CompleteCandidatesOnUnchangedMap() {
    std::vector<Action::Goal> completed;
    for (std::size_t index = 0U; index < 100U; ++index) {
      const auto goal = server_->goal(index);
      EXPECT_TRUE(std::none_of(completed.begin(), completed.end(),
                               [&goal](const auto& previous) {
                                 return SameNavigationPose(previous, goal);
                               }));
      completed.push_back(goal);
      server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
      if (!WaitFor([this, index] {
            const auto status = LastStatus();
            return server_->goal_count() > index + 1U ||
                   (status && status->reason_code == "WAITING_FOR_MAP_CHANGE");
          })) {
        ADD_FAILURE() << "successful candidate did not advance or wait for map change";
        return 0U;
      }
      const auto status = LastStatus();
      if (status && status->reason_code == "WAITING_FOR_MAP_CHANGE") {
        EXPECT_EQ(status->state, Status::WAITING_FOR_INPUT);
        EXPECT_EQ(status->completed_goal_count, completed.size());
        EXPECT_EQ(status->failed_candidate_count, 0U);
        EXPECT_EQ(status->reachable_candidate_count, 0U);
        EXPECT_GT(status->frontier_cluster_count, 0U);
        EXPECT_LT(status->coverage_ratio, 1.0);
        EXPECT_EQ(server_->goal_count(), completed.size());
        EXPECT_TRUE(std::any_of(completed.begin(), completed.end(),
                                [&completed](const auto& left) {
                                  return std::any_of(
                                      completed.begin(), completed.end(),
                                      [&left](const auto& right) {
                                        return left.target_x_m == right.target_x_m &&
                                               left.target_y_m == right.target_y_m &&
                                               left.target_yaw_rad != right.target_yaw_rad;
                                      });
                                })) << "different yaw views at one position must remain eligible";
        return completed.size();
      }
    }
    ADD_FAILURE() << "unchanged map generated more than 100 successful goals";
    return 0U;
  }

  inline static std::atomic<std::uint64_t> next_id_{0U};
  IncrementalExplorationNodeParameters parameters_;
  std::shared_ptr<rclcpp::Node> server_node_;
  std::shared_ptr<rclcpp::Node> observer_;
  std::shared_ptr<IncrementalExplorationNode> node_;
  std::unique_ptr<FakeNavigationServer> server_;
  rclcpp::Publisher<Task>::SharedPtr task_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_publisher_;
  rclcpp::Subscription<Status>::SharedPtr status_subscription_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_subscription_;
  std::atomic<std::uint64_t> map_sequence_{0U};
  std::atomic<std::size_t> successful_candidate_count_{0U};
  std::atomic<std::size_t> observation_memory_count_{0U};
  std::atomic<std::size_t> observation_patch_cells_{0U};
  std::atomic<std::size_t> observation_evictions_{0U};
  std::atomic<std::size_t> observation_invalidations_{0U};
  std::atomic<std::size_t> eligible_candidate_count_{0U};
  std::atomic<std::size_t> suppressed_candidate_count_{0U};
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr
      task_map_subscription_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr
      frontiers_subscription_;
  mutable std::mutex status_mutex_;
  std::optional<Status> last_status_;
  std::vector<Status> statuses_;
  mutable std::mutex task_map_mutex_;
  std::optional<visualization_msgs::msg::MarkerArray> last_task_map_;
  std::vector<visualization_msgs::msg::MarkerArray> frontier_messages_;
  std::size_t task_map_message_count_{0U};
  rclcpp::executors::MultiThreadedExecutor executor_;
  std::jthread spin_;
};

class IncrementalExplorationMapWaitTest : public IncrementalExplorationNodeTest {
 protected:
  double MapWaitTimeoutSeconds() const override { return 45.0; }
  bool UseSimTime() const override { return true; }

  void SetUp() override {
    IncrementalExplorationNodeTest::SetUp();
    clock_publisher_ = observer_->create_publisher<rosgraph_msgs::msg::Clock>(
        "/clock", rclcpp::ClockQoS{});
    ASSERT_TRUE(WaitFor([this] {
      return clock_publisher_->get_subscription_count() == 1U;
    }));
    SetClock(100);
  }

  void SetClock(std::int32_t seconds) {
    rosgraph_msgs::msg::Clock clock;
    clock.clock.sec = seconds;
    clock_publisher_->publish(clock);
    ASSERT_TRUE(WaitFor([this, seconds] {
      return node_->now().seconds() == static_cast<double>(seconds);
    }));
  }

  void WaitForMapFeedback() {
    const auto before_feedback = StatusCount();
    server_->PublishFeedback(Action::Feedback::PLANNING, "WAITING_FOR_MAP");
    ASSERT_TRUE(WaitFor([this, before_feedback] {
      const auto status = LastStatus();
      return StatusCount() > before_feedback && status &&
             status->reason_code == "NAVIGATION_WAITING_FOR_MAP";
    }));
  }

  void TriggerMapWaitTimeout() {
    WaitForMapFeedback();
    SetClock(145);
    ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
  }

  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_publisher_;
};

class IncrementalExplorationDisabledMapWaitTest
    : public IncrementalExplorationMapWaitTest {
 protected:
  double MapWaitTimeoutSeconds() const override { return 0.0; }
};

class IncrementalExplorationDefaultMapWaitTest
    : public IncrementalExplorationMapWaitTest {
 protected:
  double MapWaitTimeoutSeconds() const override { return -1.0; }
};

class OmnidirectionalExplorationTest : public IncrementalExplorationNodeTest {
 protected:
  void ConfigureParameters(IncrementalExplorationNodeParameters& p) override {
    p.sensor_model.field_of_view_rad = 2.0 * std::numbers::pi;
  }
};

TEST_F(OmnidirectionalExplorationTest, RequiresTerminalViewingYaw) {
  Start();
  EXPECT_TRUE(server_->goal(0).has_target_yaw);
}

TEST_F(IncrementalExplorationNodeTest, DirectionalSensorRequiresTerminalViewingYaw) {
  Start();
  EXPECT_TRUE(server_->goal(0).has_target_yaw);
}

TEST_F(IncrementalExplorationDefaultMapWaitTest,
       DefaultThirtySecondsCancelsOnlyAtDeadline) {
  EXPECT_DOUBLE_EQ(node_->get_parameter("navigation_map_wait_timeout_s").as_double(), 30.0);
  Start(false);
  WaitForMapFeedback();
  SetClock(129);
  EXPECT_FALSE(WaitFor([this] { return server_->cancel_count() != 0U; }, 100ms));
  SetClock(130);
  ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
  EXPECT_EQ(server_->goal_count(), 1U);
  EXPECT_EQ(LastStatus()->completed_goal_count, 0U);
}

TEST_F(IncrementalExplorationNodeTest, RejectsInvalidMapWaitTimeout) {
  for (const double timeout : {-1.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(std::make_shared<IncrementalExplorationNode>(
                     parameters_, rclcpp::NodeOptions{}.enable_rosout(false).parameter_overrides({
                                      {"navigation_map_wait_timeout_s", timeout}})),
                 std::invalid_argument);
  }
}

TEST_F(IncrementalExplorationDisabledMapWaitTest,
       ExplicitZeroDisablesTimeoutCancellation) {
  Start(false);
  WaitForMapFeedback();
  SetClock(10000);
  EXPECT_FALSE(WaitFor([this] { return server_->cancel_count() != 0U; }, 150ms));
  EXPECT_EQ(LastStatus()->reason_code, "NAVIGATION_WAITING_FOR_MAP");
  EXPECT_EQ(server_->goal_count(), 1U);
}

TEST_F(IncrementalExplorationDisabledMapWaitTest,
       FrontierAddsAndDeletesUseMapFrameAndCurrentSimulationStamp) {
  using Marker = visualization_msgs::msg::Marker;
  Start(false);
  ASSERT_TRUE(WaitFor([this] {
    const auto messages = FrontierMessages();
    return std::any_of(messages.begin(), messages.end(), [](const auto& message) {
      return !message.markers.empty();
    });
  }));
  SetClock(101);
  task_publisher_->publish(Command(Task::CANCEL));
  ASSERT_TRUE(WaitFor([this] {
    const auto messages = FrontierMessages();
    return std::any_of(messages.begin(), messages.end(), [](const auto& message) {
      return std::any_of(message.markers.begin(), message.markers.end(),
                         [](const auto& marker) { return marker.action == Marker::DELETE; });
    });
  }));
  for (const auto& message : FrontierMessages()) {
    for (const auto& marker : message.markers) {
      EXPECT_EQ(marker.header.frame_id, "map");
      EXPECT_EQ(marker.header.stamp.sec, marker.action == Marker::DELETE ? 101 : 100);
      EXPECT_EQ(marker.header.stamp.nanosec, 0U);
    }
  }
}

TEST_F(IncrementalExplorationMapWaitTest,
       TimeoutWaitsForCanceledTerminalBeforeSuppressingAndReselecting) {
  Start(false);
  const auto first = server_->goal(0U);
  WaitForMapFeedback();
  SetClock(144);
  EXPECT_FALSE(WaitFor([this] { return server_->cancel_count() != 0U; }, 100ms));
  SetClock(145);
  ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
  ASSERT_TRUE(WaitFor([this] {
    return LastStatus()->reason_code == "NAVIGATION_MAP_WAIT_TIMEOUT_CANCELING";
  }));
  server_->PublishFeedback(Action::Feedback::EXECUTING, "EXECUTING");
  SetClock(1000);
  EXPECT_FALSE(WaitFor([this] { return server_->goal_count() != 1U; }, 100ms));
  EXPECT_EQ(LastStatus()->reason_code, "NAVIGATION_MAP_WAIT_TIMEOUT_CANCELING");
  EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
  EXPECT_EQ(LastStatus()->completed_goal_count, 0U);
  const auto before_terminal = StatusCount();
  server_->Finish(Action::Result::CANCELED, "CANCELED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_FALSE(SameNavigationPose(first, server_->goal(1U)));
  EXPECT_EQ(LastStatus()->failed_candidate_count, 1U);
  EXPECT_EQ(LastStatus()->completed_goal_count, 0U);
  EXPECT_TRUE(HasStatusReasonPrefixSince(before_terminal,
                                        "NAVIGATION_MAP_WAIT_TIMEOUT"));
}

TEST_F(IncrementalExplorationMapWaitTest,
       IdenticalMapReceiptsAndRepeatedWaitingFeedbackDoNotResetTimeout) {
  Start(false);
  WaitForMapFeedback();
  SetClock(140);
  auto map = ExplorationMap(false);
  map.header.stamp.sec = 140;
  map_publisher_->publish(map);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  WaitForMapFeedback();
  SetClock(145);
  EXPECT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
}

TEST_F(IncrementalExplorationMapWaitTest,
       ChangedMapEvidenceAndExecutingFeedbackResetContinuousWait) {
  Start(false);
  WaitForMapFeedback();
  SetClock(140);
  auto map = ExplorationMap(false);
  map.data.front() = 0;
  map_publisher_->publish(map);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  SetClock(145);
  EXPECT_FALSE(WaitFor([this] { return server_->cancel_count() != 0U; }, 100ms));
  SetClock(170);
  server_->PublishFeedback(Action::Feedback::EXECUTING, "EXECUTING");
  ASSERT_TRUE(WaitFor([this] {
    return LastStatus()->reason_code == "NAVIGATION_EXECUTING";
  }));
  SetClock(300);
  EXPECT_FALSE(WaitFor([this] { return server_->cancel_count() != 0U; }, 100ms));
  WaitForMapFeedback();
  SetClock(345);
  EXPECT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
}

TEST_F(IncrementalExplorationMapWaitTest,
       PauseAndResumeDoNotConvertTheirPendingCancellationIntoMapTimeout) {
  Start(false);
  const auto first = server_->goal(0U);
  WaitForMapFeedback();
  SetClock(120);
  task_publisher_->publish(Command(Task::PAUSE));
  ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
  SetClock(300);
  task_publisher_->publish(Command(Task::RESUME));
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->reason_code == "RESUMED"; }));
  server_->PublishFeedback(Action::Feedback::PLANNING, "WAITING_FOR_MAP");
  SetClock(600);
  EXPECT_FALSE(WaitFor([this] { return server_->goal_count() != 1U; }, 100ms));
  server_->Finish(Action::Result::CANCELED, "CANCELED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_TRUE(SameNavigationPose(first, server_->goal(1U)));
  EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
}

TEST_F(IncrementalExplorationMapWaitTest,
       ReplacementTaskDoesNotInheritPendingTimeoutFailure) {
  Start(false);
  const auto first = server_->goal(0U);
  TriggerMapWaitTimeout();
  task_publisher_->publish(StartTask("replacement"));
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->task_id == "replacement"; }));
  EXPECT_EQ(server_->goal_count(), 1U);
  server_->Finish(Action::Result::CANCELED, "CANCELED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_TRUE(SameNavigationPose(first, server_->goal(1U)));
  EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
  EXPECT_EQ(LastStatus()->completed_goal_count, 0U);
}

TEST_F(IncrementalExplorationMapWaitTest,
       MapChangeDuringCancellationDoesNotRecordObsoleteTimeoutEvidence) {
  Start(false);
  const auto first = server_->goal(0U);
  TriggerMapWaitTimeout();
  auto map = ExplorationMap(false);
  map.data.front() = 0;
  map_publisher_->publish(map);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  server_->Finish(Action::Result::CANCELED, "CANCELED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_TRUE(SameNavigationPose(first, server_->goal(1U)));
  EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
}

TEST_F(IncrementalExplorationMapWaitTest,
       InternalErrorDuringTimeoutCancellationIsNotConvertedToRetry) {
  Start(false);
  TriggerMapWaitTimeout();
  server_->Finish(Action::Result::INTERNAL_ERROR, "TEST_NAVIGATION_INTERNAL_ERROR");
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->state == Status::ERROR; }));
  EXPECT_EQ(LastStatus()->reason_code, "TEST_NAVIGATION_INTERNAL_ERROR");
  EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
  EXPECT_EQ(server_->goal_count(), 1U);
}

TEST_F(IncrementalExplorationMapWaitTest,
       GoalReachedDuringTimeoutCancellationRemainsActualSuccess) {
  Start(false);
  TriggerMapWaitTimeout();
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_EQ(LastStatus()->completed_goal_count, 1U);
  EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
}

TEST_F(IncrementalExplorationMapWaitTest,
       ActualNoPathDuringPausedTimeoutCancellationKeepsPauseAndFailure) {
  Start(false);
  const auto first = server_->goal(0U);
  TriggerMapWaitTimeout();
  task_publisher_->publish(Command(Task::PAUSE));
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->state == Status::PAUSED; }));
  server_->Finish(Action::Result::NO_PATH, "NO_PATH");
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->failed_candidate_count == 1U; }));
  ASSERT_EQ(LastStatus()->state, Status::PAUSED);
  EXPECT_EQ(LastStatus()->completed_goal_count, 0U);
  EXPECT_EQ(server_->goal_count(), 1U);
  task_publisher_->publish(Command(Task::RESUME));
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_FALSE(SameNavigationPose(first, server_->goal(1U)));
}

TEST_F(IncrementalExplorationNodeTest, StartWithoutMapWaitsForInput) {
  task_publisher_->publish(StartTask());
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->state == Status::WAITING_FOR_INPUT;
  }));
  EXPECT_EQ(server_->goal_count(), 0U);
}

TEST_F(IncrementalExplorationNodeTest,
       RejectsEveryValueOutsideThePublishedThreeStateExplorationMap) {
  task_publisher_->publish(StartTask());
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->state == Status::WAITING_FOR_INPUT;
  }));
  tf_publisher_->publish(MapFromOdom());
  odometry_publisher_->publish(Odometry());

  for (const std::int8_t invalid_value : {std::int8_t{49}, std::int8_t{50},
                                          std::int8_t{101}}) {
    auto invalid_map = ExplorationMap(false);
    invalid_map.data.at(1U) = invalid_value;
    const auto statuses_before_map = StatusCount();
    map_publisher_->publish(std::move(invalid_map));

    ASSERT_TRUE(WaitFor([this, statuses_before_map] {
      return HasStatusReasonPrefixSince(statuses_before_map,
                                        "INVALID_EXPLORATION_MAP:");
    }));
    EXPECT_EQ(server_->goal_count(), 0U);
  }
}

TEST_F(IncrementalExplorationNodeTest,
       RejectsBaseFootprintOdometryUnderTheSharedOdomToBaseLinkContract) {
  task_publisher_->publish(StartTask());
  map_publisher_->publish(ExplorationMap(false));
  tf_publisher_->publish(MapFromOdom());
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->state == Status::WAITING_FOR_INPUT;
  }));

  auto invalid_odometry = Odometry();
  invalid_odometry.child_frame_id = "base_footprint";
  odometry_publisher_->publish(std::move(invalid_odometry));

  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->reason_code.rfind("INVALID_POSE_INPUT:", 0U) == 0U;
  }));
  EXPECT_EQ(server_->goal_count(), 0U);
}

TEST_F(IncrementalExplorationNodeTest,
       ExplorationMapAloneBuildsCandidatesAndOneCycleSendsOneGoal) {
  Start(false);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->candidate_count > 0U;
  }));
  const auto first_count = LastStatus()->candidate_count;
  const auto task_map_subscriptions = node_->get_subscriptions_info_by_topic(
      parameters_.task_map_markers_topic);
  EXPECT_TRUE(std::none_of(
      task_map_subscriptions.begin(), task_map_subscriptions.end(),
      [](const auto& endpoint) {
        return endpoint.node_name() == "incremental_exploration";
      }));

  map_publisher_->publish(ExplorationMap(true));
  ASSERT_TRUE(WaitFor([this, first_count] {
    const auto status = LastStatus();
    return status && status->candidate_count > first_count;
  }));
  EXPECT_EQ(server_->goal_count(), 1U);
}

TEST_F(IncrementalExplorationNodeTest,
       UnknownCurrentCellStillSubmitsKnownFreeFrontierGoal) {
  const auto map = ExplorationMapWithUnknownRobotCell();
  map_publisher_->publish(map);
  odometry_publisher_->publish(Odometry());
  tf_publisher_->publish(MapFromOdom());
  task_publisher_->publish(StartTask());

  EXPECT_TRUE(WaitFor([this] { return server_->goal_count() == 1U; }));
  const auto goal = server_->goal(0U);
  const auto goal_x = static_cast<std::int32_t>(std::floor(goal.target_x_m));
  const auto goal_y = static_cast<std::int32_t>(std::floor(goal.target_y_m));
  ASSERT_GE(goal_x, 0);
  ASSERT_GE(goal_y, 0);
  ASSERT_LT(static_cast<std::uint32_t>(goal_x), map.info.width);
  ASSERT_LT(static_cast<std::uint32_t>(goal_y), map.info.height);
  EXPECT_EQ(map.data.at(static_cast<std::size_t>(goal_y) * map.info.width +
                        static_cast<std::uint32_t>(goal_x)),
            0);
}

TEST_F(IncrementalExplorationNodeTest,
       NoMapBackedFreeEvidenceWaitsBeforeCoverageCompletion) {
  map_publisher_->publish(FullyOccupiedExplorationMap());
  odometry_publisher_->publish(Odometry());
  tf_publisher_->publish(MapFromOdom());
  task_publisher_->publish(StartTask());

  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->state == Status::WAITING_FOR_INPUT &&
           status->reason_code == "WAITING_FOR_KNOWN_FREE_MAP_EVIDENCE";
  }));
  EXPECT_EQ(server_->goal_count(), 0U);
}

TEST_F(IncrementalExplorationNodeTest,
       TaskRasterGridStaysFixedAcrossExplorationMapGeometryChanges) {
  Start();
  ASSERT_TRUE(WaitFor([this] { return LastTaskMap().has_value(); }));
  const auto before_count = TaskMapMessageCount();
  ASSERT_FALSE(LastTaskMap()->markers.empty());
  EXPECT_DOUBLE_EQ(LastTaskMap()->markers.front().scale.x, 1.0);

  map_publisher_->publish(FinerExplorationMap());
  ASSERT_TRUE(WaitFor([this, before_count] {
    return TaskMapMessageCount() > before_count;
  }));
  ASSERT_FALSE(LastTaskMap()->markers.empty());
  EXPECT_DOUBLE_EQ(LastTaskMap()->markers.front().scale.x, 1.0);
  EXPECT_EQ(server_->goal_count(), 1U);
}

TEST_F(IncrementalExplorationNodeTest,
       NoPathSuppressesCurrentEvidenceButMapUnavailableDoesNot) {
  Start();
  const auto first = server_->goal(0U);
  server_->Finish(Action::Result::NO_PATH, "NO_PATH");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  const auto second = server_->goal(1U);
  EXPECT_TRUE(first.target_x_m != second.target_x_m ||
              first.target_y_m != second.target_y_m ||
              first.target_yaw_rad != second.target_yaw_rad);

  server_->Finish(Action::Result::MAP_UNAVAILABLE, "MAP_UNAVAILABLE");
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(server_->goal_count(), 2U);
  map_publisher_->publish(ExplorationMap());
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 3U; }));
  const auto third = server_->goal(2U);
  EXPECT_DOUBLE_EQ(second.target_x_m, third.target_x_m);
  EXPECT_DOUBLE_EQ(second.target_y_m, third.target_y_m);
  EXPECT_DOUBLE_EQ(second.target_yaw_rad, third.target_yaw_rad);

  server_->Finish(Action::Result::TIMEOUT, "TIMEOUT");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 4U; }));
  const auto fourth = server_->goal(3U);
  EXPECT_TRUE(third.target_x_m != fourth.target_x_m ||
              third.target_y_m != fourth.target_y_m ||
              third.target_yaw_rad != fourth.target_yaw_rad);
}

TEST_F(IncrementalExplorationNodeTest,
       GoalReachedContinuesWithAnotherCandidateOnUnchangedMap) {
  Start();
  const auto first = server_->goal(0U);
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_FALSE(SameNavigationPose(first, server_->goal(1U)));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->completed_goal_count == 1U;
  }));
  EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
}

TEST_F(IncrementalExplorationNodeTest,
       GoalReachedAfterMapUpdateDoesNotRepeatJustReachedCandidate) {
  Start(false);
  const auto first = server_->goal(0U);
  auto changed_map = ExplorationMap(false);
  changed_map.data.front() = 0;
  map_publisher_->publish(changed_map);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_FALSE(SameNavigationPose(first, server_->goal(1U)));
}

TEST_F(IncrementalExplorationNodeTest,
       SuccessfulObservationDoesNotSuppressOtherPositions) {
  Start();
  const auto first = server_->goal(0U);
  for (std::size_t index = 0U; index < 30U; ++index) {
    server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
    ASSERT_TRUE(WaitFor([this, index] {
      return server_->goal_count() == index + 2U;
    }));
    const auto next = server_->goal(index + 1U);
    if (next.target_x_m != first.target_x_m || next.target_y_m != first.target_y_m) {
      return;
    }
  }
  FAIL() << "success at one pose must not suppress different positions";
}

TEST_F(IncrementalExplorationNodeTest,
       SuccessfulCandidatesIgnoreRemoteEvidenceAndRetryAfterSensorRangeChange) {
  Start(false);
  const auto first = server_->goal(0U);
  const auto completed = CompleteCandidatesOnUnchangedMap();
  ASSERT_GT(completed, 1U);

  const auto before_pose = StatusCount();
  odometry_publisher_->publish(Odometry());
  tf_publisher_->publish(MapFromOdom());
  ASSERT_TRUE(WaitFor([this, before_pose] {
    return StatusCount() >= before_pose + 2U;
  }));
  EXPECT_EQ(LastStatus()->reason_code, "WAITING_FOR_MAP_CHANGE");
  EXPECT_EQ(server_->goal_count(), completed);

  auto same_map = ExplorationMap(false);
  same_map.header.stamp.sec = 123;
  same_map.info.map_load_time.sec = 456;
  map_publisher_->publish(same_map);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  EXPECT_EQ(successful_candidate_count_.load(), completed);
  EXPECT_EQ(LastStatus()->reason_code, "WAITING_FOR_MAP_CHANGE");
  EXPECT_EQ(server_->goal_count(), completed);

  auto changed_map = same_map;
  changed_map.data.front() = 0;  // Remote from every view of the remaining hole.
  map_publisher_->publish(changed_map);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 3U; }));
  EXPECT_EQ(successful_candidate_count_.load(), completed);
  EXPECT_EQ(observation_memory_count_.load(), completed);
  EXPECT_EQ(LastStatus()->reason_code, "WAITING_FOR_MAP_CHANGE");
  EXPECT_EQ(server_->goal_count(), completed);

  // Change a cell behind the reached view, beyond the navigation failure
  // footprint patch but inside the sensor range. The original hole and the
  // candidate's collision footprint remain unchanged.
  std::optional<std::size_t> local_cell;
  for (std::size_t index = 0U; index < changed_map.data.size(); ++index) {
    const double x = static_cast<double>(index % changed_map.info.width) + 0.5;
    const double y = static_cast<double>(index / changed_map.info.width) + 0.5;
    const double distance = std::hypot(x - first.target_x_m, y - first.target_y_m);
    const double toward_hole = (x - first.target_x_m) * (7.5 - first.target_x_m) +
                               (y - first.target_y_m) * (6.5 - first.target_y_m);
    if (changed_map.data[index] == 0 && distance > 2.5 &&
        distance < parameters_.sensor_model.range_m && toward_hole < 0.0) {
      local_cell = index;
      break;
    }
  }
  ASSERT_TRUE(local_cell);
  changed_map.data[*local_cell] = 100;
  map_publisher_->publish(changed_map);
  ASSERT_TRUE(WaitFor([this, completed] {
    return server_->goal_count() == completed + 1U;
  }));
  EXPECT_TRUE(SameNavigationPose(first, server_->goal(completed)));
  EXPECT_GT(observation_invalidations_.load(), 0U);
}

TEST_F(IncrementalExplorationNodeTest,
       NewTaskClearsSuccessfulCandidatesEvenWhenTaskIdIsReused) {
  Start(false);
  const auto first = server_->goal(0U);
  const auto completed = CompleteCandidatesOnUnchangedMap();
  ASSERT_GT(completed, 1U);
  task_publisher_->publish(StartTask());
  ASSERT_TRUE(WaitFor([this, completed] {
    return server_->goal_count() == completed + 1U;
  }));
  EXPECT_TRUE(SameNavigationPose(first, server_->goal(completed)));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->completed_goal_count == 0U;
  }));
}

TEST_F(IncrementalExplorationNodeTest,
       DistantMapGrowthAndAlignedWindowShiftPreserveObservationEvidence) {
  Start(false);
  const auto completed = CompleteCandidatesOnUnchangedMap();
  ASSERT_GT(completed, 1U);
  const auto original = ExplorationMap(false);
  auto expanded = original;
  expanded.info.width = 22U;
  expanded.info.height = 22U;
  expanded.info.origin.position.x = -5.0;
  expanded.info.origin.position.y = -5.0;
  expanded.data.assign(22U * 22U, 0);
  for (std::size_t y = 0U; y < original.info.height; ++y) {
    for (std::size_t x = 0U; x < original.info.width; ++x) {
      expanded.data[(y + 5U) * expanded.info.width + x + 5U] =
          original.data[y * original.info.width + x];
    }
  }
  map_publisher_->publish(expanded);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  EXPECT_EQ(observation_memory_count_.load(), completed);
  EXPECT_EQ(observation_invalidations_.load(), 0U);
  EXPECT_EQ(LastStatus()->reason_code, "WAITING_FOR_MAP_CHANGE");
  EXPECT_EQ(server_->goal_count(), completed);
}

TEST_F(IncrementalExplorationNodeTest,
       ChangedLocalObservationResolutionAllowsRetry) {
  Start(false);
  const auto first = server_->goal(0U);
  const auto completed = CompleteCandidatesOnUnchangedMap();
  ASSERT_GT(completed, 1U);
  const auto original = ExplorationMap(false);
  auto finer = original;
  finer.info.width = 24U;
  finer.info.height = 24U;
  finer.info.resolution = 0.5F;
  finer.data.resize(24U * 24U);
  for (std::size_t y = 0U; y < finer.info.height; ++y) {
    for (std::size_t x = 0U; x < finer.info.width; ++x) {
      finer.data[y * finer.info.width + x] =
          original.data[(y / 2U) * original.info.width + x / 2U];
    }
  }
  map_publisher_->publish(finer);
  ASSERT_TRUE(WaitFor([this, completed] {
    return server_->goal_count() == completed + 1U;
  }));
  EXPECT_TRUE(SameNavigationPose(first, server_->goal(completed)));
  EXPECT_EQ(observation_invalidations_.load(), completed);
}

TEST_F(IncrementalExplorationNodeTest,
       FractionalOriginAlignedWindowShiftPreservesObservationEvidence) {
  auto original = ExplorationMap(false);
  original.info.origin.position.x = 0.1;
  original.info.origin.position.y = 0.1;
  map_publisher_->publish(original);
  odometry_publisher_->publish(Odometry());
  tf_publisher_->publish(MapFromOdom());
  task_publisher_->publish(StartTask());
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 1U; }));
  const auto completed = CompleteCandidatesOnUnchangedMap();
  ASSERT_GT(completed, 1U);

  auto shifted = original;
  shifted.info.width = 22U;
  shifted.info.height = 22U;
  shifted.info.origin.position.x = -4.9;
  shifted.info.origin.position.y = -4.9;
  shifted.data.assign(22U * 22U, 0);
  for (std::size_t y = 0U; y < original.info.height; ++y) {
    for (std::size_t x = 0U; x < original.info.width; ++x) {
      shifted.data[(y + 5U) * shifted.info.width + x + 5U] =
          original.data[y * original.info.width + x];
    }
  }
  map_publisher_->publish(shifted);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  EXPECT_EQ(observation_memory_count_.load(), completed);
  EXPECT_EQ(observation_invalidations_.load(), 0U);
  EXPECT_EQ(LastStatus()->reason_code, "WAITING_FOR_MAP_CHANGE");
  EXPECT_EQ(server_->goal_count(), completed);

  // The integer shift is reversible without accumulating or hiding drift.
  map_publisher_->publish(original);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 3U; }));
  EXPECT_EQ(observation_memory_count_.load(), completed);
  EXPECT_EQ(observation_invalidations_.load(), 0U);
  EXPECT_EQ(server_->goal_count(), completed);
}

TEST_F(IncrementalExplorationNodeTest,
       HalfCellOriginShiftInvalidatesObservationEvidence) {
  Start(false);
  const auto completed = CompleteCandidatesOnUnchangedMap();
  ASSERT_GT(completed, 1U);
  auto shifted = ExplorationMap(false);
  shifted.info.origin.position.x += 0.5;
  shifted.info.origin.position.y += 0.5;
  map_publisher_->publish(shifted);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  EXPECT_EQ(observation_memory_count_.load(), 0U);
  EXPECT_EQ(observation_invalidations_.load(), completed);
}

TEST_F(IncrementalExplorationNodeTest,
       SmallRealOriginShiftInvalidatesObservationEvidence) {
  Start(false);
  const auto completed = CompleteCandidatesOnUnchangedMap();
  ASSERT_GT(completed, 1U);
  auto shifted = ExplorationMap(false);
  shifted.info.origin.position.x += 1.0e-5;
  map_publisher_->publish(shifted);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  EXPECT_EQ(observation_memory_count_.load(), 0U);
  EXPECT_EQ(observation_invalidations_.load(), completed);
}

TEST_F(IncrementalExplorationNodeTest,
       SuccessRacingWithPauseStaysPausedUntilResume) {
  Start(false);
  const auto first = server_->goal(0U);
  task_publisher_->publish(Command(Task::PAUSE));
  ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->completed_goal_count == 1U;
  }));
  EXPECT_EQ(LastStatus()->state, Status::PAUSED);
  ASSERT_TRUE(WaitFor([this] { return observation_memory_count_.load() == 1U; }));
  auto remote_change = ExplorationMap(false);
  remote_change.data.front() = 0;
  map_publisher_->publish(remote_change);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  EXPECT_EQ(observation_memory_count_.load(), 1U);
  const auto before_pose = StatusCount();
  odometry_publisher_->publish(Odometry());
  ASSERT_TRUE(WaitFor([this, before_pose] { return StatusCount() > before_pose; }));
  EXPECT_EQ(server_->goal_count(), 1U);

  task_publisher_->publish(Command(Task::RESUME));
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  EXPECT_FALSE(SameNavigationPose(first, server_->goal(1U)));
}

TEST_F(IncrementalExplorationNodeTest,
       OldTaskSuccessAndCancelDoNotRepopulateObservationMemory) {
  Start(false);
  const auto first = server_->goal(0U);
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  task_publisher_->publish(StartTask());
  ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() == 1U; }));
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 3U; }));
  EXPECT_TRUE(SameNavigationPose(first, server_->goal(2U)));
  ASSERT_TRUE(WaitFor([this] {
    return observation_memory_count_.load() == 0U &&
           LastStatus()->completed_goal_count == 0U;
  }));
  task_publisher_->publish(Command(Task::CANCEL));
  ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() == 2U; }));
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->state == Status::IDLE; }));
  EXPECT_EQ(observation_memory_count_.load(), 0U);
  EXPECT_EQ(server_->goal_count(), 3U);
}

TEST_F(IncrementalExplorationNodeTest,
       SuppressedCandidatesAreGrayAndActiveCandidateStaysRed) {
  Start(false);
  const auto first = server_->goal(0U);
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  const auto second = server_->goal(1U);
  server_->Finish(Action::Result::NO_PATH, "NO_PATH");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 3U; }));
  const auto active = server_->goal(2U);
  ASSERT_TRUE(WaitFor([this] { return suppressed_candidate_count_.load() == 2U; }));
  EXPECT_EQ(observation_memory_count_.load(), 1U);
  EXPECT_EQ(LastStatus()->failed_candidate_count, 1U);
  EXPECT_EQ(eligible_candidate_count_.load() + suppressed_candidate_count_.load(),
            LastStatus()->candidate_count);
  EXPECT_EQ(LastStatus()->reachable_candidate_count, eligible_candidate_count_.load());
  ASSERT_TRUE(WaitFor([this, first, second, active] {
    const auto messages = FrontierMessages();
    if (messages.empty()) {
      return false;
    }
    std::size_t gray = 0U;
    bool selected_red = false;
    for (const auto& marker : messages.back().markers) {
      if (marker.ns != "candidates" || marker.action != marker.ADD) {
        continue;
      }
      auto matches = [&marker](const Action::Goal& goal) {
        return marker.pose.position.x == goal.target_x_m &&
               marker.pose.position.y == goal.target_y_m &&
               marker.pose.orientation.z == std::sin(goal.target_yaw_rad / 2.0) &&
               marker.pose.orientation.w == std::cos(goal.target_yaw_rad / 2.0);
      };
      if ((matches(first) || matches(second)) &&
          marker.color.r == marker.color.g && marker.color.g == marker.color.b &&
          marker.color.a < 0.5F) {
        ++gray;
      }
      if (matches(active) && marker.color.r == 1.0F && marker.color.g == 0.0F &&
          marker.color.a == 1.0F) {
        selected_red = true;
      }
    }
    return gray == 2U && selected_red;
  }));
}

class IncrementalExplorationSmallObservationMemoryTest
    : public IncrementalExplorationNodeTest {
 protected:
  void ConfigureParameters(IncrementalExplorationNodeParameters& parameters) override {
    // A 5m observation at 1m resolution uses at most 11x11 patch cells.
    // Visibility of the single UNKNOWN cell fits, but only two patches fit.
    parameters.information_gain_limits.maximum_visibility_work_units = 256U;
  }
};

TEST_F(IncrementalExplorationSmallObservationMemoryTest,
       ObservationMemoryEvictsOldVisitsInsteadOfFailingAtCapacity) {
  Start(false);
  for (std::size_t index = 0U; index < 1005U; ++index) {
    server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
    ASSERT_TRUE(WaitFor([this, index] {
      return server_->goal_count() == index + 2U;
    }));
    ASSERT_TRUE(WaitFor([this, index] {
      return LastStatus()->completed_goal_count == index + 1U;
    }));
    EXPECT_LE(observation_memory_count_.load(), 2U);
    EXPECT_LE(observation_patch_cells_.load(), 256U);
    EXPECT_EQ(LastStatus()->failed_candidate_count, 0U);
    EXPECT_NE(LastStatus()->state, Status::ERROR);
  }
  EXPECT_GT(observation_evictions_.load(), 0U);
  EXPECT_EQ(observation_invalidations_.load(), 0U);
}

TEST_F(IncrementalExplorationSmallObservationMemoryTest,
       ChangedObservationEvidenceIsRemovedBeforeCapacityEviction) {
  Start(false);
  for (std::size_t index = 0U; index < 2U; ++index) {
    server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
    ASSERT_TRUE(WaitFor([this, index] {
      return server_->goal_count() == index + 2U;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return observation_memory_count_.load() == 2U; }));
  auto local_change = ExplorationMap(false);
  local_change.data[6U * local_change.info.width + 10U] = 100;
  map_publisher_->publish(local_change);
  ASSERT_TRUE(WaitFor([this] { return map_sequence_.load() == 2U; }));
  EXPECT_GT(observation_invalidations_.load(), 0U);
  EXPECT_EQ(observation_evictions_.load(), 0U);
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 4U; }));
  EXPECT_EQ(observation_evictions_.load(), 0U);
}

TEST_F(IncrementalExplorationNodeTest,
       PauseResumeCancelAndNewStartCancelTheActionGoal) {
  Start();
  task_publisher_->publish(Command(Task::PAUSE));
  ASSERT_TRUE(WaitFor([this] { return server_->cancel_count() >= 1U; }));
  task_publisher_->publish(Command(Task::RESUME));
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(server_->goal_count(), 1U);
  server_->Finish(Action::Result::CANCELED, "CANCELED");
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));

  const auto before_cancel = server_->cancel_count();
  task_publisher_->publish(Command(Task::CANCEL));
  ASSERT_TRUE(WaitFor(
      [this, before_cancel] { return server_->cancel_count() > before_cancel; }));
  server_->Finish(Action::Result::CANCELED, "CANCELED");
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->state == Status::IDLE;
  }));

  task_publisher_->publish(StartTask("replacement-one"));
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 3U; }));
  const auto before_replacement = server_->cancel_count();
  task_publisher_->publish(StartTask("replacement-two"));
  ASSERT_TRUE(WaitFor([this, before_replacement] {
    return server_->cancel_count() > before_replacement;
  }));
  server_->Finish(Action::Result::CANCELED, "CANCELED");
  EXPECT_TRUE(WaitFor([this] { return server_->goal_count() == 4U; }));
}

TEST_F(IncrementalExplorationNodeTest, OnlyInternalNavigationErrorEntersError) {
  Start();
  server_->Finish(Action::Result::MAP_UNAVAILABLE, "MAP_UNAVAILABLE");
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->state == Status::WAITING_FOR_INPUT;
  }));
  map_publisher_->publish(ExplorationMap());
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));
  server_->Finish(Action::Result::INTERNAL_ERROR, "NAVIGATION_INTERNAL_ERROR");
  EXPECT_TRUE(WaitFor([this] {
    const auto status = LastStatus();
    return status && status->state == Status::ERROR;
  }));
}

class CoverageMilestoneTest : public IncrementalExplorationNodeTest {
 protected:
  void ConfigureParameters(IncrementalExplorationNodeParameters& p) override {
    p.coverage_target = 0.80;
  }
};
class SecondCoverageMilestoneTest : public CoverageMilestoneTest {
 protected:
  void ConfigureParameters(IncrementalExplorationNodeParameters& p) override {
    p.coverage_target = 0.99;
  }
};
TEST_F(CoverageMilestoneTest, CandidatesContinueBeyondEightyPercent) {
  Start(false);
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->coverage_ratio > 0.99; }));
  EXPECT_NE(LastStatus()->state, Status::COMPLETED);
}
TEST_F(SecondCoverageMilestoneTest, CandidatesContinueBeyondNinetyNinePercent) {
  Start(false);
  ASSERT_TRUE(WaitFor([this] { return LastStatus()->coverage_ratio > 0.99; }));
  EXPECT_NE(LastStatus()->state, Status::COMPLETED);
}
TEST_F(CoverageMilestoneTest, FullyKnownMapCompletesWithoutNavigation) {
  auto map = ExplorationMap(false);
  std::fill(map.data.begin(), map.data.end(), 0);
  map_publisher_->publish(map);
  odometry_publisher_->publish(Odometry());
  tf_publisher_->publish(MapFromOdom());
  task_publisher_->publish(StartTask());
  ASSERT_TRUE(WaitFor([this] {
    return LastStatus() && LastStatus()->state == Status::COMPLETED;
  }));
  EXPECT_EQ(LastStatus()->reason_code, "COMPLETED_NO_REACHABLE_FRONTIER");
  EXPECT_EQ(server_->goal_count(), 0U);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
