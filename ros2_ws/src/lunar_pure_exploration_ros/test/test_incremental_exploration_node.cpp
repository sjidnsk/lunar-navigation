#include "lunar_pure_exploration_ros/incremental_exploration_node.hpp"

#include <algorithm>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
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
  void SetUp() override {
    const auto id = next_id_.fetch_add(1U);
    const std::string prefix = "/incremental_test_" + std::to_string(id);
    parameters_ = Parameters(prefix);
    server_node_ = std::make_shared<rclcpp::Node>(
        "incremental_navigation_server_" + std::to_string(id));
    observer_ = std::make_shared<rclcpp::Node>(
        "incremental_exploration_observer_" + std::to_string(id));
    server_ = std::make_unique<FakeNavigationServer>(
        *server_node_, parameters_.navigation_action);
    node_ = std::make_shared<IncrementalExplorationNode>(parameters_);

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
    task_map_subscription_ = observer_->create_subscription<
        visualization_msgs::msg::MarkerArray>(
        parameters_.task_map_markers_topic,
        rclcpp::QoS{1}.reliable().transient_local(),
        [this](visualization_msgs::msg::MarkerArray::SharedPtr markers) {
          std::scoped_lock lock{task_map_mutex_};
          last_task_map_ = *markers;
          ++task_map_message_count_;
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

  void PublishPoseAndMap(bool broad_frontier = true) {
    map_publisher_->publish(ExplorationMap(broad_frontier));
    odometry_publisher_->publish(Odometry());
    tf_publisher_->publish(MapFromOdom());
  }

  void Start(bool broad_frontier = true) {
    PublishPoseAndMap(broad_frontier);
    task_publisher_->publish(StartTask());
    ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 1U; }));
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
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr
      task_map_subscription_;
  mutable std::mutex status_mutex_;
  std::optional<Status> last_status_;
  std::vector<Status> statuses_;
  mutable std::mutex task_map_mutex_;
  std::optional<visualization_msgs::msg::MarkerArray> last_task_map_;
  std::size_t task_map_message_count_{0U};
  rclcpp::executors::MultiThreadedExecutor executor_;
  std::jthread spin_;
};

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
       GoalReachedWaitsForANewerMapOrImmediatelyUsesOneAlreadyReceived) {
  Start();
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(server_->goal_count(), 1U);
  map_publisher_->publish(ExplorationMap());
  ASSERT_TRUE(WaitFor([this] { return server_->goal_count() == 2U; }));

  map_publisher_->publish(ExplorationMap());
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(server_->goal_count(), 2U);
  server_->Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  EXPECT_TRUE(WaitFor([this] { return server_->goal_count() == 3U; }));
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

}  // namespace
}  // namespace lunar::pure_exploration_ros
