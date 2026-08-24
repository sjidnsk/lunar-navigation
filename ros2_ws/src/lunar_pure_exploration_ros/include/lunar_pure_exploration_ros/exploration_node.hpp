#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <lunar_pure_exploration_core/candidate_ranker.hpp>
#include <lunar_pure_exploration_core/coverage.hpp>
#include <lunar_pure_exploration_core/failure_memory.hpp>
#include <lunar_pure_exploration_core/frontier_detector.hpp>
#include <lunar_pure_exploration_core/information_gain.hpp>
#include <lunar_pure_exploration_core/progress_monitor.hpp>
#include <lunar_pure_exploration_core/task_raster.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/timer.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::pure_exploration_ros {

struct ExecutionMonitorParameters {
  std::size_t maximum_executable_path_points;
  double position_tolerance_m;
  double yaw_tolerance_rad;
  std::chrono::steady_clock::duration stuck_window;
  double minimum_progress_m;
};

// Owns only execution-time geometric/progress checks.  Goal ownership and
// lifecycle transitions remain with ExplorationStateMachine.
class ExecutionMonitor final {
 public:
  explicit ExecutionMonitor(ExecutionMonitorParameters parameters);
  bool ReachedFinalGoal(lunar::pure_exploration::Pose2 position,
                        lunar::pure_exploration::Pose2 target) const;
  bool ReachedSegmentEndpoint(lunar::pure_exploration::Vec2 position,
                              lunar::pure_exploration::Vec2 endpoint) const;
  void ResetProgress(std::chrono::steady_clock::time_point now,
                     std::span<const lunar::pure_exploration::Vec2> polyline,
                     lunar::pure_exploration::Vec2 position);
  bool UpdateProgress(std::chrono::steady_clock::time_point now,
                      lunar::pure_exploration::Vec2 position, bool paused);

 private:
  double position_tolerance_m_;
  double yaw_tolerance_rad_;
  lunar::pure_exploration::ProgressMonitor progress_;
};

struct PlannerEvaluation;

struct FrozenGlobalMapContent {
  lunar::pure_exploration::GridGeometry geometry;
  std::vector<std::int8_t> data;

  [[nodiscard]] bool EqualsGeometryAndData(
      const nav_msgs::msg::OccupancyGrid& latest) const;
};

class FrozenCandidateBatch {
 public:
  FrozenCandidateBatch(
      FrozenGlobalMapContent global_map_content,
      std::vector<lunar::pure_exploration::FrontierCluster> frontiers,
      std::vector<lunar::pure_exploration::CandidateView> candidates);

  [[nodiscard]] std::span<
      const lunar::pure_exploration::FrontierCluster>
  frontiers() const;
  [[nodiscard]] std::span<const lunar::pure_exploration::CandidateView>
  candidates() const;
  void RegisterRequest(std::string request_id, std::size_t candidate_index);
  [[nodiscard]] std::optional<std::size_t> CandidateIndexForRequest(
      std::string_view request_id) const;
  [[nodiscard]] std::optional<std::string> LatestRequestIdForCandidate(
      std::size_t candidate_index) const;
  [[nodiscard]] bool GlobalMapContentEquals(
      const nav_msgs::msg::OccupancyGrid& latest) const;

 private:
  const FrozenGlobalMapContent global_map_content_;
  const std::vector<lunar::pure_exploration::FrontierCluster> frontiers_;
  const std::vector<lunar::pure_exploration::CandidateView> candidates_;
  std::unordered_map<std::string, std::size_t>
      request_to_candidate_index_;
  std::vector<std::optional<std::string>> latest_request_by_candidate_;
};

using FrozenCandidateBatchPtr = std::shared_ptr<FrozenCandidateBatch>;

struct FrozenReachableCandidate {
  lunar::pure_exploration::PlannedCandidate metrics;
  lunar_planning_msgs::msg::MotionReference reference;
};

class FrozenPlanningCycle {
 public:
  FrozenPlanningCycle(
      FrozenCandidateBatchPtr batch,
      lunar::pure_exploration::Pose2 frozen_robot_pose,
      double frozen_resolution_m,
      std::vector<lunar::pure_exploration::CandidateGain> complete_gains,
      std::vector<lunar::pure_exploration::RankedCandidate> coarse_order);

  [[nodiscard]] const FrozenCandidateBatchPtr& batch() const;
  [[nodiscard]] lunar::pure_exploration::Pose2 frozen_robot_pose() const;
  [[nodiscard]] double frozen_resolution_m() const;
  [[nodiscard]] std::span<const lunar::pure_exploration::CandidateGain>
  complete_gains() const;
  [[nodiscard]] std::span<const lunar::pure_exploration::RankedCandidate>
  coarse_order() const;
  [[nodiscard]] std::size_t coarse_cursor() const;
  std::vector<std::size_t> TakeNextCandidateIndices();
  void AddReachable(FrozenReachableCandidate result);
  [[nodiscard]] std::span<const FrozenReachableCandidate> reachable() const;

 private:
  const FrozenCandidateBatchPtr batch_;
  const lunar::pure_exploration::Pose2 frozen_robot_pose_;
  const double frozen_resolution_m_;
  const std::vector<lunar::pure_exploration::CandidateGain> complete_gains_;
  const std::vector<lunar::pure_exploration::RankedCandidate> coarse_order_;
  std::size_t coarse_cursor_{0U};
  std::vector<FrozenReachableCandidate> reachable_;
};

using FrozenPlanningCyclePtr = std::shared_ptr<FrozenPlanningCycle>;

struct ExplorationPipelineSeams {
  std::function<std::vector<lunar::pure_exploration::CandidateView>(
      const lunar::pure_exploration::TaskRaster&,
      std::span<const lunar::pure_exploration::FrontierCluster>)>
      generate_candidates;
  std::function<double(
      const lunar::pure_exploration::TaskRaster&,
      const lunar::pure_exploration::CandidateView&)>
      evaluate_gain;
  std::function<std::vector<lunar::pure_exploration::RankedCandidate>(
      std::span<const lunar::pure_exploration::CandidateView>,
      std::span<const lunar::pure_exploration::FrontierCluster>,
      std::span<const lunar::pure_exploration::CandidateGain>,
      std::span<const lunar::pure_exploration::PlannedCandidate>,
      lunar::pure_exploration::Pose2,
      std::span<const lunar::pure_exploration::Vec2>, double)>
      final_rank;
  std::function<lunar_planning_msgs::msg::MotionReference(
      const lunar_planning_msgs::msg::MotionReference&)>
      copy_reference;
  std::function<void()> before_reference_publish;
  std::function<void()> after_final_rank_map_observed;
};

struct ExplorationNodeParameters {
  lunar::pure_exploration::PlatformGeometry platform;
  lunar::pure_exploration::CandidateParameters candidate_parameters;
  lunar::pure_exploration::CandidateGenerator::Limits candidate_limits;
  lunar::pure_exploration::TaskRaster::Limits task_raster_limits;
  lunar::pure_exploration::SensorModel sensor_model;
  lunar::pure_exploration::InformationGainEvaluator::Limits
      information_gain_limits;
  lunar::pure_exploration::ScoreWeights score_weights;
  lunar::pure_exploration::FailureMemoryLimits failure_memory_limits;
  std::int8_t global_occupied_threshold;
  std::size_t maximum_path_preview_poses;
  std::size_t maximum_executable_path_points;
  std::uint8_t maximum_replans;
  double goal_yaw_tolerance_rad;
  std::chrono::steady_clock::duration planner_goal_response_timeout{
      std::chrono::seconds{1}};
  std::chrono::steady_clock::duration planner_result_timeout;
  std::string global_map_topic;
  std::string odometry_topic;
  std::string tf_topic;
  std::string task_topic;
  std::string planner_action;
  std::string planner_diagnostics_topic;
  std::string motion_reference_topic;
  std::string execution_cancel_topic;
  std::string status_topic;
  std::string current_goal_topic;
  std::string frontiers_topic;
  std::string diagnostics_topic;
  std::function<std::chrono::steady_clock::time_point()> steady_now{};
  std::shared_ptr<ExplorationPipelineSeams> pipeline_seams{};
};

class ExplorationNodeTestPeer;

class ExplorationNode final : public rclcpp::Node {
 public:
  explicit ExplorationNode(
      ExplorationNodeParameters parameters,
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});
  explicit ExplorationNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});
  ~ExplorationNode() noexcept override;

  ExplorationNode(const ExplorationNode&) = delete;
  ExplorationNode& operator=(const ExplorationNode&) = delete;
  ExplorationNode(ExplorationNode&&) = delete;
  ExplorationNode& operator=(ExplorationNode&&) = delete;

  // Explicit test seam. Production invokes the same path from a wall timer.
  void PollExecution();

 private:
  struct Runtime;

  void Initialize(ExplorationNodeParameters parameters);
  [[nodiscard]] FrozenPlanningCyclePtr SnapshotActiveCycleForTest() const;
  [[nodiscard]] std::optional<lunar_planning_msgs::msg::MotionReference>
  SnapshotActiveReferenceForTest() const;
  [[nodiscard]] std::optional<std::string>
  SnapshotActiveRequestIdForTest() const;
  [[nodiscard]] std::optional<lunar::pure_exploration::Pose2>
  SnapshotActiveTargetForTest() const;
  [[nodiscard]] std::vector<lunar::pure_exploration::Vec2>
  SnapshotExecutablePolylineForTest() const;
  [[nodiscard]] std::optional<lunar::pure_exploration::Vec2>
  SnapshotExecutableEndpointForTest() const;
  [[nodiscard]] double PlatformWidthForTest() const;
  [[nodiscard]] std::size_t CoarseCursorForTest() const;
  [[nodiscard]] std::optional<std::size_t>
  CandidateIndexForRequestForTest(const std::string& request_id) const;
  [[nodiscard]] std::optional<lunar::pure_exploration::Pose2>
  LatestPoseForTest() const;
  [[nodiscard]] std::optional<double> LatestMapResolutionForTest() const;
  void InjectEvaluationForTest(PlannerEvaluation evaluation);
  void ResetActiveCycleForTest();

  std::shared_ptr<Runtime> runtime_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      global_map_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription_;
  rclcpp::Subscription<
      lunar_pure_exploration_msgs::msg::PureExplorationTask>::SharedPtr
      task_subscription_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      planner_diagnostics_subscription_;
  rclcpp::TimerBase::SharedPtr execution_timer_;

  friend class ExplorationNodeTestPeer;
};

}  // namespace lunar::pure_exploration_ros
