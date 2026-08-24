#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <lunar_pure_exploration_core/candidate_generator.hpp>
#include <rclcpp/node.hpp>

namespace lunar::pure_exploration_ros {

enum class PlannerEvaluationKind : std::uint8_t {
  kReachable,
  kExhaustiveNoPath,
  kRetryable,
  kCanceled,
  kContractError,
  kResourceError,
};

enum class PlannerResourceKind : std::uint8_t {
  kNone,
  kPathPreviewPoses,
  kExecutablePathPoints,
  kResultCopyFailure,
};

struct PlannerEvaluation {
  std::string request_id;
  std::uint64_t candidate_id;  // Display only; request_id is the association key.
  PlannerEvaluationKind kind;
  PlannerResourceKind resource_kind{PlannerResourceKind::kNone};
  std::string reason_code;
  std::optional<double> path_length_m;
  std::optional<lunar_planning_msgs::msg::MotionReference> reference;
};

struct PlannerClientParameters {
  std::size_t maximum_path_preview_poses;
  std::size_t maximum_executable_path_points;
  std::chrono::steady_clock::duration goal_response_timeout;
  std::chrono::steady_clock::duration result_timeout;
};

class PlannerClient {
 public:
  using Completion = std::function<void(PlannerEvaluation)>;
  using SteadyNow =
      std::function<std::chrono::steady_clock::time_point()>;

  PlannerClient(
      rclcpp::Node& node, std::string action_name,
      PlannerClientParameters parameters,
      SteadyNow now = [] { return std::chrono::steady_clock::now(); });
  ~PlannerClient() noexcept;
  PlannerClient(const PlannerClient&) = delete;
  PlannerClient& operator=(const PlannerClient&) = delete;
  PlannerClient(PlannerClient&&) = delete;
  PlannerClient& operator=(PlannerClient&&) = delete;

  static std::string MakeRequestId(std::string_view task_id,
                                   std::uint64_t sequence);
  void Evaluate(std::string task_id, std::string request_id,
                const lunar::pure_exploration::CandidateView& candidate,
                double position_tolerance_m, double yaw_tolerance_rad,
                Completion completion);
  void CancelActive();
  void PollTimeout();

 private:
  struct CallbackState;
  std::shared_ptr<CallbackState> state_;
};

}  // namespace lunar::pure_exploration_ros
