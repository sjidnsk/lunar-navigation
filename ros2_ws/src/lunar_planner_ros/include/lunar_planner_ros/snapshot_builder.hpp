#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>

#include <rclcpp/time.hpp>

#include "lunar_planner_core/types/planner_io.hpp"
#include "lunar_planner_ros/grid_map_adapter.hpp"
#include "lunar_planner_ros/snapshot_store.hpp"

namespace lunar::planning::ros {

enum class SnapshotErrorCode : std::uint8_t {
  kConfigurationInvalid,
  kMissingGlobalMap,
  kMissingLocalMap,
  kMissingOdometry,
  kMissingLocalizationStatus,
  kInvalidGlobalMap,
  kInvalidLocalMap,
  kStaleGlobalMap,
  kStaleLocalMap,
  kStaleOdometry,
  kStaleLocalizationStatus,
  kInputSkew,
  kInvalidOdometry,
  kInvalidLocalization,
  kCovarianceLimit,
  kStaleTf,
  kInvalidGoal,
};

struct SnapshotError final {
  SnapshotErrorCode code{SnapshotErrorCode::kConfigurationInvalid};
  std::string reason_code;
  std::string detail;
};

struct SnapshotPolicy final {
  std::chrono::nanoseconds global_map_max_age{};
  std::chrono::nanoseconds local_map_max_age{};
  std::chrono::nanoseconds odometry_max_age{};
  std::chrono::nanoseconds localization_status_max_age{};
  std::chrono::nanoseconds tf_max_age{};
  std::chrono::nanoseconds max_pairwise_skew{};
  double degraded_pose_covariance_limit{};
  double degraded_twist_covariance_limit{};
};

[[nodiscard]] bool ValidateSnapshotPolicy(
    const SnapshotPolicy& policy) noexcept;

struct GoalRequest final {
  std::string request_id;
  std::string frame_id;
  rclcpp::Time stamp;
  lunar::planning::GoalRegion goal;
  std::optional<lunar::planning::ExecutionContext> previous_execution;
  std::stop_token stop_token;
};

struct SnapshotBuildResult final {
  std::optional<lunar::planning::PlannerInput> input;
  std::optional<SnapshotError> error;

  [[nodiscard]] bool ok() const noexcept {
    return input.has_value() && !error.has_value();
  }
};

class SnapshotBuilder final {
 public:
  SnapshotBuilder(
      std::shared_ptr<const SnapshotStore> store,
      SnapshotPolicy policy,
      lunar::planning::PlatformCapability capability,
      lunar::planning::PlannerConfig planner_config = {});

  [[nodiscard]] SnapshotBuildResult Freeze(
      const GoalRequest& request,
      rclcpp::Time now) const;

 private:
  std::shared_ptr<const SnapshotStore> store_;
  SnapshotPolicy policy_;
  lunar::planning::PlatformCapability capability_;
  lunar::planning::PlannerConfig planner_config_;
  GridMapAdapter grid_map_adapter_;
};

}  // namespace lunar::planning::ros
