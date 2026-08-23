#include "hierarchical/reference_composer.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "hierarchical/frame_transform.hpp"
#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Quaternion value) noexcept {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
         std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Pose3& pose) noexcept {
  return Finite(pose.position_m) && Finite(pose.orientation);
}

[[nodiscard]] const Pose3& CurrentPose(const PlanningRequest& input) {
  return std::visit(
      [](const auto& state) -> const Pose3& {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, LeggedState>) {
          return state.body_pose;
        } else {
          return state.pose;
        }
      },
      input.current_state);
}

[[nodiscard]] std::string PlanPrefix(const PlatformType platform) {
  switch (platform) {
    case PlatformType::kWheeled:
      return "wheel/";
    case PlatformType::kLegged:
      return "legged/";
    case PlatformType::kHopper:
      return "hopper/";
  }
  return {};
}

[[nodiscard]] std::optional<std::string> InvalidDataReason(
    const PlatformType platform, const MotionReferenceData& data,
    const SearchControl& control) {
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return std::string{*stopped};
  }
  if (const auto* trajectory = std::get_if<TrajectoryReference>(&data)) {
    const bool semantic_matches =
        (platform == PlatformType::kWheeled &&
         trajectory->semantics == TrajectorySemantics::kWheeledBase) ||
        (platform == PlatformType::kLegged &&
         trajectory->semantics ==
             TrajectorySemantics::kLeggedBodyReference);
    if (!semantic_matches || trajectory->points.empty()) {
      return std::string{"PLANNER_ERROR"};
    }
    for (std::size_t index = 0U; index < trajectory->points.size(); ++index) {
      if (shared::ControlCheckDue(index)) {
        if (const auto stopped = shared::StopReason(control);
            stopped.has_value()) {
          return std::string{*stopped};
        }
      }
      if (!Finite(trajectory->points[index].pose)) {
        return std::string{"PLANNER_ERROR"};
      }
    }
    return std::nullopt;
  }
  const auto* hops = std::get_if<HopReference>(&data);
  if (platform != PlatformType::kHopper || hops == nullptr) {
    return std::string{"PLANNER_ERROR"};
  }
  for (std::size_t index = 0U; index < hops->segments.size(); ++index) {
    if (shared::ControlCheckDue(index)) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return std::string{*stopped};
      }
    }
    const HopSegment& segment = hops->segments[index];
    if (!Finite(segment.launch_pose) ||
        !Finite(segment.nominal_landing_point_m)) {
      return std::string{"PLANNER_ERROR"};
    }
  }
  return std::nullopt;
}

struct PreviewBuildResult final {
  std::optional<std::vector<Pose3>> value;
  std::string reason_code;
};

struct DataTransformResult final {
  std::optional<MotionReferenceData> value;
  std::string reason_code;
};

[[nodiscard]] DataTransformResult TransformExecutionDataToMap(
    MotionReferenceData data, const RigidTransform& map_from_odom,
    const SearchControl& control) {
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  if (auto* trajectory = std::get_if<TrajectoryReference>(&data)) {
    for (std::size_t index = 0U; index < trajectory->points.size(); ++index) {
      if (shared::ControlCheckDue(index)) {
        if (const auto stopped = shared::StopReason(control);
            stopped.has_value()) {
          return {.reason_code = std::string{*stopped}};
        }
      }
      TrajectoryPoint& point = trajectory->points[index];
      const auto pose = TransformPose(point.pose, map_from_odom,
                                      TransformDirection::kChildToParent);
      const auto linear =
          TransformVector(point.velocity.linear_mps, map_from_odom,
                          TransformDirection::kChildToParent);
      const auto angular =
          TransformVector(point.velocity.angular_radps, map_from_odom,
                          TransformDirection::kChildToParent);
      if (!pose.has_value() || !linear.has_value() || !angular.has_value()) {
        return {.reason_code = "PLANNER_ERROR"};
      }
      point.pose = *pose;
      point.velocity.linear_mps = *linear;
      point.velocity.angular_radps = *angular;
    }
  } else {
    auto& hops = std::get<HopReference>(data);
    for (std::size_t segment_index = 0U;
         segment_index < hops.segments.size(); ++segment_index) {
      if (shared::ControlCheckDue(segment_index)) {
        if (const auto stopped = shared::StopReason(control);
            stopped.has_value()) {
          return {.reason_code = std::string{*stopped}};
        }
      }
      HopSegment& segment = hops.segments[segment_index];
      const auto launch_pose =
          TransformPose(segment.launch_pose, map_from_odom,
                        TransformDirection::kChildToParent);
      const auto nominal_landing =
          TransformPoint(segment.nominal_landing_point_m, map_from_odom,
                         TransformDirection::kChildToParent);
      const auto launch_velocity =
          TransformVector(segment.launch_velocity_mps, map_from_odom,
                          TransformDirection::kChildToParent);
      if (!launch_pose.has_value() || !nominal_landing.has_value() ||
          !launch_velocity.has_value()) {
        return {.reason_code = "PLANNER_ERROR"};
      }
      segment.launch_pose = *launch_pose;
      segment.nominal_landing_point_m = *nominal_landing;
      segment.launch_velocity_mps = *launch_velocity;
      for (std::size_t boundary_index = 0U;
           boundary_index < segment.landing_region_boundary_m.size();
           ++boundary_index) {
        if (shared::ControlCheckDue(boundary_index)) {
          if (const auto stopped = shared::StopReason(control);
              stopped.has_value()) {
            return {.reason_code = std::string{*stopped}};
          }
        }
        const auto boundary = TransformPoint(
            segment.landing_region_boundary_m[boundary_index], map_from_odom,
            TransformDirection::kChildToParent);
        if (!boundary.has_value()) {
          return {.reason_code = "PLANNER_ERROR"};
        }
        segment.landing_region_boundary_m[boundary_index] = *boundary;
      }
    }
  }
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  return {.value = std::move(data)};
}

[[nodiscard]] PreviewBuildResult CavePreview(
    const PlanningRequest& input, const MotionReferenceData& data,
    const SearchControl& control) {
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  std::vector<Pose3> poses_odom;
  if (const auto* trajectory = std::get_if<TrajectoryReference>(&data)) {
    poses_odom.reserve(trajectory->points.size());
    for (std::size_t index = 0U; index < trajectory->points.size(); ++index) {
      if (shared::ControlCheckDue(index)) {
        if (const auto stopped = shared::StopReason(control);
            stopped.has_value()) {
          return {.reason_code = std::string{*stopped}};
        }
      }
      poses_odom.push_back(trajectory->points[index].pose);
    }
  } else {
    const auto& hops = std::get<HopReference>(data);
    if (hops.segments.empty()) {
      poses_odom.push_back(CurrentPose(input));
    } else {
      poses_odom.reserve(hops.segments.size() + 1U);
      poses_odom.push_back(hops.segments.front().launch_pose);
      for (std::size_t index = 0U; index < hops.segments.size(); ++index) {
        if (shared::ControlCheckDue(index)) {
          if (const auto stopped = shared::StopReason(control);
              stopped.has_value()) {
            return {.reason_code = std::string{*stopped}};
          }
        }
        const HopSegment& segment = hops.segments[index];
        poses_odom.push_back({
            .position_m = segment.nominal_landing_point_m,
            .orientation = segment.launch_pose.orientation,
        });
      }
    }
  }
  std::vector<Pose3> poses_map;
  poses_map.reserve(poses_odom.size());
  for (std::size_t index = 0U; index < poses_odom.size(); ++index) {
    if (shared::ControlCheckDue(index)) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    const auto transformed = TransformPose(
        poses_odom[index], input.world.map_from_odom,
        TransformDirection::kChildToParent);
    if (!transformed.has_value()) {
      return {.reason_code = "PLANNER_ERROR"};
    }
    poses_map.push_back(*transformed);
  }
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  return {.value = std::move(poses_map)};
}

[[nodiscard]] ReferenceComposeResult BuildReference(
    const PlanningRequest& input, std::vector<Pose3> preview,
    MotionReferenceData data, const SearchControl& control) {
  const PlatformType platform = CapabilityPlatform(input.capability);
  const std::string prefix = PlanPrefix(platform);
  if (input.request_id.empty() || prefix.empty() || preview.empty()) {
    return {.reason_code = "PLANNER_ERROR"};
  }
  if (const auto invalid = InvalidDataReason(platform, data, control);
      invalid.has_value()) {
    return {.reason_code = std::move(*invalid)};
  }
  for (std::size_t index = 0U; index < preview.size(); ++index) {
    if (shared::ControlCheckDue(index)) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    if (!Finite(preview[index])) {
      return {.reason_code = "PLANNER_ERROR"};
    }
  }
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  auto transformed_data = TransformExecutionDataToMap(
      std::move(data), input.world.map_from_odom, control);
  if (!transformed_data.value.has_value()) {
    return {.reason_code = std::move(transformed_data.reason_code)};
  }
  return {
      .reference = MotionReference{
          .plan_id = prefix + input.request_id,
          .platform_type = platform,
          .input_time = {},
          .preview = GlobalRoutePreview{.poses_map = std::move(preview)},
          .data = std::move(*transformed_data.value),
      },
  };
}

}  // namespace

ReferenceComposeResult ComposeSurfaceReference(
    const PlanningRequest& input, const GlobalRoute& route,
    MotionReferenceData data, SearchControl control) {
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  std::vector<Pose3> preview;
  preview.reserve(route.poses_map.size());
  for (std::size_t index = 0U; index < route.poses_map.size(); ++index) {
    if (shared::ControlCheckDue(index)) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    preview.push_back(route.poses_map[index]);
  }
  return BuildReference(input, std::move(preview), std::move(data), control);
}

ReferenceComposeResult ComposeCaveReference(
    const PlanningRequest& input, MotionReferenceData data,
    SearchControl control) {
  auto preview = CavePreview(input, data, control);
  if (!preview.value.has_value()) {
    return {.reason_code = std::move(preview.reason_code)};
  }
  return BuildReference(input, std::move(*preview.value), std::move(data),
                        control);
}

}  // namespace lunar::pure_planning::hierarchical
