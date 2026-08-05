#include "hierarchical/reference_composer.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace lunar::planning::hierarchical {
namespace {

[[nodiscard]] ReferenceComposeResult Failure(std::string reason_code) {
  return ReferenceComposeResult{
      .reference = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Quaternion value) noexcept {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
         std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] bool ValidPreview(const GlobalRoute &route) noexcept {
  if (route.poses_map.empty()) {
    return false;
  }
  for (const Pose3 &pose : route.poses_map) {
    if (!Finite(pose.position_m) || !Finite(pose.orientation)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool SuccessfulLocalOutput(const PlannerOutput &output) noexcept {
  return output.outcome == PlanningOutcome::kNewReferenceAvailable ||
         output.outcome == PlanningOutcome::kSafeFrontierReferenceAvailable;
}

[[nodiscard]] bool ValidLocalData(const PlatformType platform,
                                  const MotionReferenceData &data) noexcept {
  if (const auto *trajectory = std::get_if<TrajectoryReference>(&data)) {
    if (trajectory->points.empty()) {
      return false;
    }
    return (platform == PlatformType::kWheeled &&
            trajectory->semantics == TrajectorySemantics::kWheeledBase) ||
           (platform == PlatformType::kLegged &&
            trajectory->semantics == TrajectorySemantics::kLeggedBodyReference);
  }
  const auto *hops = std::get_if<HopReference>(&data);
  return platform == PlatformType::kHopper && hops != nullptr &&
         hops->segments.size() == 1U;
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

} // namespace

ReferenceComposeResult ComposeReference(const PlannerInput &input,
                                        const GlobalRoute &global_route,
                                        PlannerOutput local_output) {
  if (global_route.poses_map.empty()) {
    return Failure("REFERENCE_GLOBAL_PREVIEW_EMPTY");
  }
  if (!ValidPreview(global_route)) {
    return Failure("REFERENCE_GLOBAL_PREVIEW_NONFINITE");
  }
  if (!SuccessfulLocalOutput(local_output) ||
      !local_output.reference.has_value()) {
    return Failure("REFERENCE_LOCAL_DATA_MISSING");
  }
  const PlatformType platform = CapabilityPlatform(input.capability);
  if (local_output.reference->platform_type != platform) {
    return Failure("REFERENCE_PLATFORM_MISMATCH");
  }
  if (platform == PlatformType::kHopper) {
    const auto *hops = std::get_if<HopReference>(&local_output.reference->data);
    if (hops == nullptr || hops->segments.size() != 1U) {
      return Failure("REFERENCE_HOP_AUTHORIZATION_INVALID");
    }
  } else if (!ValidLocalData(platform, local_output.reference->data)) {
    return Failure("REFERENCE_LOCAL_DATA_INVALID");
  }
  const std::string prefix = PlanPrefix(platform);
  if (input.request_id.empty() || prefix.empty()) {
    return Failure("REFERENCE_REQUEST_IDENTITY_INVALID");
  }
  return ReferenceComposeResult{
      .reference =
          MotionReference{
              .plan_id = prefix + input.request_id,
              .platform_type = platform,
              .input_time = input.state_time,
              .preview =
                  GlobalRoutePreview{
                      .poses_map = global_route.poses_map,
                  },
              .data = std::move(local_output.reference->data),
          },
      .reason_code = {},
  };
}

} // namespace lunar::planning::hierarchical
