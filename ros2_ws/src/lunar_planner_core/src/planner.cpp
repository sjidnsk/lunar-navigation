#include "lunar_planner_core/planner.hpp"

#include <array>
#include <cctype>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "wheel/wheel_planner.hpp"

#ifdef LUNAR_HAS_LEGACY_V3
#include "migration/legacy_v3_adapter.hpp"
#endif

namespace lunar::planning {
namespace {

constexpr std::array<std::string_view, 10> kRequiredMapLayers{
    "elevation",
    "valid_mask",
    "obstacle",
    "obstacle_height",
    "observation_age_s",
    "observation_quality",
    "elevation_variance",
    "obstacle_variance",
    "observation_count",
    "forbidden",
};

[[nodiscard]] PlannerOutput Failure(
    const PlanningOutcome outcome,
    const ExecutionDirective directive,
    std::string reason_code) {
  return PlannerOutput{
      .outcome = outcome,
      .directive = directive,
      .reason_code = std::move(reason_code),
      .reference = std::nullopt,
      .diagnostics = {},
  };
}

[[nodiscard]] std::string MissingLayerReason(const std::string_view layer) {
  std::string reason{"MISSING_MAP_LAYER_"};
  reason.reserve(reason.size() + layer.size());
  for (const char character : layer) {
    reason.push_back(static_cast<char>(
        std::toupper(static_cast<unsigned char>(character))));
  }
  return reason;
}

[[nodiscard]] std::string MissingRequiredLayer(const WorldSnapshot& world) {
  for (const GridMap* map : {&world.global_map, &world.local_map}) {
    for (const std::string_view layer : kRequiredMapLayers) {
      if (!map->HasLayer(layer)) {
        return MissingLayerReason(layer);
      }
    }
  }
  return {};
}

}  // namespace

struct Planner::Impl final {
  wheel::WheelPlanner wheel_planner;
#ifdef LUNAR_HAS_LEGACY_V3
  LegacyV3Adapter adapter;
#endif
};

Planner::Planner() : impl_(std::make_unique<Impl>()) {}

Planner::~Planner() = default;

Planner::Planner(Planner&&) noexcept = default;

Planner& Planner::operator=(Planner&&) noexcept = default;

PlannerOutput Planner::Plan(const PlannerInput& input) noexcept {
  try {
    if (input.stop_token.stop_requested()) {
      return Failure(
          PlanningOutcome::kCanceled,
          ExecutionDirective::kHoldPosition,
          "REQUEST_CANCELED");
    }
    if (impl_ == nullptr) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference,
          "PLANNER_MOVED_FROM");
    }
    if (std::string reason = MissingRequiredLayer(input.world); !reason.empty()) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference,
          std::move(reason));
    }
    if (CapabilityPlatform(input.capability) == PlatformType::kWheeled) {
      return impl_->wheel_planner.Plan(input);
    }
#ifdef LUNAR_HAS_LEGACY_V3
    return impl_->adapter.Plan(input);
#else
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "PLANNER_BACKEND_NOT_CONFIGURED");
#endif
  } catch (const std::bad_alloc&) {
    return Failure(
        PlanningOutcome::kResourceExhausted,
        ExecutionDirective::kNoSafeReference,
        "RESOURCE_EXHAUSTED");
  } catch (const std::exception&) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kNoSafeReference,
        "INTERNAL_PLANNER_EXCEPTION");
  } catch (...) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kNoSafeReference,
        "INTERNAL_PLANNER_EXCEPTION");
  }
}

}  // namespace lunar::planning
