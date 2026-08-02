#include "lunar_planner_core/planner.hpp"

#include <array>
#include <cctype>
#include <exception>
#include <new>
#include <string>
#include <string_view>
#include <utility>

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

struct Planner::Impl final {};

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
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "PLANNER_BACKEND_NOT_CONFIGURED");
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
