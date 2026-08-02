#include "lunar_path_planner/v3/api/planner_v3.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/api/execution_arbitrator.hpp"
#include "lunar_path_planner/v3/api/reference_bundle_builder.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/crypto/sha256.hpp"
#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"
#include "lunar_path_planner/v3/hopper/hopper_planner.hpp"
#include "lunar_path_planner/v3/legged/legged_planner.hpp"
#include "lunar_path_planner/v3/wheel/wheel_planner.hpp"

namespace lunar::planning::v3 {
namespace {

constexpr double kPoseMatchToleranceM = 1.0e-9;

[[nodiscard]] ReasonCode ReasonForError(const ErrorCode code) {
  switch (code) {
    case ErrorCode::kInvalidArgument:
    case ErrorCode::kSchemaMismatch:
    case ErrorCode::kMissingRegistryObject:
      return "INVALID_REQUEST";
    case ErrorCode::kStaleInput:
    case ErrorCode::kInconsistentSnapshot:
      return "STALE_INPUT";
    case ErrorCode::kResourceLimit:
      return "RESOURCE_LIMIT";
    case ErrorCode::kNumericalFailure:
      return "NUMERICAL_FAILURE";
    case ErrorCode::kNoKnownSafeRoute:
      return "NO_KNOWN_SAFE_ROUTE";
  }
  return "INVALID_REQUEST";
}

[[nodiscard]] PlanningOutcome OutcomeForError(
    const ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kInvalidArgument:
    case ErrorCode::kSchemaMismatch:
    case ErrorCode::kMissingRegistryObject:
      return PlanningOutcome::kInvalidRequest;
    case ErrorCode::kStaleInput:
    case ErrorCode::kInconsistentSnapshot:
      return PlanningOutcome::kStaleInput;
    case ErrorCode::kResourceLimit:
      return PlanningOutcome::kResourceLimit;
    case ErrorCode::kNumericalFailure:
      return PlanningOutcome::kNumericalFailure;
    case ErrorCode::kNoKnownSafeRoute:
      return PlanningOutcome::kNoKnownSafeRoute;
  }
  return PlanningOutcome::kInvalidRequest;
}

[[nodiscard]] std::string UpperReason(std::string value) {
  for (char& character : value) {
    const auto byte = static_cast<unsigned char>(character);
    character = std::isalnum(byte) != 0
                    ? static_cast<char>(std::toupper(byte))
                    : '_';
  }
  value.erase(
      std::unique(
          value.begin(), value.end(),
          [](const char left, const char right) {
            return left == '_' && right == '_';
          }),
      value.end());
  while (!value.empty() && value.front() == '_') {
    value.erase(value.begin());
  }
  while (!value.empty() && value.back() == '_') {
    value.pop_back();
  }
  return value.empty() ? "PLANNER_FAILURE" : value;
}

[[nodiscard]] Error ValidationError(
    const ValidationReport& report) {
  if (report.issues.empty()) {
    return {
        ErrorCode::kInvalidArgument,
        "planning_request",
        "request validation failed",
    };
  }
  const ValidationIssue& issue = report.issues.front();
  const bool stale =
      issue.reason_code == "input_time_skew_exceeded";
  const bool missing =
      issue.reason_code == "missing_registry_object";
  return {
      stale ? ErrorCode::kStaleInput
            : (missing ? ErrorCode::kMissingRegistryObject
                       : ErrorCode::kInvalidArgument),
      issue.field_path,
      issue.message,
  };
}

[[nodiscard]] CallDiagnostics BaseDiagnostics(
    const PlanningRequest& request,
    std::string termination_reason) {
  CallDiagnostics diagnostics{
      .api_latency =
          DurationNanoseconds{std::chrono::nanoseconds{0}},
      .termination_reason = std::move(termination_reason),
  };
  if (request.learned_cost_snapshot.has_value()) {
    diagnostics.learned_cost_usage =
        LearnedCostUsage::kFellBackToAnalytic;
    diagnostics.learned_cost_snapshot_ref =
        request.learned_cost_snapshot->snapshot_ref;
  }
  return diagnostics;
}

[[nodiscard]] bool PreviousContextValid(
    const PlanningRequest& request) noexcept {
  if (!request.previous_execution_context.has_value()) {
    return true;
  }
  if (!request.map_snapshot || !request.safety_capability) {
    return false;
  }
  const PreviousExecutionContext& previous =
      *request.previous_execution_context;
  return previous.source_map_snapshot_ref ==
             request.map_snapshot->snapshot_ref() &&
         previous.source_capability_ref ==
             request.safety_capability->content_ref;
}

[[nodiscard]] PlanningResponse FailureResponse(
    const PlanningRequest& request,
    const Error& error,
    const PlanningOutcome outcome,
    const bool previous_context_valid,
    CallDiagnostics diagnostics) noexcept {
  const ReasonCode reason = ReasonForError(error.code);
  if (diagnostics.termination_reason.empty()) {
    diagnostics.termination_reason = reason;
  }
  diagnostics.message_codes.push_back(
      UpperReason(
          error.field_path.empty() ? error.message
                                   : error.field_path));
  return ArbitratePlanningResponse(
      PlanningAttempt{
          .request_id = request.request_id,
          .response_time = request.request_time,
          .failure = error,
          .planning_outcome = outcome,
          .reason_code = reason,
          .call_diagnostics = std::move(diagnostics),
          .previous_execution_context_valid =
              previous_context_valid,
      },
      request.previous_execution_context,
      request.current_state);
}

template <class T>
[[nodiscard]] bool SameBinding(
    const ResolvedBinding<T>& left,
    const ResolvedBinding<T>& right) noexcept {
  return left.content_ref == right.content_ref &&
         left.object.get() == right.object.get();
}

template <class T>
[[nodiscard]] bool SameOptionalBinding(
    const std::optional<ResolvedBinding<T>>& left,
    const std::optional<ResolvedBinding<T>>& right) noexcept {
  return left.has_value() == right.has_value() &&
         (!left.has_value() || SameBinding(*left, *right));
}

[[nodiscard]] bool SameBindings(
    const ResolvedCapabilityBindings& left,
    const ResolvedCapabilityBindings& right) noexcept {
  return SameBinding(left.motion_model, right.motion_model) &&
         SameBinding(
             left.analytic_cost_model,
             right.analytic_cost_model) &&
         SameOptionalBinding(
             left.gravity_model, right.gravity_model) &&
         SameOptionalBinding(
             left.error_model, right.error_model) &&
         SameOptionalBinding(
             left.actuator_or_impulse_profile,
             right.actuator_or_impulse_profile) &&
         SameOptionalBinding(
             left.body_rotation_envelope,
             right.body_rotation_envelope) &&
         SameOptionalBinding(
             left.attitude_tightening_table,
             right.attitude_tightening_table);
}

[[nodiscard]] std::optional<Error> CheckRegistryRoots(
    const PlanningRequest& request,
    const ContractObjectRegistry& registry) {
  if (!request.map_snapshot || !request.safety_capability ||
      !request.algorithm_config) {
    return Error{
        ErrorCode::kMissingRegistryObject,
        "planning_request",
        "resolved request roots are required",
    };
  }
  const auto map = registry.FindMapSnapshot(
      request.map_snapshot->snapshot_ref(),
      request.map_snapshot->immutable_data_handle());
  if (!map || map.get() != request.map_snapshot.get()) {
    return Error{
        ErrorCode::kInconsistentSnapshot,
        "map_snapshot",
        "registry did not reproduce the exact immutable map snapshot",
    };
  }
  const auto capability = registry.FindSafetyCapability(
      request.safety_capability->content_ref);
  if (!capability ||
      capability.get() != request.safety_capability.get()) {
    return Error{
        ErrorCode::kMissingRegistryObject,
        "safety_capability",
        "registry did not reproduce the exact safety capability",
    };
  }
  const auto config = registry.FindAlgorithmConfig(
      request.algorithm_config->content_ref);
  if (!config || config.get() != request.algorithm_config.get()) {
    return Error{
        ErrorCode::kMissingRegistryObject,
        "algorithm_config",
        "registry did not reproduce the exact algorithm config",
    };
  }
  const auto bindings = registry.ResolveCapabilityBindings(
      *request.safety_capability);
  if (!IsOk(bindings)) {
    return std::get<Error>(bindings);
  }
  if (!SameBindings(
          std::get<ResolvedCapabilityBindings>(bindings),
          request.capability_bindings)) {
    return Error{
        ErrorCode::kInconsistentSnapshot,
        "capability_bindings",
        "registry capability bindings differ from request-fixed objects",
    };
  }
  if (request.learned_cost_snapshot.has_value()) {
    const LearnedCostSnapshotBinding& learned =
        *request.learned_cost_snapshot;
    const auto registered = registry.FindLearnedCost(
        learned.snapshot_ref, learned.registry_handle);
    if (!registered ||
        registered.get() != learned.resolved_snapshot.get()) {
      return Error{
          ErrorCode::kInconsistentSnapshot,
          "learned_cost_snapshot",
          "registry learned-cost snapshot differs from request binding",
      };
    }
  }
  return std::nullopt;
}

[[nodiscard]] Result<std::shared_ptr<const SafeProjection>>
ResolveProjection(
    const PlanningRequest& request,
    SafeProjectionCache& cache) {
  const std::optional<ContentRef> learned_ref =
      request.learned_cost_snapshot.has_value()
          ? std::optional<ContentRef>{
                request.learned_cost_snapshot->snapshot_ref}
          : std::nullopt;
  const SafeProjectionCacheKey key =
      MakeSafeProjectionCacheKey(
          request.map_snapshot->snapshot_ref().id,
          std::to_string(request.map_snapshot->map_revision()),
          std::string{request.map_snapshot->LayerManifestHash()},
          request.safety_capability->content_ref,
          request.algorithm_config->content_ref,
          learned_ref, request.frame_id,
          request.map_snapshot->geometry().resolution_m,
          request.algorithm_config->error_bound_model_id);
  if (const auto cached = cache.Get(key); cached) {
    return cached;
  }
  auto built = BuildSafeProjection(
      SafeProjectionRequest{
          .map = request.map_snapshot,
          .capability = request.safety_capability,
          .algorithm_config = request.algorithm_config,
          .learned_cost =
              request.learned_cost_snapshot.has_value()
                  ? request.learned_cost_snapshot->resolved_snapshot
                  : nullptr,
      });
  if (!IsOk(built)) {
    return std::get<Error>(built);
  }
  auto projection = std::make_shared<const SafeProjection>(
      std::get<SafeProjection>(std::move(built)));
  cache.Publish(key, projection);
  return projection;
}

[[nodiscard]] Vec3 StatePosition(
    const PlatformState& state) noexcept {
  return std::visit(
      [](const auto& value) { return value.position_m; }, state);
}

[[nodiscard]] std::optional<Cell> PositionCell(
    const GridGeometry& geometry,
    const Vec3& position) noexcept {
  if (!std::isfinite(position.x) ||
      !std::isfinite(position.y) ||
      !std::isfinite(geometry.resolution_m) ||
      geometry.resolution_m <= 0.0) {
    return std::nullopt;
  }
  const double raw_x =
      std::floor(
          (position.x - geometry.origin_m.x) /
          geometry.resolution_m);
  const double raw_y =
      std::floor(
          (position.y - geometry.origin_m.y) /
          geometry.resolution_m);
  if (raw_x < 0.0 || raw_y < 0.0 ||
      raw_x >= static_cast<double>(geometry.width) ||
      raw_y >= static_cast<double>(geometry.height) ||
      raw_x >
          static_cast<double>(
              std::numeric_limits<std::int32_t>::max()) ||
      raw_y >
          static_cast<double>(
              std::numeric_limits<std::int32_t>::max())) {
    return std::nullopt;
  }
  return Cell{
      static_cast<std::int32_t>(raw_x),
      static_cast<std::int32_t>(raw_y),
  };
}

[[nodiscard]] const ReferenceId& ReferenceIdOf(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& value) -> const ReferenceId& {
        return value.reference_id;
      },
      reference);
}

[[nodiscard]] const Sha256Digest& ReferenceHashOf(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& value) -> const Sha256Digest& {
        return value.reference_hash;
      },
      reference);
}

[[nodiscard]] ClockStamp ReferenceTimeOf(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& value) {
        return value.reference_time_origin;
      },
      reference);
}

[[nodiscard]] double NormalizeYaw(const double yaw) noexcept {
  const double two_pi = 2.0 * std::numbers::pi;
  double normalized = std::remainder(yaw, two_pi);
  if (normalized == -std::numbers::pi) {
    normalized = std::numbers::pi;
  }
  return normalized == 0.0 ? 0.0 : normalized;
}

[[nodiscard]] double QuaternionYaw(
    const Quaternion& orientation) noexcept {
  const double sin_yaw =
      2.0 *
      (orientation.w * orientation.z +
       orientation.x * orientation.y);
  const double cos_yaw =
      1.0 -
      2.0 *
          (orientation.y * orientation.y +
           orientation.z * orientation.z);
  return NormalizeYaw(std::atan2(sin_yaw, cos_yaw));
}

[[nodiscard]] PoseXyzYaw CurrentPose(
    const PlatformState& state) noexcept {
  return std::visit(
      [](const auto& value) -> PoseXyzYaw {
        using State = std::decay_t<decltype(value)>;
        if constexpr (
            std::is_same_v<State, WheeledOrLeggedState>) {
          return {value.position_m, value.yaw_rad};
        } else {
          return {
              value.position_m,
              QuaternionYaw(
                  value.orientation_body_to_frame),
          };
        }
      },
      state);
}

[[nodiscard]] Vec3 LandingRegionCenter(
    const NextLandingRegion& region) noexcept {
  Vec2 center{};
  if (!region.convex_polygon.vertices_uv.empty()) {
    for (const Vec2& vertex :
         region.convex_polygon.vertices_uv) {
      center.x += vertex.x;
      center.y += vertex.y;
    }
    const double count = static_cast<double>(
        region.convex_polygon.vertices_uv.size());
    center.x /= count;
    center.y /= count;
  }
  return {
      region.landing_plane.origin_m.x +
          center.x * region.landing_plane.basis_u.x +
          center.y * region.landing_plane.basis_v.x,
      region.landing_plane.origin_m.y +
          center.x * region.landing_plane.basis_u.y +
          center.y * region.landing_plane.basis_v.y,
      region.landing_plane.origin_m.z +
          center.x * region.landing_plane.basis_u.z +
          center.y * region.landing_plane.basis_v.z,
  };
}

[[nodiscard]] PoseXyzYaw ReferenceEndPose(
    const PlatformReference& reference) noexcept {
  return std::visit(
      [](const auto& value) -> PoseXyzYaw {
        using Reference = std::decay_t<decltype(value)>;
        if constexpr (
            std::is_same_v<Reference, WheeledReference> ||
            std::is_same_v<Reference, LeggedBodyReference>) {
          return value.safe_stop_anchor.pose;
        } else {
          return {
              LandingRegionCenter(value.next_landing_region),
              NormalizeYaw(
                  value.next_landing_region
                          .allowed_yaw_interval.start_rad +
                      0.5 *
                          value.next_landing_region
                              .allowed_yaw_interval.span_rad),
          };
        }
      },
      reference);
}

void AppendUniqueRef(
    std::vector<ContentRef>& refs,
    const ContentRef& ref) {
  if (std::find(refs.begin(), refs.end(), ref) == refs.end()) {
    refs.push_back(ref);
  }
}

[[nodiscard]] std::vector<ContentRef> CertificateRefs(
    const PlatformReference& reference) {
  std::vector<ContentRef> refs;
  std::visit(
      [&](const auto& value) {
        using Reference = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Reference, WheeledReference>) {
          AppendUniqueRef(
              refs,
              value.safe_stop_anchor.terrain_certification_ref);
          for (const WheeledSegment& segment : value.segments) {
            const auto* drive =
                std::get_if<DriveSegment>(&segment);
            if (drive == nullptr) {
              continue;
            }
            const auto* chain =
                std::get_if<ValidatedPrimitiveChain>(
                    &drive->geometric_path);
            if (chain == nullptr) {
              continue;
            }
            for (const ValidatedPrimitive& primitive :
                 chain->primitives) {
              AppendUniqueRef(refs, primitive.validation_ref);
            }
          }
        } else if constexpr (
            std::is_same_v<Reference, LeggedBodyReference>) {
          AppendUniqueRef(
              refs,
              value.terrain_normal_envelope
                  .source_terrain_certification_ref);
          AppendUniqueRef(
              refs,
              value.safe_stop_anchor.terrain_certification_ref);
          if (const auto* chain =
                  std::get_if<ValidatedPrimitiveChain>(
                      &value.geometric_path);
              chain != nullptr) {
            for (const ValidatedPrimitive& primitive :
                 chain->primitives) {
              AppendUniqueRef(refs, primitive.validation_ref);
            }
          }
        } else {
          AppendUniqueRef(
              refs,
              value.ground_hold_anchor
                  .terrain_certification_ref);
          AppendUniqueRef(
              refs,
              value.next_landing_region
                  .terrain_certification_ref);
          AppendUniqueRef(
              refs,
              value.attitude_boundary.certification_ref);
          AppendUniqueRef(
              refs, value.physical_certification_ref);
        }
      },
      reference);
  return refs;
}

[[nodiscard]] GenerationMode GenerationModeOf(
    const PlatformReference& reference) noexcept {
  return std::visit(
      [](const auto& value) {
        using Reference = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Reference, HopperReference>) {
          return GenerationMode::kCertifiedBallisticReference;
        } else if constexpr (
            std::is_same_v<Reference, LeggedBodyReference>) {
          return std::holds_alternative<ClampedCubicBSplinePath>(
                     value.geometric_path)
                     ? GenerationMode::kSmoothedSplineReference
                     : GenerationMode::
                           kValidatedPrimitiveChainReference;
        } else {
          const bool smoothed = std::any_of(
              value.segments.begin(), value.segments.end(),
              [](const WheeledSegment& segment) {
                const auto* drive =
                    std::get_if<DriveSegment>(&segment);
                return drive != nullptr &&
                       std::holds_alternative<
                           ClampedCubicBSplinePath>(
                           drive->geometric_path);
              });
          return smoothed
                     ? GenerationMode::kSmoothedSplineReference
                     : GenerationMode::
                           kValidatedPrimitiveChainReference;
        }
      },
      reference);
}

[[nodiscard]] DurationNanoseconds ReferenceDuration(
    const PlatformReference& reference) noexcept {
  return std::visit(
      [](const auto& value) -> DurationNanoseconds {
        using Reference = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Reference, WheeledReference>) {
          if (value.segments.empty()) {
            return {};
          }
          return std::visit(
              [](const auto& segment) {
                return segment.time_interval.end_offset;
              },
              value.segments.back());
        } else if constexpr (
            std::is_same_v<Reference, LeggedBodyReference>) {
          if (value.time_scaling.segments.empty()) {
            return {};
          }
          return value.time_scaling.segments.back().end_offset;
        } else {
          return value.jump_boundary.ballistic_flight_time;
        }
      },
      reference);
}

[[nodiscard]] std::string SelectedCandidateId(
    const PlatformReference& reference,
    const ResolvedTerminalSet& terminals) {
  if (const auto* hopper =
          std::get_if<HopperReference>(&reference);
      hopper != nullptr) {
    return hopper->next_landing_region.region_id;
  }
  const PoseXyzYaw end = ReferenceEndPose(reference);
  const TerminalCandidate* best = nullptr;
  double best_distance =
      std::numeric_limits<double>::infinity();
  for (const TerminalCandidate& candidate :
       terminals.candidates) {
    const double dx = candidate.position_m.x - end.position_m.x;
    const double dy = candidate.position_m.y - end.position_m.y;
    const double dz = candidate.position_m.z - end.position_m.z;
    const double distance = dx * dx + dy * dy + dz * dz;
    if (distance < best_distance) {
      best = &candidate;
      best_distance = distance;
    }
  }
  if (best != nullptr &&
      best_distance <=
          kPoseMatchToleranceM * kPoseMatchToleranceM) {
    return best->stable_id;
  }
  return best != nullptr ? best->stable_id : ReferenceIdOf(reference);
}

[[nodiscard]] Result<std::string> StableId(
    std::string prefix,
    const PlanningRequest& request,
    const PlatformReference& reference) {
  const auto digest = Sha256Hex(
      prefix + "\n" + request.request_id + "\n" +
      ReferenceIdOf(reference) + "\n" +
      ReferenceHashOf(reference));
  if (!IsOk(digest)) {
    return std::get<Error>(digest);
  }
  return prefix + "-" +
         std::get<Sha256Digest>(digest).substr(0U, 32U);
}

struct ViewPair final {
  ReferenceViewContent committed;
  ReferenceViewContent preview;
};

[[nodiscard]] ViewPair MakeViews(
    const PlatformReference& reference) {
  const ReferenceId id = ReferenceIdOf(reference);
  const Sha256Digest hash = ReferenceHashOf(reference);
  return std::visit(
      [&](const auto& value) -> ViewPair {
        using Reference = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Reference, WheeledReference>) {
          const std::size_t first_end =
              std::min<std::size_t>(1U, value.segments.size());
          return {
              {
                  .role =
                      ReferenceViewContent::Role::kCommittedPrefix,
                  .source_reference_id = id,
                  .source_reference_hash = hash,
                  .selector =
                      SegmentViewSelector{0U, first_end},
              },
              {
                  .role = ReferenceViewContent::Role::kPreview,
                  .source_reference_id = id,
                  .source_reference_hash = hash,
                  .selector =
                      SegmentViewSelector{
                          first_end, value.segments.size()},
              },
          };
        } else if constexpr (
            std::is_same_v<Reference, LeggedBodyReference>) {
          const DurationNanoseconds start =
              value.time_scaling.segments.front().start_offset;
          const DurationNanoseconds committed_end =
              value.time_scaling.segments.front().end_offset;
          const DurationNanoseconds end =
              value.time_scaling.segments.back().end_offset;
          return {
              {
                  .role =
                      ReferenceViewContent::Role::kCommittedPrefix,
                  .source_reference_id = id,
                  .source_reference_hash = hash,
                  .selector =
                      TimeViewSelector{{start, committed_end}},
              },
              {
                  .role = ReferenceViewContent::Role::kPreview,
                  .source_reference_id = id,
                  .source_reference_hash = hash,
                  .selector =
                      TimeViewSelector{{committed_end, end}},
              },
          };
        } else {
          return {
              {
                  .role =
                      ReferenceViewContent::Role::kCommittedPrefix,
                  .source_reference_id = id,
                  .source_reference_hash = hash,
                  .selector =
                      GroundHoldViewSelector{
                          value.ground_hold_anchor.anchor_id},
              },
              {
                  .role = ReferenceViewContent::Role::kPreview,
                  .source_reference_id = id,
                  .source_reference_hash = hash,
                  .selector =
                      JumpViewSelector{
                          value.jump_boundary.boundary_id,
                          JumpViewSelector::Scope::kNextHop},
              },
          };
        }
      },
      reference);
}

[[nodiscard]] std::variant<
    WheeledOrLeggedErrorBounds, HopperErrorBounds>
CertifiedErrorBounds(
    const SafetyCapabilityProfile& profile) {
  return std::visit(
      [](const auto& capability)
          -> std::variant<
              WheeledOrLeggedErrorBounds, HopperErrorBounds> {
        using Capability = std::decay_t<decltype(capability)>;
        if constexpr (std::is_same_v<Capability, HopperCapability>) {
          return capability.certified_state_error_bounds;
        } else {
          return capability.certified_state_error_bounds;
        }
      },
      profile.content);
}

[[nodiscard]] Result<BundleBuildRequest> MakeBundleRequest(
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminals,
    const SafeProjection& projection,
    PlatformReference reference) {
  const auto bundle_id =
      StableId("bundle", request, reference);
  const auto route_id =
      StableId("route", request, reference);
  const auto committed_id =
      StableId("committed", request, reference);
  const auto preview_id =
      StableId("preview", request, reference);
  if (!IsOk(bundle_id)) {
    return std::get<Error>(bundle_id);
  }
  if (!IsOk(route_id)) {
    return std::get<Error>(route_id);
  }
  if (!IsOk(committed_id)) {
    return std::get<Error>(committed_id);
  }
  if (!IsOk(preview_id)) {
    return std::get<Error>(preview_id);
  }

  std::vector<Vec3> unresolved_tail;
  if (terminals.unresolved_tail.has_value()) {
    unresolved_tail =
        terminals.unresolved_tail->intent_polyline_m;
  }
  const ViewPair views = MakeViews(reference);
  const std::vector<ContentRef> certificates =
      CertificateRefs(reference);
  if (certificates.empty()) {
    return Error{
        ErrorCode::kInvalidArgument,
        "platform_reference",
        "platform reference carries no activation certificate",
    };
  }
  const ClockStamp origin = ReferenceTimeOf(reference);
  const ReferenceId reference_id = ReferenceIdOf(reference);
  const Sha256Digest reference_hash = ReferenceHashOf(reference);
  const PoseXyzYaw current_pose =
      CurrentPose(request.current_state);
  const PoseXyzYaw end_pose = ReferenceEndPose(reference);
  const std::string selected_candidate_id =
      SelectedCandidateId(reference, terminals);
  const GenerationMode generation_mode =
      GenerationModeOf(reference);
  const bool learned_used =
      projection.soft_cost_source ==
      SoftCostSource::kAnalyticPlusPinnedLearned;
  const std::string termination_reason =
      terminals.reason_code.empty()
          ? "REFERENCE_READY"
          : UpperReason(terminals.reason_code);

  return BundleBuildRequest{
      .bundle_id = std::get<std::string>(bundle_id),
      .bundle_revision = 1U,
      .source_request_id = request.request_id,
      .source_map_snapshot_ref =
          request.map_snapshot->snapshot_ref(),
      .source_safety_capability_ref =
          request.safety_capability->content_ref,
      .source_algorithm_config_ref =
          request.algorithm_config->content_ref,
      .platform_reference = std::move(reference),
      .route_skeleton =
          {
              std::get<std::string>(route_id),
              RouteSkeletonContent{
                  .source_reference_id =
                      reference_id,
                  .source_reference_hash =
                      reference_hash,
                  .waypoints =
                      {
                          current_pose,
                          end_pose,
                      },
                  .unresolved_tail =
                      std::move(unresolved_tail),
              },
          },
      .committed_prefix =
          {
              std::get<std::string>(committed_id),
              views.committed,
          },
      .preview =
          {
              std::get<std::string>(preview_id),
              views.preview,
          },
      .validity =
          {
              .valid_from = origin,
              .required_map_snapshot_ref =
                  request.map_snapshot->snapshot_ref(),
              .required_capability_ref =
                  request.safety_capability->content_ref,
              .allowed_state_deviation =
                  CertifiedErrorBounds(
                      *request.safety_capability),
              .invalidation_conditions =
                  {
                      InvalidationCondition::
                          kMapSafetyRevisionChanged,
                      InvalidationCondition::
                          kStateDeviationExceeded,
                      InvalidationCondition::
                          kCapabilityRevisionChanged,
                      InvalidationCondition::
                          kReferenceHorizonExhausted,
                  },
          },
      .validation_summary =
          {
              .hard_constraints_passed = true,
              .continuous_validation_passed = true,
              .certificate_refs = certificates,
          },
      .generation_evidence =
          {
              .selected_candidate_id =
                  selected_candidate_id,
              .generation_mode = generation_mode,
              .termination_reason = termination_reason,
              .evidence_refs = certificates,
              .learned_cost_snapshot_ref =
                  learned_used &&
                          request.learned_cost_snapshot.has_value()
                      ? std::optional<ContentRef>{
                            request.learned_cost_snapshot
                                ->snapshot_ref}
                      : std::nullopt,
          },
  };
}

[[nodiscard]] Result<PlatformReference> DispatchPlatform(
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminals,
    WheelPlanner* wheel,
    LeggedPlanner* legged,
    HopperPlanner* hopper) {
  switch (request.platform_type) {
    case PlatformType::kWheeled: {
      if (wheel == nullptr) {
        return Error{
            ErrorCode::kMissingRegistryObject,
            "wheel_planner",
            "wheel planner dependency is absent",
        };
      }
      auto result = wheel->Plan(request, terminals);
      if (!IsOk(result)) {
        return std::get<Error>(result);
      }
      return PlatformReference{
          std::get<WheeledReference>(std::move(result))};
    }
    case PlatformType::kLegged: {
      if (legged == nullptr) {
        return Error{
            ErrorCode::kMissingRegistryObject,
            "legged_planner",
            "legged planner dependency is absent",
        };
      }
      auto result = legged->Plan(request, terminals);
      if (!IsOk(result)) {
        return std::get<Error>(result);
      }
      return PlatformReference{
          std::get<LeggedBodyReference>(std::move(result))};
    }
    case PlatformType::kHopper: {
      if (hopper == nullptr) {
        return Error{
            ErrorCode::kMissingRegistryObject,
            "hopper_planner",
            "hopper planner dependency is absent",
        };
      }
      auto result = hopper->Plan(request, terminals);
      if (!IsOk(result)) {
        return std::get<Error>(result);
      }
      return PlatformReference{
          std::get<HopperReference>(std::move(result))};
    }
  }
  return Error{
      ErrorCode::kInvalidArgument,
      "platform_type",
      "unsupported platform type",
  };
}

[[nodiscard]] PlanningOutcome ReadyOutcome(
    const ResolvedTerminalSet& terminals) noexcept {
  return terminals.kind == TerminalKind::kSafeFrontier
             ? PlanningOutcome::kSafeFrontierReferenceReady
             : PlanningOutcome::kNewReferenceReady;
}

[[nodiscard]] ReasonCode ReadyReason(
    const ResolvedTerminalSet& terminals) {
  return terminals.kind == TerminalKind::kSafeFrontier
             ? "SAFE_FRONTIER_REFERENCE_READY"
             : "NEW_REFERENCE_READY";
}

}  // namespace

PlannerV3Impl::PlannerV3Impl(
    const SemanticValidator& semantic_validator,
    const ContractObjectRegistry& contract_registry,
    SafeProjectionCache& projection_cache,
    std::unique_ptr<WheelPlanner> wheel,
    std::unique_ptr<LeggedPlanner> legged,
    std::unique_ptr<HopperPlanner> hopper) noexcept
    : semantic_validator_(semantic_validator),
      contract_registry_(contract_registry),
      projection_cache_(projection_cache),
      wheel_(std::move(wheel)),
      legged_(std::move(legged)),
      hopper_(std::move(hopper)) {}

PlannerV3Impl::~PlannerV3Impl() = default;

PlanningResponse PlannerV3Impl::Plan(
    const PlanningRequest& request) noexcept {
  try {
    const bool previous_valid = PreviousContextValid(request);
    const ValidationReport request_report =
        semantic_validator_.Validate(request);
    if (!request_report.ok()) {
      const Error error = ValidationError(request_report);
      return FailureResponse(
          request, error, OutcomeForError(error.code),
          previous_valid,
          BaseDiagnostics(
              request, ReasonForError(error.code)));
    }
    if (const auto registry_error =
            CheckRegistryRoots(request, contract_registry_);
        registry_error.has_value()) {
      return FailureResponse(
          request, *registry_error,
          OutcomeForError(registry_error->code),
          previous_valid,
          BaseDiagnostics(
              request,
              ReasonForError(registry_error->code)));
    }

    auto projection_result =
        ResolveProjection(request, projection_cache_);
    if (!IsOk(projection_result)) {
      const Error& error = std::get<Error>(projection_result);
      return FailureResponse(
          request, error, OutcomeForError(error.code),
          previous_valid,
          BaseDiagnostics(
              request, ReasonForError(error.code)));
    }
    const auto& projection =
        *std::get<std::shared_ptr<const SafeProjection>>(
            projection_result);
    const auto start =
        PositionCell(
            projection.geometry,
            StatePosition(request.current_state));
    if (!start.has_value()) {
      const Error error{
          ErrorCode::kNoKnownSafeRoute,
          "current_state.position_m",
          "current state is outside the immutable map",
      };
      return FailureResponse(
          request, error,
          PlanningOutcome::kNoKnownSafeRoute,
          previous_valid,
          BaseDiagnostics(request, "NO_KNOWN_SAFE_ROUTE"));
    }

    auto terminal_result = ResolveTerminal(
        TerminalResolutionRequest{
            .projection = projection,
            .goal_region = request.goal,
            .start_cell = *start,
            .capability = *request.safety_capability,
            .algorithm_config = *request.algorithm_config,
        });
    if (!IsOk(terminal_result)) {
      const Error& error = std::get<Error>(terminal_result);
      return FailureResponse(
          request, error, OutcomeForError(error.code),
          previous_valid,
          BaseDiagnostics(
              request, ReasonForError(error.code)));
    }
    const ResolvedTerminalSet& terminals =
        std::get<ResolvedTerminalSet>(terminal_result);
    if (terminals.kind == TerminalKind::kGoalInfeasible) {
      const Error error{
          ErrorCode::kNoKnownSafeRoute,
          "goal",
          "goal is known and hard infeasible",
      };
      return FailureResponse(
          request, error,
          PlanningOutcome::kGoalInfeasible,
          previous_valid,
          BaseDiagnostics(request, "GOAL_INFEASIBLE"));
    }
    if (terminals.kind == TerminalKind::kNoKnownSafeRoute ||
        terminals.candidates.empty()) {
      const Error error{
          ErrorCode::kNoKnownSafeRoute,
          "terminal_set",
          "no known safe terminal is available",
      };
      return FailureResponse(
          request, error,
          PlanningOutcome::kNoKnownSafeRoute,
          previous_valid,
          BaseDiagnostics(request, "NO_KNOWN_SAFE_ROUTE"));
    }

    auto platform_result = DispatchPlatform(
        request, terminals, wheel_.get(), legged_.get(),
        hopper_.get());
    if (!IsOk(platform_result)) {
      const Error& error = std::get<Error>(platform_result);
      return FailureResponse(
          request, error, OutcomeForError(error.code),
          previous_valid,
          BaseDiagnostics(
              request, ReasonForError(error.code)));
    }
    PlatformReference platform_reference =
        std::get<PlatformReference>(std::move(platform_result));
    const ValidationReport platform_report =
        semantic_validator_.Validate(platform_reference);
    if (!platform_report.ok()) {
      const Error error{
          ErrorCode::kNumericalFailure,
          "platform_reference",
          platform_report.issues.front().reason_code,
      };
      return FailureResponse(
          request, error,
          PlanningOutcome::kNumericalFailure,
          previous_valid,
          BaseDiagnostics(request, "NUMERICAL_FAILURE"));
    }

    auto build_request = MakeBundleRequest(
        request, terminals, projection,
        std::move(platform_reference));
    if (!IsOk(build_request)) {
      const Error& error = std::get<Error>(build_request);
      return FailureResponse(
          request, error, OutcomeForError(error.code),
          previous_valid,
          BaseDiagnostics(
              request, ReasonForError(error.code)));
    }
    const ReferenceActivationContext activation_context{
        request, contract_registry_};
    auto bundle = BuildReferenceBundle(
        std::get<BundleBuildRequest>(
            std::move(build_request)),
        activation_context);
    if (!IsOk(bundle)) {
      const Error& error = std::get<Error>(bundle);
      return FailureResponse(
          request, error, OutcomeForError(error.code),
          previous_valid,
          BaseDiagnostics(
              request, ReasonForError(error.code)));
    }

    CallDiagnostics diagnostics =
        BaseDiagnostics(request, ReadyReason(terminals));
    diagnostics.final_epsilon =
        request.algorithm_config->ara_star.target_epsilon;
    diagnostics.expected_execution_time =
        ReferenceDuration(
            std::get<ReferenceBundle>(bundle)
                .platform_reference);
    diagnostics.candidate_count =
        static_cast<std::uint64_t>(
            terminals.candidates.size());
    if (request.learned_cost_snapshot.has_value()) {
      diagnostics.learned_cost_usage =
          projection.soft_cost_source ==
                  SoftCostSource::kAnalyticPlusPinnedLearned
              ? LearnedCostUsage::kUsedBoundedSoftCost
              : LearnedCostUsage::kFellBackToAnalytic;
      if (projection.soft_cost_fallback_reason_code.has_value()) {
        diagnostics.message_codes.push_back(
            UpperReason(
                *projection.soft_cost_fallback_reason_code));
      }
    }

    PlanningResponse response = ArbitratePlanningResponse(
        PlanningAttempt{
            .request_id = request.request_id,
            .response_time = request.request_time,
            .validated_new_bundle =
                std::get<ReferenceBundle>(std::move(bundle)),
            .planning_outcome = ReadyOutcome(terminals),
            .reason_code = ReadyReason(terminals),
            .call_diagnostics = std::move(diagnostics),
            .previous_execution_context_valid = previous_valid,
        },
        request.previous_execution_context,
        request.current_state);
    const ValidationReport response_report =
        semantic_validator_.ValidateForActivation(
            response, activation_context);
    if (!response_report.ok()) {
      const Error error{
          ErrorCode::kNumericalFailure,
          "planning_response",
          response_report.issues.front().reason_code,
      };
      return FailureResponse(
          request, error,
          PlanningOutcome::kNumericalFailure,
          previous_valid,
          BaseDiagnostics(request, "FINAL_VALIDATION_FAILED"));
    }
    return response;
  } catch (const std::bad_alloc&) {
    return FailureResponse(
        request,
        Error{
            ErrorCode::kResourceLimit,
            "planner_v3",
            "planner allocation failed",
        },
        PlanningOutcome::kResourceLimit,
        PreviousContextValid(request),
        BaseDiagnostics(request, "RESOURCE_LIMIT"));
  } catch (const std::exception& exception) {
    return FailureResponse(
        request,
        Error{
            ErrorCode::kNumericalFailure,
            "planner_v3",
            exception.what(),
        },
        PlanningOutcome::kNumericalFailure,
        PreviousContextValid(request),
        BaseDiagnostics(request, "NUMERICAL_FAILURE"));
  } catch (...) {
    return FailureResponse(
        request,
        Error{
            ErrorCode::kNumericalFailure,
            "planner_v3",
            "unknown planner failure",
        },
        PlanningOutcome::kNumericalFailure,
        PreviousContextValid(request),
        BaseDiagnostics(request, "NUMERICAL_FAILURE"));
  }
}

std::unique_ptr<PlannerV3> MakeDefaultPlannerV3(
    const SemanticValidator& semantic_validator,
    const ContractObjectRegistry& contract_registry,
    SafeProjectionCache& projection_cache) {
  return std::make_unique<PlannerV3Impl>(
      semantic_validator, contract_registry, projection_cache,
      std::make_unique<WheelPlanner>(),
      std::make_unique<LeggedPlanner>(),
      std::make_unique<HopperPlanner>());
}

}  // namespace lunar::planning::v3
