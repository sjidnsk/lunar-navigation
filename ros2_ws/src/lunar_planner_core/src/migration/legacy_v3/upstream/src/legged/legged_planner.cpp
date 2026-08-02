#include "lunar_path_planner/v3/legged/legged_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/legged/body_motion_primitive.hpp"
#include "lunar_path_planner/v3/legged/legged_search.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"
#include "lunar_path_planner/v3/search/ara_star.hpp"

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-9;

[[nodiscard]] Error Failure(ErrorCode code, std::string path,
                            std::string message) {
  return Error{
      .code = code,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool FiniteState(
    const WheeledOrLeggedState& state) noexcept {
  return std::isfinite(state.position_m.x) &&
         std::isfinite(state.position_m.y) &&
         std::isfinite(state.position_m.z) &&
         std::isfinite(state.yaw_rad) &&
         std::isfinite(state.linear_velocity_mps.x) &&
         std::isfinite(state.linear_velocity_mps.y) &&
         std::isfinite(state.linear_velocity_mps.z) &&
         std::isfinite(state.yaw_rate_radps);
}

[[nodiscard]] bool ValidContentRef(
    const ContentRef& reference) noexcept {
  return !reference.id.empty() && reference.revision > 0U &&
         reference.content_hash.size() == 64U;
}

[[nodiscard]] Result<std::monostate> ValidateBindings(
    const PlanningRequest& request,
    const LeggedCapability& capability) {
  const auto& bindings = request.capability_bindings;
  if (!bindings.motion_model.object ||
      bindings.motion_model.content_ref !=
          capability.motion_model_ref) {
    return Failure(
        ErrorCode::kMissingRegistryObject,
        "/capability_bindings/motion_model",
        "LEGGED_MOTION_MODEL_BINDING_MISMATCH");
  }
  if (!bindings.analytic_cost_model.object ||
      bindings.analytic_cost_model.content_ref !=
          capability.analytic_cost_model_ref) {
    return Failure(
        ErrorCode::kMissingRegistryObject,
        "/capability_bindings/analytic_cost_model",
        "LEGGED_ANALYTIC_COST_BINDING_MISMATCH");
  }
  if (bindings.gravity_model.has_value() ||
      bindings.error_model.has_value() ||
      bindings.actuator_or_impulse_profile.has_value() ||
      bindings.body_rotation_envelope.has_value() ||
      bindings.attitude_tightening_table.has_value()) {
    return Failure(
        ErrorCode::kInvalidArgument,
        "/capability_bindings",
        "LEGGED_REQUEST_HAS_HOPPER_ONLY_BINDING");
  }
  return std::monostate{};
}

[[nodiscard]] Result<std::monostate> ValidateRequest(
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminal_set) {
  if (request.request_id.empty() ||
      request.request_time.clock_id.empty() ||
      request.frame_id.empty() ||
      request.platform_type != PlatformType::kLegged ||
      !std::holds_alternative<WheeledOrLeggedState>(
          request.current_state) ||
      !request.map_snapshot || !request.safety_capability ||
      !request.algorithm_config) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/planning_request",
                   "INCOMPLETE_LEGGED_PLANNING_REQUEST");
  }
  const auto& state =
      std::get<WheeledOrLeggedState>(request.current_state);
  if (!FiniteState(state)) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/current_state",
                   "NONFINITE_LEGGED_CURRENT_STATE");
  }
  if (!std::holds_alternative<LeggedCapability>(
          request.safety_capability->content)) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/safety_capability/content",
                   "LEGGED_CAPABILITY_REQUIRED");
  }
  const auto& capability = std::get<LeggedCapability>(
      request.safety_capability->content);
  if (request.map_snapshot->frame_id() != request.frame_id ||
      capability.frame_id != request.frame_id) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/frame_id",
                   "LEGGED_FRAME_MISMATCH");
  }
  if (!ValidContentRef(request.map_snapshot->snapshot_ref()) ||
      !ValidContentRef(request.safety_capability->content_ref) ||
      !ValidContentRef(request.algorithm_config->content_ref)) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/planning_request/content_refs",
                   "INVALID_LEGGED_CONTENT_REFERENCE");
  }
  const auto binding_result =
      ValidateBindings(request, capability);
  if (!IsOk(binding_result)) {
    return std::get<Error>(binding_result);
  }
  if (terminal_set.kind != TerminalKind::kGoal &&
      terminal_set.kind != TerminalKind::kSafeFrontier) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/resolved_terminal_set",
                   "NO_EXECUTABLE_LEGGED_TERMINAL_SET");
  }
  const auto maximum_terminals =
      request.algorithm_config->legged.pose_lattice
          .maximum_terminal_candidates;
  if (terminal_set.candidates.empty() ||
      maximum_terminals == 0U ||
      terminal_set.candidates.size() > maximum_terminals) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/resolved_terminal_set/candidates",
                   "INVALID_LEGGED_TERMINAL_CANDIDATES");
  }
  if (request.learned_cost_snapshot.has_value()) {
    const auto& binding = *request.learned_cost_snapshot;
    if (!binding.resolved_snapshot ||
        binding.snapshot_ref !=
            binding.resolved_snapshot->snapshot_ref()) {
      return Failure(
          ErrorCode::kMissingRegistryObject,
          "/learned_cost_snapshot",
          "LEGGED_LEARNED_COST_BINDING_MISMATCH");
    }
  }
  return std::monostate{};
}

[[nodiscard]] std::optional<Cell> PositionCell(
    const GridGeometry& geometry,
    const Vec3& position) noexcept {
  if (!std::isfinite(geometry.resolution_m) ||
      geometry.resolution_m <= 0.0 ||
      !std::isfinite(position.x) ||
      !std::isfinite(position.y)) {
    return std::nullopt;
  }
  const double x =
      (position.x - geometry.origin_m.x) /
      geometry.resolution_m;
  const double y =
      (position.y - geometry.origin_m.y) /
      geometry.resolution_m;
  if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 ||
      y < 0.0 || x >= static_cast<double>(geometry.width) ||
      y >= static_cast<double>(geometry.height)) {
    return std::nullopt;
  }
  return Cell{
      .x = static_cast<std::int32_t>(std::floor(x)),
      .y = static_cast<std::int32_t>(std::floor(y)),
  };
}

[[nodiscard]] std::int32_t QuantizeYaw(
    double yaw_rad, std::size_t bin_count) noexcept {
  const double step =
      2.0 * std::numbers::pi /
      static_cast<double>(bin_count);
  const auto raw =
      static_cast<std::int64_t>(std::llround(yaw_rad / step));
  const auto count = static_cast<std::int64_t>(bin_count);
  return static_cast<std::int32_t>(
      ((raw % count) + count) % count);
}

[[nodiscard]] double UnwrapNear(double yaw,
                                double previous) noexcept {
  const double period = 2.0 * std::numbers::pi;
  while (yaw - previous > std::numbers::pi) {
    yaw -= period;
  }
  while (yaw - previous < -std::numbers::pi) {
    yaw += period;
  }
  return yaw;
}

[[nodiscard]] PoseXyzYaw StatePose(
    const LeggedDiscretePlan& plan, std::size_t index,
    double preferred_height,
    std::optional<double> previous_yaw) noexcept {
  const auto& state = plan.states[index];
  double yaw =
      static_cast<double>(state.iyaw) *
      (2.0 * std::numbers::pi /
       static_cast<double>(plan.grid.yaw_bin_count));
  if (previous_yaw.has_value()) {
    yaw = UnwrapNear(yaw, *previous_yaw);
  }
  return PoseXyzYaw{
      .position_m =
          {
              plan.grid_origin_m.x +
                  (static_cast<double>(state.ix) + 0.5) *
                      plan.grid.xy_resolution_m,
              plan.grid_origin_m.y +
                  (static_cast<double>(state.iy) + 0.5) *
                      plan.grid.xy_resolution_m,
              std::clamp(preferred_height,
                         state.reachable_z.min_m,
                         state.reachable_z.max_m),
          },
      .yaw_rad = yaw,
  };
}

[[nodiscard]] PrimitiveKind ToPrimitiveKind(
    BodyMotionKind kind) noexcept {
  switch (kind) {
    case BodyMotionKind::kSpin:
      return PrimitiveKind::kBodySpin;
    case BodyMotionKind::kCoupledTranslationYaw:
      return PrimitiveKind::kBodyCoupled;
    default:
      return PrimitiveKind::kBodyTranslation;
  }
}

[[nodiscard]] Result<ValidatedPrimitiveChain>
BuildPrimitiveChain(
    const LeggedDiscretePlan& plan,
    const LeggedCapabilityView& capability) {
  if (plan.edges.empty() ||
      plan.states.size() != plan.edges.size() + 1U ||
      plan.grid.yaw_bin_count == 0U) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/legged_discrete_plan",
                   "LEGGED_PRIMITIVE_CHAIN_IS_EMPTY");
  }
  std::vector<PoseXyzYaw> poses;
  poses.reserve(plan.states.size());
  for (std::size_t index = 0U; index < plan.states.size();
       ++index) {
    poses.push_back(
        StatePose(
            plan, index, capability.preferred_body_height_m,
            poses.empty()
                ? std::optional<double>{}
                : std::optional<double>{poses.back().yaw_rad}));
  }

  ValidatedPrimitiveChain chain;
  chain.primitives.reserve(plan.edges.size());
  for (std::size_t index = 0U; index < plan.edges.size();
       ++index) {
    const auto& edge = plan.edges[index];
    if (edge.transition_time.value.count() <= 0 ||
        !ValidContentRef(edge.sampled_body_sweep_ref)) {
      return Failure(ErrorCode::kInvalidArgument,
                     "/legged_discrete_plan/edges",
                     "INVALID_CERTIFIED_LEGGED_EDGE");
    }
    chain.primitives.push_back(
        ValidatedPrimitive{
            .primitive_id =
                "legged/" + std::to_string(index) + "/" +
                edge.primitive_id,
            .capability_primitive_id = edge.primitive_id,
            .primitive_kind =
                ToPrimitiveKind(edge.motion_kind),
            .start_pose = poses[index],
            .end_pose = poses[index + 1U],
            .nominal_duration = edge.transition_time,
            .validation_ref = edge.sampled_body_sweep_ref,
        });
  }
  return chain;
}

[[nodiscard]] std::vector<LeggedFrozenControlPoint> FrozenEndpoints(
    const LeggedDiscretePlan& plan,
    const LeggedCapabilityView& capability) {
  const std::size_t control_count =
      std::max<std::size_t>(4U, plan.states.size());
  const PoseXyzYaw first = StatePose(
      plan, 0U, capability.preferred_body_height_m,
      std::nullopt);
  PoseXyzYaw last = first;
  for (std::size_t index = 1U; index < plan.states.size();
       ++index) {
    last = StatePose(
        plan, index, capability.preferred_body_height_m,
        last.yaw_rad);
  }
  return {
      LeggedFrozenControlPoint{
          .control_point_index = 0U,
          .pose = first,
      },
      LeggedFrozenControlPoint{
          .control_point_index = control_count - 1U,
          .pose = last,
      },
  };
}

[[nodiscard]] LeggedTimingConfig TimingConfig(
    const PlannerAlgorithmConfig& algorithm,
    const LeggedCapabilityView& capability) noexcept {
  return LeggedTimingConfig{
      .sampling = algorithm.legged.time_scaling,
      .maximum_linear_acceleration_mps2 =
          capability.maximum_linear_acceleration_mps2,
      .maximum_yaw_acceleration_radps2 =
          capability.maximum_yaw_acceleration_radps2,
  };
}

[[nodiscard]] Result<LeggedTimingResult> TimePrimitiveChain(
    const ValidatedPrimitiveChain& chain,
    const PlannerAlgorithmConfig& algorithm,
    const LeggedCapabilityView& capability) {
  return ParameterizeLeggedBodyPath(
      GeometricPath{chain}, capability.velocity_envelope,
      TimingConfig(algorithm, capability));
}

[[nodiscard]] Result<std::string> StableReferenceId(
    const RequestId& request_id,
    const std::string& selected_candidate_id) {
  if (request_id.empty() || selected_candidate_id.empty()) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/reference_id",
                   "REFERENCE_ID_INPUT_IS_EMPTY");
  }
  const std::vector<std::string> identity_parts{
      request_id, selected_candidate_id, "LEGGED"};
  return "legged/" +
         search_detail::StableEdgeIdHash(identity_parts);
}

[[nodiscard]] bool FinitePose(
    const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool SamePose(
    const PoseXyzYaw& lhs,
    const PoseXyzYaw& rhs) noexcept {
  return std::hypot(
             std::hypot(
                 lhs.position_m.x - rhs.position_m.x,
                 lhs.position_m.y - rhs.position_m.y),
             lhs.position_m.z - rhs.position_m.z) <=
             kGeometryTolerance &&
         std::abs(lhs.yaw_rad - rhs.yaw_rad) <=
             kGeometryTolerance;
}

[[nodiscard]] std::optional<PoseXyzYaw> PathEndpoint(
    const GeometricPath& path) noexcept {
  return std::visit(
      [](const auto& concrete) -> std::optional<PoseXyzYaw> {
        using Path = std::decay_t<decltype(concrete)>;
        if constexpr (
            std::is_same_v<Path, ClampedCubicBSplinePath>) {
          if (concrete.control_points.empty()) {
            return std::nullopt;
          }
          return EvaluateLeggedSpline(concrete, 1.0);
        } else {
          if (concrete.primitives.empty()) {
            return std::nullopt;
          }
          return concrete.primitives.back().end_pose;
        }
      },
      path);
}

[[nodiscard]] Result<LeggedBodyReference> AssembleReference(
    const PlanningRequest& request,
    const LeggedDiscretePlan& discrete,
    const LeggedCapabilityView& capability,
    GeometricPath path,
    LeggedTimingResult timing) {
  const auto endpoint = PathEndpoint(path);
  const auto stable_id =
      StableReferenceId(request.request_id,
                        discrete.stable_candidate_id);
  if (!endpoint.has_value() || !FinitePose(*endpoint) ||
      !IsOk(stable_id) ||
      timing.time_scaling.segments.empty() ||
      timing.duration.value.count() <= 0 ||
      !timing.diagnostics.ends_stopped) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/legged_reference",
                   "INCOMPLETE_LEGGED_REFERENCE_TRANSACTION");
  }
  const auto& map_ref = request.map_snapshot->snapshot_ref();
  LeggedBodyReference reference{
      .reference_id = std::get<std::string>(stable_id),
      .reference_hash = std::string(64U, '0'),
      .reference_point_id = capability.reference_point_id,
      .reference_time_origin = request.request_time,
      .geometric_path = std::move(path),
      .time_scaling = std::move(timing.time_scaling),
      .body_frame_velocity_envelope =
          capability.velocity_envelope,
      .terrain_normal_envelope =
          {
              .maximum_normal_deviation_rad =
                  capability.terrain_thresholds.maximum_slope_rad,
              .source_terrain_certification_ref = map_ref,
          },
      .roll_pitch_diagnostic_envelope =
          {
              .roll_rad =
                  {
                      -capability.terrain_thresholds
                           .maximum_slope_rad,
                      capability.terrain_thresholds
                          .maximum_slope_rad,
                  },
              .pitch_rad =
                  {
                      -capability.terrain_thresholds
                           .maximum_slope_rad,
                      capability.terrain_thresholds
                          .maximum_slope_rad,
                  },
          },
      .safe_stop_anchor =
          {
              .anchor_id =
                  "legged-safe-stop/" +
                  discrete.stable_candidate_id,
              .pose = *endpoint,
              .target_linear_velocity_mps = 0.0,
              .target_yaw_rate_radps = 0.0,
              .terrain_certification_ref = map_ref,
          },
      .feasibility_scope =
          "body_geometry_and_terrain_thresholds_only",
      .footstep_feasibility_guaranteed = false,
  };
  if (!SamePose(reference.safe_stop_anchor.pose, *endpoint)) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/legged_reference/safe_stop_anchor",
                   "LEGGED_SAFE_STOP_ENDPOINT_MISMATCH");
  }
  const auto hash =
      CanonicalReferenceHash(PlatformReference{reference});
  if (!IsOk(hash)) {
    return std::get<Error>(hash);
  }
  reference.reference_hash =
      std::get<Sha256Digest>(hash);
  const auto validation =
      SemanticValidator{}.Validate(
          PlatformReference{reference});
  if (!validation.ok()) {
    return Failure(
        ErrorCode::kInvalidArgument,
        validation.issues.front().field_path,
        "FINAL_LEGGED_REFERENCE_SEMANTIC_VALIDATION_FAILED:" +
            validation.issues.front().reason_code);
  }
  return reference;
}

[[nodiscard]] Result<LeggedBodyReference> PlanImpl(
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminal_set,
    const LeggedPlannerDependencies& dependencies) {
  const auto request_validation =
      ValidateRequest(request, terminal_set);
  if (!IsOk(request_validation)) {
    return std::get<Error>(request_validation);
  }
  const auto capability_result =
      LeggedCapabilityView::Create(*request.safety_capability);
  if (!IsOk(capability_result)) {
    return std::get<Error>(capability_result);
  }
  const auto& capability =
      std::get<LeggedCapabilityView>(capability_result);
  const auto catalog_result =
      BodyMotionPrimitiveCatalog::Create(
          *request.safety_capability);
  if (!IsOk(catalog_result)) {
    return std::get<Error>(catalog_result);
  }
  const auto& catalog =
      std::get<BodyMotionPrimitiveCatalog>(catalog_result);

  std::shared_ptr<const LearnedCostSnapshot> learned;
  if (request.learned_cost_snapshot.has_value()) {
    learned =
        request.learned_cost_snapshot->resolved_snapshot;
  }
  const auto projection_result =
      BuildSafeProjection(
          SafeProjectionRequest{
              .map = request.map_snapshot,
              .capability = request.safety_capability,
              .algorithm_config = request.algorithm_config,
              .learned_cost = std::move(learned),
          });
  if (!IsOk(projection_result)) {
    return std::get<Error>(projection_result);
  }
  const auto& projection =
      std::get<SafeProjection>(projection_result);
  const auto& state =
      std::get<WheeledOrLeggedState>(request.current_state);
  const auto start_cell =
      PositionCell(projection.geometry, state.position_m);
  if (!start_cell.has_value()) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/current_state/position_m",
                   "LEGGED_START_OUTSIDE_MAP");
  }
  const auto terrain =
      LeggedTerrainEvaluator(projection, capability)
          .EvaluatePose(
              PoseXyzYaw{
                  .position_m = state.position_m,
                  .yaw_rad = state.yaw_rad,
              },
              capability.collision_envelope);
  if (!terrain.hard_feasible ||
      !IsValidHeightInterval(terrain.body_height_interval) ||
      state.position_m.z <
          terrain.body_height_interval.min_m -
              kGeometryTolerance ||
      state.position_m.z >
          terrain.body_height_interval.max_m +
              kGeometryTolerance) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/current_state/position_m/z",
                   "LEGGED_START_HEIGHT_NOT_HARD_FEASIBLE");
  }

  const auto& algorithm = *request.algorithm_config;
  const LeggedLatticeState start{
      .ix = start_cell->x,
      .iy = start_cell->y,
      .iyaw =
          QuantizeYaw(
              state.yaw_rad,
              algorithm.legged.pose_lattice.yaw_bin_count),
      .reachable_z = terrain.body_height_interval,
  };
  const auto discrete_result =
      PlanLeggedDiscrete(
          LeggedPlanningProblem{
              .projection = projection,
              .start = start,
              .terminals = terminal_set.candidates,
              .grid = algorithm.legged.pose_lattice,
              .time_equivalence_tolerance =
                  algorithm.time_equivalence_tolerance,
          },
          catalog, algorithm.ara_star);
  if (!IsOk(discrete_result)) {
    return std::get<Error>(discrete_result);
  }
  const auto& discrete =
      std::get<LeggedDiscretePlan>(discrete_result);
  const auto primitive_result =
      BuildPrimitiveChain(discrete, capability);
  if (!IsOk(primitive_result)) {
    return std::get<Error>(primitive_result);
  }
  const auto& primitive_chain =
      std::get<ValidatedPrimitiveChain>(primitive_result);

  GeometricPath selected_path{primitive_chain};
  Result<LeggedTimingResult> selected_timing =
      TimePrimitiveChain(
          primitive_chain, algorithm, capability);
  if (!IsOk(selected_timing)) {
    return std::get<Error>(selected_timing);
  }

  const auto corridor_result =
      BuildLeggedCorridor(
          LeggedCorridorRequest{
              .projection = projection,
              .discrete_plan = discrete,
              .capability = capability,
              .config = algorithm.legged.corridor,
              .product_certifier =
                  dependencies.product_certifier,
          });
  if (IsOk(corridor_result) &&
      dependencies.qp_solver != nullptr) {
    const auto frozen =
        FrozenEndpoints(discrete, capability);
    const auto optimized =
        OptimizeLeggedBodySpline(
            LeggedSplineRequest{
                .discrete_plan = discrete,
                .corridor =
                    std::get<LeggedCorridor>(
                        corridor_result),
                .config =
                    LeggedSplineConfig{
                        .smoothing =
                            algorithm.legged.smoothing,
                        .qp_settings =
                            dependencies.qp_settings,
                        .validation_sample_count =
                            std::max<std::size_t>(
                                2U,
                                algorithm.legged
                                    .continuous_validation_maximum_subdivisions),
                    },
                .preferred_body_height_m =
                    capability.preferred_body_height_m,
                .committed_points = frozen,
            },
            *dependencies.qp_solver);
    if (optimized.body_spline.has_value()) {
      const GeometricPath spline_path{
          *optimized.body_spline};
      auto spline_timing =
          ParameterizeLeggedBodyPath(
              spline_path, capability.velocity_envelope,
              TimingConfig(algorithm, capability));
      if (IsOk(spline_timing)) {
        selected_path = spline_path;
        selected_timing = std::move(spline_timing);
      }
    }
  }

  return AssembleReference(
      request, discrete, capability,
      std::move(selected_path),
      std::get<LeggedTimingResult>(
          std::move(selected_timing)));
}

}  // namespace

LeggedPlanner::LeggedPlanner(
    LeggedPlannerDependencies dependencies) noexcept
    : dependencies_(dependencies) {}

Result<LeggedBodyReference> LeggedPlanner::Plan(
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminal_set) const noexcept {
  try {
    return PlanImpl(request, terminal_set, dependencies_);
  } catch (const std::bad_alloc&) {
    return Failure(ErrorCode::kResourceLimit,
                   "/legged_planner",
                   "LEGGED_PLANNER_ALLOCATION_LIMIT");
  } catch (const std::exception&) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/legged_planner",
                   "LEGGED_PLANNER_INTERNAL_EXCEPTION");
  } catch (...) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/legged_planner",
                   "LEGGED_PLANNER_UNKNOWN_EXCEPTION");
  }
}

}  // namespace lunar::planning::v3
