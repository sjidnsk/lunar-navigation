#include "lunar_path_planner/v3/hopper/hopper_config.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>

#include <Eigen/LU>

#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error Invalid(
    std::string path, std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool FiniteNonnegative(const double value) {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool FinitePositive(const double value) {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool ValidBox(const AxisAlignedBox3& box) {
  return std::isfinite(box.center.x) &&
         std::isfinite(box.center.y) &&
         std::isfinite(box.center.z) &&
         FiniteNonnegative(box.half_extent.x) &&
         FiniteNonnegative(box.half_extent.y) &&
         FiniteNonnegative(box.half_extent.z);
}

[[nodiscard]] bool ValidRef(const ContentRef& ref) {
  return !ref.id.empty() && ref.revision > 0U &&
         ref.content_hash.size() == 64U &&
         std::ranges::all_of(
             ref.content_hash, [](const char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
             });
}

template <class T>
[[nodiscard]] bool BindingMatchesObject(
    const ResolvedBinding<T>& binding) {
  return binding.object != nullptr &&
         binding.content_ref == binding.object->content_ref;
}

[[nodiscard]] bool PointSatisfiesHalfspaces(
    const Eigen::Vector2d& point,
    const std::vector<std::pair<Eigen::Vector2d, double>>& halfspaces) {
  return std::ranges::all_of(
      halfspaces, [&point](const auto& halfspace) {
        return halfspace.first.dot(point) <=
               halfspace.second + 1.0e-9;
      });
}

[[nodiscard]] Result<ConvexPolygon2d> ExtractLandingFootprint(
    const BodyConvexPolytope& envelope) {
  std::vector<std::pair<Eigen::Vector2d, double>> halfspaces;
  for (const Halfspace3& halfspace :
       envelope.body_frame_halfspaces.halfspaces) {
    const Eigen::Vector2d normal{
        halfspace.normal.x, halfspace.normal.y};
    const double norm = normal.norm();
    if (std::abs(halfspace.normal.z) <= 1.0e-9 &&
        norm > 1.0e-12 && std::isfinite(halfspace.offset_m)) {
      halfspaces.emplace_back(
          normal / norm, halfspace.offset_m / norm);
    }
  }
  if (halfspaces.size() < 3U) {
    return Invalid(
        "content.collision_envelope",
        "HOPPER_LANDING_FOOTPRINT_NOT_RESOLVABLE");
  }

  std::vector<Eigen::Vector2d> intersections;
  for (std::size_t first = 0U; first < halfspaces.size(); ++first) {
    for (std::size_t second = first + 1U;
         second < halfspaces.size(); ++second) {
      Eigen::Matrix2d matrix;
      matrix.row(0) = halfspaces[first].first.transpose();
      matrix.row(1) = halfspaces[second].first.transpose();
      const double determinant = matrix.determinant();
      if (std::abs(determinant) <= 1.0e-12) {
        continue;
      }
      const Eigen::Vector2d point =
          matrix.inverse() *
          Eigen::Vector2d{
              halfspaces[first].second,
              halfspaces[second].second};
      if (point.allFinite() &&
          PointSatisfiesHalfspaces(point, halfspaces)) {
        const bool duplicate = std::ranges::any_of(
            intersections, [&point](const Eigen::Vector2d& other) {
              return (point - other).norm() <= 1.0e-9;
            });
        if (!duplicate) {
          intersections.push_back(point);
        }
      }
    }
  }
  if (intersections.size() < 3U) {
    return Invalid(
        "content.collision_envelope",
        "HOPPER_LANDING_FOOTPRINT_EMPTY_OR_UNBOUNDED");
  }

  const Eigen::Vector2d center =
      std::accumulate(
          intersections.begin(), intersections.end(),
          Eigen::Vector2d::Zero().eval()) /
      static_cast<double>(intersections.size());
  std::sort(
      intersections.begin(), intersections.end(),
      [&center](const Eigen::Vector2d& lhs,
                const Eigen::Vector2d& rhs) {
        return std::tuple{
                   std::atan2(lhs.y() - center.y(),
                              lhs.x() - center.x()),
                   lhs.x(), lhs.y()} <
               std::tuple{
                   std::atan2(rhs.y() - center.y(),
                              rhs.x() - center.x()),
                   rhs.x(), rhs.y()};
      });
  ConvexPolygon2d polygon{std::move(intersections)};
  if (!validate_convex_polygon(polygon).ok()) {
    return Invalid(
        "content.collision_envelope",
        "HOPPER_LANDING_FOOTPRINT_NOT_STRICTLY_CONVEX");
  }
  return polygon;
}

[[nodiscard]] bool ValidGravity(const GravityModel& model) {
  return ValidRef(model.content_ref) && !model.frame_id.empty() &&
         model.nominal_acceleration_mps2.allFinite() &&
         model.nominal_acceleration_mps2.norm() > 0.0 &&
         ValidBox(model.acceleration_error_mps2) &&
         ValidBox(model.spatial_validity_m) &&
         !model.valid_from.clock_id.empty() &&
         model.valid_from.clock_id == model.valid_until.clock_id &&
         model.valid_until.tick >= model.valid_from.tick;
}

[[nodiscard]] bool ValidErrorModel(
    const DeterministicErrorModel& model) {
  return ValidRef(model.content_ref) &&
         ValidBox(model.initial_position_error_m) &&
         ValidBox(model.initial_velocity_error_mps) &&
         ValidBox(model.launch_execution_velocity_error_mps) &&
         ValidBox(model.gravity_error_mps2) &&
         ValidBox(model.landing_plane_origin_error_m) &&
         FiniteNonnegative(model.landing_plane_normal_error.radius_rad) &&
         model.landing_plane_normal_error.radius_rad <=
             std::numbers::pi &&
         std::isfinite(
             model.landing_plane_residual_error_m.center) &&
         FiniteNonnegative(
             model.landing_plane_residual_error_m.half_width);
}

[[nodiscard]] bool ValidBodyEnvelope(
    const BodyRotationEnvelope& envelope) {
  if (!ValidRef(envelope.content_ref) ||
      envelope.arbitrary_attitude_body_envelope.halfspaces.size() <
          4U) {
    return false;
  }
  return std::ranges::all_of(
      envelope.arbitrary_attitude_body_envelope.halfspaces,
      [](const Halfspace3& halfspace) {
        const double norm =
            std::hypot(
                halfspace.normal.x, halfspace.normal.y,
                halfspace.normal.z);
        return std::isfinite(halfspace.offset_m) &&
               std::isfinite(norm) &&
               std::abs(norm - 1.0) <= 1.0e-9;
      });
}

[[nodiscard]] bool ValidTighteningTable(
    const AttitudeTighteningTable& table,
    const ArbitraryAxisAttitudeEnvelope& base) {
  if (!ValidRef(table.content_ref)) {
    return false;
  }
  double previous_angle = -1.0;
  double previous_speed = base.maximum_angular_speed_radps;
  double previous_acceleration =
      base.maximum_angular_acceleration_radps2;
  for (const auto& entry : table.entries) {
    if (!FiniteNonnegative(entry.maximum_rotation_angle_rad) ||
        entry.maximum_rotation_angle_rad > std::numbers::pi ||
        !(entry.maximum_rotation_angle_rad > previous_angle) ||
        !FinitePositive(entry.maximum_angular_speed_radps) ||
        !FinitePositive(
            entry.maximum_angular_acceleration_radps2) ||
        entry.maximum_angular_speed_radps >
            base.maximum_angular_speed_radps ||
        entry.maximum_angular_acceleration_radps2 >
            base.maximum_angular_acceleration_radps2 ||
        entry.maximum_angular_speed_radps > previous_speed ||
        entry.maximum_angular_acceleration_radps2 >
            previous_acceleration) {
      return false;
    }
    previous_angle = entry.maximum_rotation_angle_rad;
    previous_speed = entry.maximum_angular_speed_radps;
    previous_acceleration =
        entry.maximum_angular_acceleration_radps2;
  }
  return true;
}

}  // namespace

Result<HopperCapabilityView> bind_hopper_capability(
    const SafetyCapabilityProfile& profile,
    const ResolvedCapabilityBindings& bindings) {
  if (!SemanticValidator{}.Validate(profile).ok()) {
    return Invalid("safety_capability", "INVALID_HOPPER_CAPABILITY");
  }
  const auto* hopper =
      std::get_if<HopperCapability>(&profile.content);
  if (hopper == nullptr) {
    return Invalid(
        "safety_capability.content",
        "HOPPER_CAPABILITY_REQUIRED");
  }
  if (bindings.motion_model.content_ref !=
          hopper->motion_model_ref ||
      bindings.analytic_cost_model.content_ref !=
          hopper->analytic_cost_model_ref) {
    return Invalid(
        "capability_bindings",
        "HOPPER_BASE_MODEL_BINDING_MISMATCH");
  }
  if (!bindings.gravity_model.has_value() ||
      !BindingMatchesObject(*bindings.gravity_model) ||
      bindings.gravity_model->content_ref !=
          hopper->gravity_model_ref ||
      !ValidGravity(*bindings.gravity_model->object)) {
    return Invalid(
        "capability_bindings.gravity_model",
        "INVALID_HOPPER_GRAVITY_BINDING");
  }
  if (!bindings.error_model.has_value() ||
      !BindingMatchesObject(*bindings.error_model) ||
      !ValidErrorModel(*bindings.error_model->object)) {
    return Invalid(
        "capability_bindings.error_model",
        "INVALID_HOPPER_ERROR_MODEL_BINDING");
  }
  if (!bindings.actuator_or_impulse_profile.has_value() ||
      !BindingMatchesObject(
          *bindings.actuator_or_impulse_profile)) {
    return Invalid(
        "capability_bindings.actuator_or_impulse_profile",
        "INVALID_HOPPER_ACTUATOR_BINDING");
  }
  const auto& actuator =
      *bindings.actuator_or_impulse_profile->object;
  if (!ValidRef(actuator.content_ref) ||
      !FinitePositive(actuator.platform_mass_kg) ||
      actuator.launch_preparation_time.value.count() < 0 ||
      actuator.landing_settle_time.value.count() < 0 ||
      !FiniteNonnegative(
          actuator.nominal_landing_center_normal_offset_m)) {
    return Invalid(
        "capability_bindings.actuator_or_impulse_profile",
        "INVALID_HOPPER_ACTUATOR_CONTENT");
  }
  if (!bindings.body_rotation_envelope.has_value() ||
      !BindingMatchesObject(
          *bindings.body_rotation_envelope) ||
      !ValidBodyEnvelope(
          *bindings.body_rotation_envelope->object)) {
    return Invalid(
        "capability_bindings.body_rotation_envelope",
        "INVALID_HOPPER_BODY_ROTATION_BINDING");
  }

  std::optional<AttitudeTighteningTable> table;
  if (hopper->attitude_tightening_table_ref.has_value()) {
    if (!bindings.attitude_tightening_table.has_value() ||
        !BindingMatchesObject(
            *bindings.attitude_tightening_table) ||
        bindings.attitude_tightening_table->content_ref !=
            *hopper->attitude_tightening_table_ref ||
        !ValidTighteningTable(
            *bindings.attitude_tightening_table->object,
            hopper->attitude_envelope)) {
      return Invalid(
          "capability_bindings.attitude_tightening_table",
          "INVALID_TIGHTEN_ONLY_ATTITUDE_TABLE");
    }
    table = *bindings.attitude_tightening_table->object;
  } else if (bindings.attitude_tightening_table.has_value()) {
    return Invalid(
        "capability_bindings.attitude_tightening_table",
        "UNDECLARED_ATTITUDE_TIGHTENING_TABLE");
  }

  const auto footprint =
      ExtractLandingFootprint(hopper->collision_envelope);
  if (!IsOk(footprint)) {
    return std::get<Error>(footprint);
  }
  return HopperCapabilityView{
      .source_safety_capability_ref = profile.content_ref,
      .frame_id = hopper->frame_id,
      .collision_envelope = hopper->collision_envelope,
      .landing_footprint_body_xy =
          std::get<ConvexPolygon2d>(footprint),
      .landing_terrain_thresholds =
          hopper->landing_terrain_thresholds,
      .launch_limits = hopper->launch_limits,
      .attitude_envelope = hopper->attitude_envelope,
      .certified_state_error_bounds =
          hopper->certified_state_error_bounds,
      .gravity_model = *bindings.gravity_model->object,
      .error_model = *bindings.error_model->object,
      .actuator_or_impulse_profile = actuator,
      .body_rotation_envelope =
          *bindings.body_rotation_envelope->object,
      .attitude_tightening_table = std::move(table),
  };
}

Result<HopperPlannerLimits> bind_hopper_limits(
    const PlannerAlgorithmConfig& config) {
  const HopperAlgorithmConfig& hopper = config.hopper;
  if (hopper.yaw_partition_count < 1U ||
      hopper.support_direction_count < 4U ||
      hopper.maximum_landing_regions < 1U ||
      hopper.landing_region_maximum_vertices < 3U ||
      hopper.landing_region_inflation_iterations < 1U ||
      hopper.maximum_graph_nodes < 1U ||
      hopper.maximum_graph_out_degree < 1U ||
      hopper.maximum_nominal_aim_points_per_region < 1U ||
      hopper.maximum_full_certification_attempts < 1U ||
      hopper.maximum_root_iterations < 1U ||
      hopper.maximum_flight_tube_sections < 1U) {
    return Invalid(
        "algorithm_config.hopper",
        "INVALID_HOPPER_RESOURCE_CAP");
  }
  return HopperPlannerLimits{
      .yaw_partition_count = hopper.yaw_partition_count,
      .support_direction_count = hopper.support_direction_count,
      .maximum_landing_regions =
          hopper.maximum_landing_regions,
      .landing_region_maximum_vertices =
          hopper.landing_region_maximum_vertices,
      .landing_region_inflation_iterations =
          hopper.landing_region_inflation_iterations,
      .landing_region_maximum_split_depth =
          hopper.landing_region_maximum_split_depth,
      .maximum_graph_nodes = hopper.maximum_graph_nodes,
      .maximum_graph_out_degree =
          hopper.maximum_graph_out_degree,
      .maximum_nominal_aim_points_per_region =
          hopper.maximum_nominal_aim_points_per_region,
      .maximum_full_certification_attempts =
          hopper.maximum_full_certification_attempts,
      .maximum_interval_subdivision_depth =
          hopper.maximum_interval_subdivision_depth,
      .maximum_root_iterations = hopper.maximum_root_iterations,
      .maximum_collision_subdivision_depth =
          hopper.maximum_collision_subdivision_depth,
      .maximum_flight_tube_sections =
          hopper.maximum_flight_tube_sections,
      .time_equivalence_tolerance =
          config.time_equivalence_tolerance,
      .ara_star = config.ara_star,
  };
}

}  // namespace lunar::planning::v3
