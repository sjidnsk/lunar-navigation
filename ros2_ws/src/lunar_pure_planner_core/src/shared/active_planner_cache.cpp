#include "shared/active_planner_cache.hpp"

#include <array>
#include <bit>
#include <string_view>
#include <type_traits>
#include <variant>

namespace lunar::pure_planning::shared {
namespace {

constexpr std::uint64_t kFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;
constexpr std::uint64_t kLeggedTraversabilityModeIdentity =
    0x4c45474745445631ULL;

void HashByte(std::uint64_t& hash, const std::uint8_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

template <typename Value>
void HashPod(std::uint64_t& hash, const Value value) noexcept {
  static_assert(std::is_trivially_copyable_v<Value>);
  const auto bytes =
      std::bit_cast<std::array<std::uint8_t, sizeof(Value)>>(value);
  for (const std::uint8_t byte : bytes) {
    HashByte(hash, byte);
  }
}

void HashString(std::uint64_t& hash, const std::string_view value) noexcept {
  HashPod(hash, value.size());
  for (const unsigned char byte : value) {
    HashByte(hash, byte);
  }
}

void HashVec2(std::uint64_t& hash, const Vec2 value) noexcept {
  HashPod(hash, value.x);
  HashPod(hash, value.y);
}

void HashVec3(std::uint64_t& hash, const Vec3 value) noexcept {
  HashPod(hash, value.x);
  HashPod(hash, value.y);
  HashPod(hash, value.z);
}

void HashPose(std::uint64_t& hash, const Pose3& pose) noexcept {
  HashVec3(hash, pose.position_m);
  HashPod(hash, pose.orientation.w);
  HashPod(hash, pose.orientation.x);
  HashPod(hash, pose.orientation.y);
  HashPod(hash, pose.orientation.z);
}

[[nodiscard]] std::uint64_t GoalFingerprint(
    const GoalRegion& goal) noexcept {
  std::uint64_t hash = kFnvOffset;
  HashString(hash, goal.goal_id);
  HashPod(hash, goal.target.index());
  std::visit(
      [&](const auto& target) {
        using Target = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Target, PointGoal>) {
          HashVec3(hash, target.position_m);
          HashPod(hash, target.tolerance_m);
        } else {
          HashPod(hash, target.boundary_m.size());
          for (const Vec3 point : target.boundary_m) {
            HashVec3(hash, point);
          }
          HashPod(hash, target.normal_tolerance_m);
        }
      },
      goal.target);
  HashPod(hash, goal.yaw_rad.has_value());
  if (goal.yaw_rad.has_value()) {
    HashPod(hash, *goal.yaw_rad);
  }
  HashPod(hash, goal.yaw_tolerance_rad);
  return hash;
}

[[nodiscard]] std::uint64_t GoalSetFingerprint(
    const LocalGoalSet& goals) noexcept {
  std::uint64_t hash = kFnvOffset;
  HashPod(hash, goals.exact_final_goal);
  HashPod(hash, goals.goals_odom.size());
  for (const GoalRegion& goal : goals.goals_odom) {
    HashPod(hash, GoalFingerprint(goal));
  }
  return hash;
}

[[nodiscard]] std::uint64_t SearchFingerprint(
    const AnytimeSearchConfig& search) noexcept {
  std::uint64_t hash = kFnvOffset;
  for (const double epsilon : search.epsilon_schedule) {
    HashPod(hash, epsilon);
  }
  HashPod(hash, search.stop_after_first_solution);
  return hash;
}

[[nodiscard]] std::uint64_t DoubleIdentity(const double value) noexcept {
  return std::bit_cast<std::uint64_t>(value);
}

}  // namespace

std::uint64_t StableCapabilityFingerprint(
    const PlatformCapability& capability) noexcept {
  std::uint64_t hash = kFnvOffset;
  HashPod(hash, static_cast<std::uint8_t>(CapabilityPlatform(capability)));
  std::visit(
      [&](const auto& typed) {
        using Capability = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          HashPod(hash, typed.footprint_xy_m.size());
          for (const Vec2 vertex : typed.footprint_xy_m) {
            HashVec2(hash, vertex);
          }
          HashVec3(hash, typed.body_extent_m);
          HashPod(hash, typed.wheel_diameter_m);
          HashPod(hash, typed.wheel_width_m);
          HashPod(hash, typed.wheelbase_m);
          HashPod(hash, typed.track_width_m);
          HashPod(hash, typed.minimum_underbody_clearance_m);
          HashPod(hash, typed.maximum_local_obstacle_relief_m);
          HashPod(hash, typed.allow_unsupported_gap);
          HashPod(hash, typed.minimum_body_z_m);
          HashPod(hash, typed.maximum_body_z_m);
          HashPod(hash, typed.maximum_forward_speed_mps);
          HashPod(hash, typed.maximum_reverse_speed_mps);
          HashPod(hash, typed.maximum_spin_rate_radps);
          HashPod(hash, typed.maximum_acceleration_mps2);
          HashPod(hash, typed.maximum_braking_deceleration_mps2);
          HashPod(hash, typed.maximum_yaw_acceleration_radps2);
          HashPod(hash, typed.maximum_lateral_acceleration_mps2);
          HashPod(hash, typed.maximum_curvature_per_m);
          HashPod(hash, typed.maximum_slope_rad);
          HashPod(hash, typed.minimum_clearance_m);
          HashPod(hash, typed.motion_primitives.size());
          for (const WheelMotionPrimitive& primitive :
               typed.motion_primitives) {
            HashString(hash, primitive.primitive_id);
            HashPod(hash, static_cast<std::uint8_t>(primitive.kind));
            HashPose(hash, primitive.relative_end_pose);
          }
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          HashVec3(hash, typed.body_extent_m);
          HashPod(hash, typed.nominal_body_height_m);
          HashPod(hash, typed.platform_mass_kg);
          HashPod(hash, typed.nominal_payload_kg);
          HashPod(hash, typed.maximum_payload_kg);
          HashPod(hash, typed.maximum_slope_rad);
          HashPod(hash, typed.maximum_step_height_m);
          HashPod(hash, typed.maximum_gap_width_m);
          HashPod(hash, typed.minimum_body_clearance_m);
          HashPod(hash, typed.step_vertical_rate_mps);
          HashPod(hash, typed.body_height_m.lower);
          HashPod(hash, typed.body_height_m.upper);
          HashPod(hash, typed.forward_speed_mps.lower);
          HashPod(hash, typed.forward_speed_mps.upper);
          HashPod(hash, typed.lateral_speed_mps.lower);
          HashPod(hash, typed.lateral_speed_mps.upper);
          HashPod(hash, typed.yaw_rate_radps.lower);
          HashPod(hash, typed.yaw_rate_radps.upper);
          HashPod(hash, typed.maximum_linear_acceleration_mps2);
          HashPod(hash, typed.maximum_yaw_acceleration_radps2);
          HashPod(hash, typed.unknown_is_traversable);
          HashPod(hash, typed.motion_primitives.size());
          for (const LeggedBodyPrimitive& primitive :
               typed.motion_primitives) {
            HashString(hash, primitive.primitive_id);
            HashPod(hash, static_cast<std::uint8_t>(primitive.kind));
            HashVec3(hash, primitive.body_frame_displacement_m);
            HashPod(hash, primitive.yaw_change_rad);
          }
        } else {
          HashPod(hash, typed.specific_impulse_s);
          HashPod(hash, typed.reference_total_mass_kg);
          HashPod(hash, typed.reference_propellant_mass_kg);
          HashVec3(hash, typed.gravity_mps2);
          HashPod(hash, typed.reference_horizontal_range_m);
          HashPod(hash, typed.reference_elevation_delta_m);
          HashPod(hash, typed.runtime_fallback_allowed);
          HashPod(hash, typed.landing_support_radius_m);
          HashPod(hash, typed.flight_collision_radius_m);
          HashPod(hash, typed.maximum_landing_plane_residual_m);
          HashPod(hash, typed.landing_lateral_margin_m);
          HashPod(hash, typed.flight_map_margin_m);
          HashPod(hash, typed.reachability_delta_v_margin_ratio);
          HashPod(hash, typed.standard_gravity_mps2);
          HashPod(hash, typed.maximum_landing_slope_rad);
        }
      },
      capability);
  return hash;
}

GlobalSnapshotCacheKey MakeGlobalSnapshotCacheKey(
    const std::uint64_t global_map_sequence) noexcept {
  return {.source_sequences = {global_map_sequence}};
}

GlobalProjectionCacheKey MakeGlobalProjectionCacheKey(
    const std::uint64_t global_map_sequence,
    const std::int32_t occupancy_threshold, const double inflation_m,
    const std::uint64_t capability_fingerprint) noexcept {
  return {
      .source_sequences = {global_map_sequence},
      .semantic_identities = {
          static_cast<std::uint64_t>(static_cast<std::int64_t>(occupancy_threshold)),
          DoubleIdentity(inflation_m), capability_fingerprint},
  };
}

GlobalProjectionCacheKey MakeLeggedTraversabilityProjectionCacheKey(
    const std::uint64_t traversability_revision,
    const std::uint64_t profile_hash,
    const std::uint64_t capability_fingerprint) noexcept {
  return {
      .source_sequences = {traversability_revision},
      .semantic_identities = {profile_hash, capability_fingerprint,
                              kLeggedTraversabilityModeIdentity},
  };
}

GlobalRouteCacheKey MakeGlobalRouteCacheKey(
    const PlanningRequest& input, const double inflation_m,
    const std::uint64_t capability_fingerprint) noexcept {
  return {
      .source_sequences = {input.world.global_map_sequence,
                           input.world.odometry_sequence,
                           input.world.tf_sequence},
      .semantic_identities = {
          static_cast<std::uint64_t>(static_cast<std::int64_t>(
              input.config.global_occupancy_threshold)),
          DoubleIdentity(inflation_m), capability_fingerprint,
          GoalFingerprint(input.goal_map), SearchFingerprint(input.config.search)},
  };
}

GlobalRouteCacheKey MakeLeggedTraversabilityRouteCacheKey(
    const PlanningRequest& input,
    const std::uint64_t traversability_revision,
    const std::uint64_t profile_hash,
    const std::uint64_t capability_fingerprint) noexcept {
  return {
      .source_sequences = {traversability_revision,
                           input.world.odometry_sequence,
                           input.world.tf_sequence},
      .semantic_identities = {profile_hash, capability_fingerprint,
                              GoalFingerprint(input.goal_map),
                              SearchFingerprint(input.config.search),
                              kLeggedTraversabilityModeIdentity},
  };
}

LocalSnapshotCacheKey MakeLocalSnapshotCacheKey(
    const std::uint64_t local_map_sequence,
    const std::uint64_t start_patch_identity) noexcept {
  return {.source_sequences = {local_map_sequence},
          .semantic_identities = {start_patch_identity}};
}

LocalProjectionCacheKey MakeLocalProjectionCacheKey(
    const std::uint64_t local_map_sequence,
    const double occupancy_threshold,
    const std::uint64_t start_patch_identity) noexcept {
  return {.source_sequences = {local_map_sequence},
          .semantic_identities = {DoubleIdentity(occupancy_threshold),
                                  start_patch_identity}};
}

GoalFieldCacheKey MakeGoalFieldCacheKey(
    const std::uint64_t local_map_sequence,
    const double occupancy_threshold,
    const std::uint64_t capability_fingerprint, const LocalGoalSet& goals,
    const AnytimeSearchConfig& search,
    const std::uint64_t start_patch_identity) noexcept {
  return {
      .source_sequences = {local_map_sequence},
      .semantic_identities = {DoubleIdentity(occupancy_threshold),
                              capability_fingerprint,
                              GoalSetFingerprint(goals),
                              SearchFingerprint(search),
                              start_patch_identity},
  };
}

namespace detail {

std::string ActivePlannerCacheStopReason(
    const SearchControl& control) noexcept {
  try {
    if (control.canceled()) {
      return "REQUEST_CANCELED";
    }
    if (control.expired()) {
      return "TIMEOUT";
    }
  } catch (...) {
    return "PLANNER_ERROR";
  }
  return {};
}

}  // namespace detail
}  // namespace lunar::pure_planning::shared
