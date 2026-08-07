#include "shared/projection_cache.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace lunar::planning::shared {
namespace {

constexpr std::uint64_t kFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;

void HashByte(std::uint64_t& hash, const std::uint8_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

template <typename Value>
void HashPod(std::uint64_t& hash, const Value value) noexcept {
  static_assert(std::is_trivially_copyable_v<Value>);
  const auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(Value)>>(value);
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

[[nodiscard]] std::uint64_t CapabilityFingerprint(
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
          for (const WheelMotionPrimitive& primitive : typed.motion_primitives) {
            HashString(hash, primitive.primitive_id);
            HashPod(hash, static_cast<std::uint8_t>(primitive.kind));
            HashPose(hash, primitive.relative_end_pose);
          }
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          HashVec3(hash, typed.body_extent_m);
          HashPod(hash, typed.platform_mass_kg);
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
          HashPod(hash, typed.motion_primitives.size());
          for (const LeggedBodyPrimitive& primitive : typed.motion_primitives) {
            HashString(hash, primitive.primitive_id);
            HashPod(hash, static_cast<std::uint8_t>(primitive.kind));
            HashVec3(hash, primitive.body_frame_displacement_m);
            HashPod(hash, primitive.yaw_change_rad);
          }
        } else {
          HashPod(hash, typed.specific_impulse_s);
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

[[nodiscard]] std::uint64_t SafetyFingerprint(
    const MapSafetyConfig& safety) noexcept {
  std::uint64_t hash = kFnvOffset;
  HashPod(hash, safety.project_maximum_slope_rad);
  HashPod(hash, safety.maximum_elevation_variance_m2);
  HashPod(hash, safety.maximum_obstacle_variance_m2);
  HashPod(hash, safety.maximum_observation_age_s);
  HashPod(hash, safety.minimum_observation_quality);
  HashPod(hash, safety.minimum_observation_count);
  return hash;
}

[[nodiscard]] std::uint64_t MapContentFingerprint(
    const GridMap& map) noexcept {
  std::uint64_t hash = kFnvOffset;
  HashPod(hash, map.layers.size());
  for (const auto& [name, layer] : map.layers) {
    HashString(hash, name);
    HashPod(hash, layer.values.index());
    std::visit(
        [&](const auto& values) {
          HashPod(hash, values.size());
          for (const auto value : values) {
            HashPod(hash, value);
          }
        },
        layer.values);
  }
  return hash;
}

}  // namespace

ProjectionCacheKey MakeProjectionCacheKey(
    const std::uint64_t map_generation,
    std::string platform_id,
    std::string capability_version,
    const GridMap& map,
    const PlatformCapability& capability,
    const MapSafetyConfig& safety) {
  return ProjectionCacheKey{
      .map_generation = map_generation,
      .platform_type = CapabilityPlatform(capability),
      .platform_id = std::move(platform_id),
      .capability_version = std::move(capability_version),
      .frame_id = map.frame_id,
      .width = map.width,
      .height = map.height,
      .resolution_m = map.resolution_m,
      .origin_x_m = map.origin_m.x,
      .origin_y_m = map.origin_m.y,
      .origin_z_m = map.origin_m.z,
      .map_content_fingerprint = MapContentFingerprint(map),
      .capability_fingerprint = CapabilityFingerprint(capability),
      .safety_fingerprint = SafetyFingerprint(safety),
  };
}

ProjectionContextResult ProjectionCache::GetOrBuild(
    const ProjectionCacheKey& key,
    const GridMap& map,
    const PlatformCapability& capability,
    const MapSafetyConfig& safety,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return {.reason_code = "REQUEST_CANCELED"};
  }
  {
    const std::scoped_lock lock{mutex_};
    if (key_ == key && context_ != nullptr) {
      return {.context = context_, .cache_hit = true};
    }
  }
  const MapSnapshotBuildResult snapshot = MapSnapshot::Create(map);
  if (!snapshot.ok()) {
    return {.reason_code = snapshot.reason_code};
  }
  SafeProjectionBuildResult projection = BuildSafeProjection(
      snapshot.snapshot, capability, safety, stop_token);
  if (!projection.ok()) {
    return {.reason_code = projection.reason_code};
  }
  TerrainLimitsResult limits = ResolveTerrainLimits(capability, safety);
  if (!limits.ok()) {
    return {.reason_code = limits.reason_code};
  }
  if (stop_token.stop_requested()) {
    return {.reason_code = "REQUEST_CANCELED"};
  }
  auto context = std::make_shared<const ProjectionContext>(ProjectionContext{
      .map = snapshot.snapshot,
      .projection = std::make_shared<const SafeProjection>(
          std::move(*projection.projection)),
      .terrain_limits = *limits.limits,
  });
  {
    const std::scoped_lock lock{mutex_};
    key_ = key;
    context_ = context;
  }
  return {.context = std::move(context), .cache_hit = false};
}

}  // namespace lunar::planning::shared
