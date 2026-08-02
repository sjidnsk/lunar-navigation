#pragma once

#include <variant>

#include "lunar_path_planner/v3/contracts/base_types.hpp"

namespace lunar::planning::v3 {

struct AxisAlignedBox3 final {
  Vec3 center;
  Vec3 half_extent;
};

struct EuclideanBall3 final {
  Vec3 center;
  double radius{};
};

using DeterministicVectorSet3 =
    std::variant<AxisAlignedBox3, EuclideanBall3>;

struct SymmetricScalarInterval final {
  double center{};
  double half_width{};
};

struct RotationVectorBall final {
  double radius_rad{};
};

struct WheeledOrLeggedErrorBounds final {
  DeterministicVectorSet3 position_bound_m;
  SymmetricScalarInterval yaw_bound_rad;
  DeterministicVectorSet3 linear_velocity_bound_mps;
  SymmetricScalarInterval yaw_rate_bound_radps;
};

struct WheeledOrLeggedState final {
  Vec3 position_m;
  double yaw_rad{};
  Vec3 linear_velocity_mps;
  double yaw_rate_radps{};
  WheeledOrLeggedErrorBounds error_bounds;
};

struct HopperErrorBounds final {
  DeterministicVectorSet3 position_bound_m;
  RotationVectorBall orientation_bound;
  DeterministicVectorSet3 linear_velocity_bound_mps;
  DeterministicVectorSet3 angular_velocity_bound_radps;
};

struct HopperState final {
  Vec3 position_m;
  Quaternion orientation_body_to_frame;
  Vec3 linear_velocity_mps;
  Vec3 angular_velocity_radps;
  HopperErrorBounds error_bounds;
};

enum class PlatformType {
  kWheeled,
  kLegged,
  kHopper,
};

using PlatformState = std::variant<WheeledOrLeggedState, HopperState>;

}  // namespace lunar::planning::v3
