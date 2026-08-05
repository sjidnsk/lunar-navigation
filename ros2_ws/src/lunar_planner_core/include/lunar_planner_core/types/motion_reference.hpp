#pragma once

#include <chrono>
#include <string>
#include <variant>
#include <vector>

#include "lunar_planner_core/types/geometry.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"

namespace lunar::planning {

enum class TrajectorySemantics : std::uint8_t {
  kWheeledBase,
  kLeggedBodyReference,
};

struct TrajectoryPoint final {
  std::chrono::nanoseconds time_from_start{};
  Pose3 pose;
  Twist3 velocity;
};

struct TrajectoryReference final {
  TrajectorySemantics semantics{TrajectorySemantics::kWheeledBase};
  std::vector<TrajectoryPoint> points;
};

struct HopSegment final {
  std::string segment_id;
  Pose3 launch_pose;
  std::vector<Vec3> landing_region_boundary_m;
  std::chrono::nanoseconds flight_time{};
  Vec3 launch_velocity_mps;
  double flight_tube_radius_m{};
};

struct HopReference final {
  std::vector<HopSegment> segments;
};

using MotionReferenceData = std::variant<TrajectoryReference, HopReference>;

struct GlobalRoutePreview final {
  std::vector<Pose3> poses_map;
};

struct MotionReference final {
  std::string plan_id;
  PlatformType platform_type{PlatformType::kWheeled};
  TimePoint input_time;
  GlobalRoutePreview preview;
  MotionReferenceData data;
};

} // namespace lunar::planning
