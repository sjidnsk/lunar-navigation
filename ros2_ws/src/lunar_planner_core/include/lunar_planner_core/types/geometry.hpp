#pragma once

#include <compare>
#include <cstdint>
#include <vector>

namespace lunar::planning {

struct TimePoint final {
  std::int64_t nanoseconds_since_epoch{};

  auto operator<=>(const TimePoint&) const = default;
};

struct Vec2 final {
  double x{};
  double y{};

  auto operator<=>(const Vec2&) const = default;
};

struct Vec3 final {
  double x{};
  double y{};
  double z{};

  auto operator<=>(const Vec3&) const = default;
};

struct Quaternion final {
  double w{1.0};
  double x{};
  double y{};
  double z{};

  auto operator<=>(const Quaternion&) const = default;
};

struct Pose3 final {
  Vec3 position_m;
  Quaternion orientation;

  auto operator<=>(const Pose3&) const = default;
};

struct Twist3 final {
  Vec3 linear_mps;
  Vec3 angular_radps;

  auto operator<=>(const Twist3&) const = default;
};

struct Interval final {
  double lower{};
  double upper{};

  auto operator<=>(const Interval&) const = default;
};

struct Polygon2 final {
  std::vector<Vec2> vertices;
};

}  // namespace lunar::planning
