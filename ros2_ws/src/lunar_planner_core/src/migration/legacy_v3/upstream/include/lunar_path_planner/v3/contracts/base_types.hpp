#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <string>
#include <vector>

namespace lunar::planning::v3 {

using Identifier = std::string;
using Sha256Digest = std::string;
using RequestId = Identifier;
using GoalId = Identifier;
using FrameId = Identifier;
using ClockId = Identifier;
using BundleId = Identifier;
using ReferenceId = Identifier;
using ComponentId = Identifier;
using SegmentId = Identifier;
using PrimitiveId = Identifier;
using ReferencePointId = Identifier;
using LandingRegionId = Identifier;
using JumpBoundaryId = Identifier;
using ReasonCode = std::string;

struct Vec2 final {
  double x{};
  double y{};
};

struct Vec3 final {
  double x{};
  double y{};
  double z{};
};

struct Quaternion final {
  double w{1.0};
  double x{};
  double y{};
  double z{};
};

struct Interval final {
  double lower{};
  double upper{};
};

using ScalarBounds = Interval;

struct DurationNanoseconds final {
  std::chrono::nanoseconds value{};
};

struct ClockStamp final {
  std::string clock_id;
  std::chrono::nanoseconds tick{};
};

struct ContentRef final {
  Identifier id;
  std::uint32_t revision{};
  Sha256Digest content_hash;

  auto operator<=>(const ContentRef&) const = default;
};

struct PoseXyzYaw final {
  Vec3 position_m;
  double yaw_rad{};
};

struct Halfspace3 final {
  Vec3 normal;
  double offset_m{};
};

struct ConvexPolytope3 final {
  enum class Representation {
    kHalfspaceIntersection,
  };

  Representation representation{Representation::kHalfspaceIntersection};
  std::vector<Halfspace3> halfspaces;
};

}  // namespace lunar::planning::v3
