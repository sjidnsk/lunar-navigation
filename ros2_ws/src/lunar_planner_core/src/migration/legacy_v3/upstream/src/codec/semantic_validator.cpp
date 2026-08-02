#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/crypto/sha256.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lunar::planning::v3 {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kGeometryTolerance = 1.0e-9;
constexpr double kPolygonTolerance = 1.0e-12;
constexpr std::uint32_t kMaximumRevision = 2'147'483'647U;

class IssueCollector final {
 public:
  void Add(std::string field_path,
           std::string reason_code,
           std::string message) {
    issues_.push_back(ValidationIssue{
        .field_path = std::move(field_path),
        .reason_code = std::move(reason_code),
        .message = std::move(message),
    });
  }

  void Merge(ValidationReport report, std::string_view prefix = {}) {
    for (auto& issue : report.issues) {
      if (!prefix.empty()) {
        issue.field_path =
            issue.field_path.empty()
                ? std::string(prefix)
                : std::string(prefix) + "." + issue.field_path;
      }
      issues_.push_back(std::move(issue));
    }
  }

  [[nodiscard]] ValidationReport Finish() {
    std::sort(
        issues_.begin(), issues_.end(),
        [](const ValidationIssue& lhs, const ValidationIssue& rhs) {
          return std::tie(lhs.field_path, lhs.reason_code, lhs.message) <
                 std::tie(rhs.field_path, rhs.reason_code, rhs.message);
        });
    issues_.erase(
        std::unique(
            issues_.begin(), issues_.end(),
            [](const ValidationIssue& lhs, const ValidationIssue& rhs) {
              return lhs.field_path == rhs.field_path &&
                     lhs.reason_code == rhs.reason_code &&
                     lhs.message == rhs.message;
            }),
        issues_.end());
    return ValidationReport{.issues = std::move(issues_)};
  }

 private:
  std::vector<ValidationIssue> issues_;
};

[[nodiscard]] std::string ChildPath(std::string_view parent,
                                    std::string_view child) {
  if (parent.empty()) {
    return std::string(child);
  }
  if (child.empty()) {
    return std::string(parent);
  }
  return std::string(parent) + "." + std::string(child);
}

[[nodiscard]] std::string IndexedPath(std::string_view parent,
                                      const std::size_t index) {
  return std::string(parent) + "[" + std::to_string(index) + "]";
}

void CheckFinite(const double value,
                 std::string_view path,
                 IssueCollector& issues) {
  if (!std::isfinite(value)) {
    issues.Add(std::string(path), "non_finite_value",
               "floating-point values must be finite");
  }
}

void CheckNonNegative(const double value,
                      std::string_view path,
                      IssueCollector& issues) {
  CheckFinite(value, path, issues);
  if (std::isfinite(value) && value < 0.0) {
    issues.Add(std::string(path), "non_negative_value_required",
               "value must be non-negative");
  }
}

void CheckPositive(const double value,
                   std::string_view path,
                   IssueCollector& issues) {
  CheckFinite(value, path, issues);
  if (std::isfinite(value) && !(value > 0.0)) {
    issues.Add(std::string(path), "positive_value_required",
               "value must be strictly positive");
  }
}

void CheckFinite(const Vec2& value,
                 std::string_view path,
                 IssueCollector& issues) {
  CheckFinite(value.x, ChildPath(path, "x"), issues);
  CheckFinite(value.y, ChildPath(path, "y"), issues);
}

void CheckFinite(const Vec3& value,
                 std::string_view path,
                 IssueCollector& issues) {
  CheckFinite(value.x, ChildPath(path, "x"), issues);
  CheckFinite(value.y, ChildPath(path, "y"), issues);
  CheckFinite(value.z, ChildPath(path, "z"), issues);
}

void CheckFinite(const Quaternion& value,
                 std::string_view path,
                 IssueCollector& issues) {
  CheckFinite(value.w, ChildPath(path, "w"), issues);
  CheckFinite(value.x, ChildPath(path, "x"), issues);
  CheckFinite(value.y, ChildPath(path, "y"), issues);
  CheckFinite(value.z, ChildPath(path, "z"), issues);
}

[[nodiscard]] double Dot(const Vec3& lhs, const Vec3& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] Vec3 Cross(const Vec3& lhs, const Vec3& rhs) {
  return Vec3{
      .x = lhs.y * rhs.z - lhs.z * rhs.y,
      .y = lhs.z * rhs.x - lhs.x * rhs.z,
      .z = lhs.x * rhs.y - lhs.y * rhs.x,
  };
}

[[nodiscard]] Vec3 Subtract(const Vec3& lhs, const Vec3& rhs) {
  return Vec3{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

[[nodiscard]] Vec3 Add(const Vec3& lhs, const Vec3& rhs) {
  return Vec3{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

[[nodiscard]] Vec3 Scale(const Vec3& value, const double scale) {
  return Vec3{value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] double Norm(const Vec3& value) {
  return std::sqrt(Dot(value, value));
}

[[nodiscard]] double QuaternionNorm(const Quaternion& value) {
  return std::sqrt(value.w * value.w + value.x * value.x +
                   value.y * value.y + value.z * value.z);
}

[[nodiscard]] double QuaternionDot(const Quaternion& lhs,
                                   const Quaternion& rhs) {
  return lhs.w * rhs.w + lhs.x * rhs.x + lhs.y * rhs.y +
         lhs.z * rhs.z;
}

[[nodiscard]] double QuaternionAngularDistance(
    const Quaternion& lhs,
    const Quaternion& rhs) {
  const double lhs_norm = QuaternionNorm(lhs);
  const double rhs_norm = QuaternionNorm(rhs);
  if (!(lhs_norm > 0.0) || !(rhs_norm > 0.0) ||
      !std::isfinite(lhs_norm) || !std::isfinite(rhs_norm)) {
    return std::numeric_limits<double>::infinity();
  }
  const double cosine =
      std::clamp(std::abs(QuaternionDot(lhs, rhs) /
                          (lhs_norm * rhs_norm)),
                 0.0, 1.0);
  return 2.0 * std::acos(cosine);
}

[[nodiscard]] bool IsCanonicalQuaternionSign(
    const Quaternion& value) {
  const std::array<double, 4> components{
      value.w, value.x, value.y, value.z};
  for (const double component : components) {
    if (component > 0.0) {
      return true;
    }
    if (component < 0.0) {
      return false;
    }
    if (std::signbit(component)) {
      return false;
    }
  }
  return false;
}

void CheckQuaternion(const Quaternion& value,
                     std::string_view path,
                     IssueCollector& issues) {
  CheckFinite(value, path, issues);
  const double norm = QuaternionNorm(value);
  if (std::isfinite(norm) &&
      std::abs(norm - 1.0) > kGeometryTolerance) {
    issues.Add(std::string(path), "unit_quaternion_required",
               "quaternion must have unit norm");
  }
  if (std::isfinite(norm) && !IsCanonicalQuaternionSign(value)) {
    issues.Add(std::string(path), "canonical_quaternion_sign_required",
               "quaternion sign must use the canonical first-positive form");
  }
}

[[nodiscard]] bool IsLowercaseSha256(std::string_view value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void CheckHash(std::string_view value,
               std::string_view path,
               IssueCollector& issues) {
  if (!IsLowercaseSha256(value)) {
    issues.Add(std::string(path), "invalid_content_hash",
               "hash must be exactly 64 lower-case hexadecimal characters");
  }
}

void CheckContentRef(const ContentRef& reference,
                     std::string_view path,
                     IssueCollector& issues) {
  if (reference.id.empty()) {
    issues.Add(ChildPath(path, "id"), "identifier_required",
               "content reference ID must not be empty");
  }
  if (reference.revision == 0U ||
      reference.revision > kMaximumRevision) {
    issues.Add(ChildPath(path, "revision"), "revision_out_of_range",
               "revision must be in [1, 2147483647]");
  }
  CheckHash(reference.content_hash, ChildPath(path, "content_hash"),
            issues);
}

void CheckClockStamp(const ClockStamp& stamp,
                     std::string_view path,
                     IssueCollector& issues) {
  if (stamp.clock_id.empty()) {
    issues.Add(ChildPath(path, "clock_id"), "clock_id_required",
               "clock ID must not be empty");
  }
}

void CheckDuration(const DurationNanoseconds& duration,
                   std::string_view path,
                   IssueCollector& issues,
                   const bool strictly_positive = false) {
  const auto count = duration.value.count();
  if (count < 0) {
    issues.Add(std::string(path), "negative_duration",
               "duration must be non-negative");
  } else if (strictly_positive && count == 0) {
    issues.Add(std::string(path), "positive_duration_required",
               "duration must be strictly positive");
  }
}

void CheckTimeInterval(const TimeInterval& interval,
                       std::string_view path,
                       IssueCollector& issues,
                       const bool strictly_positive_width = false) {
  CheckDuration(interval.start_offset,
                ChildPath(path, "start_offset"), issues);
  CheckDuration(interval.end_offset,
                ChildPath(path, "end_offset"), issues);
  if (interval.end_offset.value < interval.start_offset.value) {
    issues.Add(std::string(path), "interval_bounds_reversed",
               "interval end must not precede interval start");
  } else if (strictly_positive_width &&
             interval.end_offset.value == interval.start_offset.value) {
    issues.Add(std::string(path), "positive_interval_width_required",
               "interval must have strictly positive width");
  }
}

void CheckInterval(const Interval& interval,
                   std::string_view path,
                   IssueCollector& issues) {
  CheckFinite(interval.lower, ChildPath(path, "lower"), issues);
  CheckFinite(interval.upper, ChildPath(path, "upper"), issues);
  if (std::isfinite(interval.lower) && std::isfinite(interval.upper) &&
      interval.lower > interval.upper) {
    issues.Add(std::string(path), "interval_bounds_reversed",
               "lower bound must not exceed upper bound");
  }
}

void CheckVectorBounds(const Vector3Bounds& bounds,
                       std::string_view path,
                       IssueCollector& issues) {
  CheckFinite(bounds.lower, ChildPath(path, "lower"), issues);
  CheckFinite(bounds.upper, ChildPath(path, "upper"), issues);
  if (std::isfinite(bounds.lower.x) && std::isfinite(bounds.upper.x) &&
      bounds.lower.x > bounds.upper.x) {
    issues.Add(ChildPath(path, "x"), "interval_bounds_reversed",
               "lower x bound must not exceed upper x bound");
  }
  if (std::isfinite(bounds.lower.y) && std::isfinite(bounds.upper.y) &&
      bounds.lower.y > bounds.upper.y) {
    issues.Add(ChildPath(path, "y"), "interval_bounds_reversed",
               "lower y bound must not exceed upper y bound");
  }
  if (std::isfinite(bounds.lower.z) && std::isfinite(bounds.upper.z) &&
      bounds.lower.z > bounds.upper.z) {
    issues.Add(ChildPath(path, "z"), "interval_bounds_reversed",
               "lower z bound must not exceed upper z bound");
  }
}

void CheckDeterministicSet(const DeterministicVectorSet3& set,
                           std::string_view path,
                           IssueCollector& issues) {
  std::visit(
      [&](const auto& value) {
        using Set = std::decay_t<decltype(value)>;
        CheckFinite(value.center, ChildPath(path, "center"), issues);
        if constexpr (std::is_same_v<Set, AxisAlignedBox3>) {
          CheckFinite(value.half_extent,
                      ChildPath(path, "half_extent"), issues);
          if (value.half_extent.x < 0.0 || value.half_extent.y < 0.0 ||
              value.half_extent.z < 0.0) {
            issues.Add(ChildPath(path, "half_extent"),
                       "negative_set_extent",
                       "axis-aligned half extents must be non-negative");
          }
        } else {
          CheckNonNegative(value.radius, ChildPath(path, "radius"),
                           issues);
        }
      },
      set);
}

void CheckRotationVectorBall(const RotationVectorBall& ball,
                             std::string_view path,
                             IssueCollector& issues) {
  CheckFinite(ball.radius_rad, ChildPath(path, "radius_rad"), issues);
  if (std::isfinite(ball.radius_rad) &&
      (ball.radius_rad < 0.0 || ball.radius_rad > kPi)) {
    issues.Add(ChildPath(path, "radius_rad"),
               "rotation_radius_out_of_range",
               "rotation-vector radius must be in [0, pi]");
  }
}

void CheckWheeledOrLeggedError(
    const WheeledOrLeggedErrorBounds& bounds,
    std::string_view path,
    IssueCollector& issues) {
  CheckDeterministicSet(
      bounds.position_bound_m,
      ChildPath(path, "position_bound_m"), issues);
  CheckFinite(bounds.yaw_bound_rad.center,
              ChildPath(path, "yaw_bound_rad.center"), issues);
  CheckNonNegative(
      bounds.yaw_bound_rad.half_width,
      ChildPath(path, "yaw_bound_rad.half_width"), issues);
  CheckDeterministicSet(
      bounds.linear_velocity_bound_mps,
      ChildPath(path, "linear_velocity_bound_mps"), issues);
  CheckFinite(bounds.yaw_rate_bound_radps.center,
              ChildPath(path, "yaw_rate_bound_radps.center"), issues);
  CheckNonNegative(
      bounds.yaw_rate_bound_radps.half_width,
      ChildPath(path, "yaw_rate_bound_radps.half_width"), issues);
}

void CheckHopperError(const HopperErrorBounds& bounds,
                      std::string_view path,
                      IssueCollector& issues) {
  CheckDeterministicSet(
      bounds.position_bound_m,
      ChildPath(path, "position_bound_m"), issues);
  CheckRotationVectorBall(
      bounds.orientation_bound,
      ChildPath(path, "orientation_bound"), issues);
  CheckDeterministicSet(
      bounds.linear_velocity_bound_mps,
      ChildPath(path, "linear_velocity_bound_mps"), issues);
  CheckDeterministicSet(
      bounds.angular_velocity_bound_radps,
      ChildPath(path, "angular_velocity_bound_radps"), issues);
}

void CheckWheeledOrLeggedState(
    const WheeledOrLeggedState& state,
    std::string_view path,
    IssueCollector& issues) {
  CheckFinite(state.position_m, ChildPath(path, "position_m"), issues);
  CheckFinite(state.yaw_rad, ChildPath(path, "yaw_rad"), issues);
  CheckFinite(state.linear_velocity_mps,
              ChildPath(path, "linear_velocity_mps"), issues);
  CheckFinite(state.yaw_rate_radps,
              ChildPath(path, "yaw_rate_radps"), issues);
  CheckWheeledOrLeggedError(
      state.error_bounds, ChildPath(path, "error_bounds"), issues);
}

void CheckHopperState(const HopperState& state,
                      std::string_view path,
                      IssueCollector& issues) {
  CheckFinite(state.position_m, ChildPath(path, "position_m"), issues);
  CheckQuaternion(state.orientation_body_to_frame,
                  ChildPath(path, "orientation_body_to_frame"), issues);
  CheckFinite(state.linear_velocity_mps,
              ChildPath(path, "linear_velocity_mps"), issues);
  CheckFinite(state.angular_velocity_radps,
              ChildPath(path, "angular_velocity_radps"), issues);
  CheckHopperError(
      state.error_bounds, ChildPath(path, "error_bounds"), issues);
}

void CheckCircularYawInterval(const CircularYawInterval& interval,
                              std::string_view path,
                              IssueCollector& issues) {
  CheckFinite(interval.start_rad, ChildPath(path, "start_rad"), issues);
  CheckFinite(interval.span_rad, ChildPath(path, "span_rad"), issues);
  if (std::isfinite(interval.start_rad) &&
      (interval.start_rad < -kPi || interval.start_rad >= kPi)) {
    issues.Add(ChildPath(path, "start_rad"),
               "noncanonical_yaw_interval_start",
               "yaw interval start must be in [-pi, pi)");
  }
  if (std::isfinite(interval.span_rad) &&
      (interval.span_rad < 0.0 || interval.span_rad > kTwoPi)) {
    issues.Add(ChildPath(path, "span_rad"),
               "yaw_interval_span_out_of_range",
               "yaw interval span must be in [0, 2*pi]");
  }
  if (!interval.closed) {
    issues.Add(ChildPath(path, "closed"),
               "closed_yaw_interval_required",
               "circular yaw intervals must be closed");
  }
}

[[nodiscard]] double NormalizePositiveAngle(const double angle) {
  double result = std::fmod(angle, kTwoPi);
  if (result < 0.0) {
    result += kTwoPi;
  }
  return result;
}

[[nodiscard]] bool IsYawSubset(const CircularYawInterval& child,
                               const CircularYawInterval& parent) {
  if (!std::isfinite(child.start_rad) ||
      !std::isfinite(child.span_rad) ||
      !std::isfinite(parent.start_rad) ||
      !std::isfinite(parent.span_rad)) {
    return false;
  }
  if (parent.span_rad >= kTwoPi - kGeometryTolerance) {
    return true;
  }
  if (child.span_rad >= kTwoPi - kGeometryTolerance) {
    return false;
  }
  const double child_start_from_parent =
      NormalizePositiveAngle(child.start_rad - parent.start_rad);
  return child_start_from_parent <=
             parent.span_rad + kGeometryTolerance &&
         child_start_from_parent + child.span_rad <=
             parent.span_rad + kGeometryTolerance;
}

void CheckLandingPlane(const LandingPlane& plane,
                       std::string_view path,
                       IssueCollector& issues) {
  CheckFinite(plane.origin_m, ChildPath(path, "origin_m"), issues);
  CheckFinite(plane.normal, ChildPath(path, "normal"), issues);
  CheckFinite(plane.basis_u, ChildPath(path, "basis_u"), issues);
  CheckFinite(plane.basis_v, ChildPath(path, "basis_v"), issues);
  CheckNonNegative(
      plane.residual_bound_m,
      ChildPath(path, "residual_bound_m"), issues);

  const double normal_norm = Norm(plane.normal);
  const double basis_u_norm = Norm(plane.basis_u);
  const double basis_v_norm = Norm(plane.basis_v);
  if (std::isfinite(normal_norm) &&
      std::abs(normal_norm - 1.0) > kGeometryTolerance) {
    issues.Add(ChildPath(path, "normal"), "unit_vector_required",
               "landing-plane normal must be unit length");
  }
  if (std::isfinite(basis_u_norm) &&
      std::abs(basis_u_norm - 1.0) > kGeometryTolerance) {
    issues.Add(ChildPath(path, "basis_u"), "unit_vector_required",
               "landing-plane basis_u must be unit length");
  }
  if (std::isfinite(basis_v_norm) &&
      std::abs(basis_v_norm - 1.0) > kGeometryTolerance) {
    issues.Add(ChildPath(path, "basis_v"), "unit_vector_required",
               "landing-plane basis_v must be unit length");
  }
  if (std::isfinite(Dot(plane.normal, plane.basis_u)) &&
      (std::abs(Dot(plane.normal, plane.basis_u)) >
           kGeometryTolerance ||
       std::abs(Dot(plane.normal, plane.basis_v)) >
           kGeometryTolerance ||
       std::abs(Dot(plane.basis_u, plane.basis_v)) >
           kGeometryTolerance)) {
    issues.Add(std::string(path), "orthogonal_plane_basis_required",
               "landing-plane normal and basis vectors must be orthogonal");
  }
  const Vec3 cross = Cross(plane.basis_u, plane.basis_v);
  if (std::isfinite(Dot(cross, plane.normal)) &&
      Dot(cross, plane.normal) < 1.0 - kGeometryTolerance) {
    issues.Add(std::string(path), "right_handed_plane_basis_required",
               "basis_u cross basis_v must align with the plane normal");
  }
}

[[nodiscard]] double Cross2(const Vec2& a,
                            const Vec2& b,
                            const Vec2& c) {
  return (b.x - a.x) * (c.y - b.y) -
         (b.y - a.y) * (c.x - b.x);
}

void CheckConvexPolygon(const ConvexPolygonUv& polygon,
                        std::string_view path,
                        IssueCollector& issues) {
  for (std::size_t index = 0; index < polygon.vertices_uv.size();
       ++index) {
    CheckFinite(polygon.vertices_uv[index],
                IndexedPath(ChildPath(path, "vertices_uv"), index),
                issues);
  }
  if (polygon.vertices_uv.size() < 3U) {
    issues.Add(ChildPath(path, "vertices_uv"),
               "polygon_requires_three_vertices",
               "convex polygon must have at least three vertices");
    return;
  }
  bool strictly_ccw = true;
  for (std::size_t index = 0; index < polygon.vertices_uv.size();
       ++index) {
    const Vec2& a = polygon.vertices_uv[index];
    const Vec2& b =
        polygon.vertices_uv[(index + 1U) % polygon.vertices_uv.size()];
    const Vec2& c =
        polygon.vertices_uv[(index + 2U) % polygon.vertices_uv.size()];
    const double cross = Cross2(a, b, c);
    if (!std::isfinite(cross) || !(cross > kPolygonTolerance)) {
      strictly_ccw = false;
      break;
    }
  }
  if (!strictly_ccw) {
    issues.Add(std::string(path), "strict_convex_ccw_polygon_required",
               "vertices must form a finite strictly convex CCW polygon");
  }
}

void CheckHalfspacePolytope(const ConvexPolytope3& polytope,
                            std::string_view path,
                            IssueCollector& issues) {
  if (polytope.halfspaces.size() < 4U) {
    issues.Add(ChildPath(path, "halfspaces"),
               "polytope_requires_four_halfspaces",
               "a 3D H-polytope needs at least four halfspaces");
  }
  for (std::size_t index = 0; index < polytope.halfspaces.size();
       ++index) {
    const Halfspace3& halfspace = polytope.halfspaces[index];
    const std::string halfspace_path =
        IndexedPath(ChildPath(path, "halfspaces"), index);
    CheckFinite(halfspace.normal,
                ChildPath(halfspace_path, "normal"), issues);
    CheckFinite(halfspace.offset_m,
                ChildPath(halfspace_path, "offset_m"), issues);
    const double norm = Norm(halfspace.normal);
    if (std::isfinite(norm) &&
        std::abs(norm - 1.0) > kGeometryTolerance) {
      issues.Add(ChildPath(halfspace_path, "normal"),
                 "unit_vector_required",
                 "halfspace normal must be unit length");
    }
  }
}

[[nodiscard]] bool NearlyEqual(const double lhs, const double rhs) {
  return std::abs(lhs - rhs) <= kGeometryTolerance;
}

[[nodiscard]] bool SamePlane(const LandingPlane& lhs,
                             const LandingPlane& rhs) {
  return Norm(Subtract(lhs.origin_m, rhs.origin_m)) <=
             kGeometryTolerance &&
         Norm(Subtract(lhs.normal, rhs.normal)) <=
             kGeometryTolerance &&
         Norm(Subtract(lhs.basis_u, rhs.basis_u)) <=
             kGeometryTolerance &&
         Norm(Subtract(lhs.basis_v, rhs.basis_v)) <=
             kGeometryTolerance &&
         NearlyEqual(lhs.residual_bound_m, rhs.residual_bound_m);
}

[[nodiscard]] Vec3 PlanePointFromUv(const LandingPlane& plane,
                                    const Vec2& uv) {
  return Add(
      plane.origin_m,
      Add(Scale(plane.basis_u, uv.x),
          Scale(plane.basis_v, uv.y)));
}

[[nodiscard]] Vec2 ProjectToPlaneUv(const LandingPlane& plane,
                                    const Vec3& point) {
  const Vec3 relative = Subtract(point, plane.origin_m);
  return Vec2{Dot(relative, plane.basis_u),
              Dot(relative, plane.basis_v)};
}

[[nodiscard]] bool PolygonWithMarginContained(
    const ConvexPolygonUv& inner,
    const LandingPlane& inner_plane,
    const ConvexPolygonUv& outer,
    const LandingPlane& outer_plane,
    const double margin) {
  if (inner.vertices_uv.size() < 3U ||
      outer.vertices_uv.size() < 3U || !std::isfinite(margin) ||
      margin < 0.0) {
    return false;
  }
  for (const Vec2& inner_uv : inner.vertices_uv) {
    const Vec3 world_point = PlanePointFromUv(inner_plane, inner_uv);
    const Vec2 outer_uv = ProjectToPlaneUv(outer_plane, world_point);
    for (std::size_t index = 0; index < outer.vertices_uv.size();
         ++index) {
      const Vec2& edge_start = outer.vertices_uv[index];
      const Vec2& edge_end =
          outer.vertices_uv[(index + 1U) % outer.vertices_uv.size()];
      const double edge_x = edge_end.x - edge_start.x;
      const double edge_y = edge_end.y - edge_start.y;
      const double edge_length = std::hypot(edge_x, edge_y);
      const double inward_cross =
          edge_x * (outer_uv.y - edge_start.y) -
          edge_y * (outer_uv.x - edge_start.x);
      if (!std::isfinite(inward_cross) ||
          inward_cross + kGeometryTolerance <
              margin * edge_length) {
        return false;
      }
    }
  }
  return true;
}

void CheckPose(const PoseXyzYaw& pose,
               std::string_view path,
               IssueCollector& issues) {
  CheckFinite(pose.position_m, ChildPath(path, "position_m"), issues);
  CheckFinite(pose.yaw_rad, ChildPath(path, "yaw_rad"), issues);
}

void CheckPolynomialSegment(const CubicPolynomialSegment& segment,
                            std::string_view path,
                            IssueCollector& issues) {
  CheckDuration(segment.start_offset,
                ChildPath(path, "start_offset"), issues);
  CheckDuration(segment.end_offset,
                ChildPath(path, "end_offset"), issues);
  if (segment.end_offset.value < segment.start_offset.value) {
    issues.Add(std::string(path), "interval_bounds_reversed",
               "polynomial segment end must not precede start");
  }
  for (std::size_t index = 0; index < segment.coefficients.size();
       ++index) {
    CheckFinite(segment.coefficients[index],
                IndexedPath(ChildPath(path, "coefficients"), index),
                issues);
  }
}

void CheckPiecewiseTrajectory(
    const PiecewiseCubicScalarTrajectory& trajectory,
    std::string_view path,
    IssueCollector& issues) {
  if (trajectory.value_semantics.empty()) {
    issues.Add(ChildPath(path, "value_semantics"),
               "value_semantics_required",
               "piecewise trajectory must declare value semantics");
  }
  if (trajectory.segments.empty()) {
    issues.Add(ChildPath(path, "segments"), "trajectory_segment_required",
               "piecewise trajectory must contain at least one segment");
  }
  for (std::size_t index = 0; index < trajectory.segments.size();
       ++index) {
    CheckPolynomialSegment(
        trajectory.segments[index],
        IndexedPath(ChildPath(path, "segments"), index), issues);
    if (index > 0U &&
        trajectory.segments[index].start_offset.value !=
            trajectory.segments[index - 1U].end_offset.value) {
      issues.Add(
          IndexedPath(ChildPath(path, "segments"), index),
          "piecewise_trajectory_discontinuity",
          "piecewise trajectory intervals must be contiguous");
    }
  }
}

[[nodiscard]] double EvaluatePolynomial(
    const CubicPolynomialSegment& segment,
    const double tau_seconds) {
  return ((segment.coefficients[3] * tau_seconds +
           segment.coefficients[2]) *
              tau_seconds +
          segment.coefficients[1]) *
             tau_seconds +
         segment.coefficients[0];
}

[[nodiscard]] double EvaluatePolynomialDerivative(
    const CubicPolynomialSegment& segment,
    const double tau_seconds) {
  return (3.0 * segment.coefficients[3] * tau_seconds +
          2.0 * segment.coefficients[2]) *
             tau_seconds +
         segment.coefficients[1];
}

void CheckTimeScaling(const MonotoneTimeScaling& scaling,
                      std::string_view path,
                      IssueCollector& issues) {
  if (scaling.segments.empty()) {
    issues.Add(ChildPath(path, "segments"),
               "time_scaling_segment_required",
               "time scaling must contain at least one segment");
    return;
  }
  for (std::size_t index = 0; index < scaling.segments.size();
       ++index) {
    const CubicPolynomialSegment& segment = scaling.segments[index];
    const std::string segment_path =
        IndexedPath(ChildPath(path, "segments"), index);
    CheckPolynomialSegment(segment, segment_path, issues);
    if (index > 0U) {
      const CubicPolynomialSegment& previous =
          scaling.segments[index - 1U];
      if (segment.start_offset.value != previous.end_offset.value) {
        issues.Add(segment_path, "time_scaling_time_discontinuity",
                   "time-scaling segments must have contiguous time");
      }
      const double previous_duration =
          std::chrono::duration<double>(
              previous.end_offset.value -
              previous.start_offset.value)
              .count();
      if (!NearlyEqual(EvaluatePolynomial(previous, previous_duration),
                       EvaluatePolynomial(segment, 0.0))) {
        issues.Add(segment_path, "time_scaling_value_discontinuity",
                   "time-scaling values must be continuous");
      }
    }
    const double duration_seconds =
        std::chrono::duration<double>(
            segment.end_offset.value - segment.start_offset.value)
            .count();
    const std::array<double, 3> samples{
        0.0, 0.5 * duration_seconds, duration_seconds};
    for (const double sample : samples) {
      const double derivative =
          EvaluatePolynomialDerivative(segment, sample);
      if (!std::isfinite(derivative) ||
          derivative < -kGeometryTolerance) {
        issues.Add(segment_path, "nonmonotone_time_scaling",
                   "time-scaling derivative must be non-negative");
        break;
      }
    }
    if (std::abs(segment.coefficients[3]) > kPolygonTolerance) {
      const double critical =
          -segment.coefficients[2] /
          (3.0 * segment.coefficients[3]);
      if (critical > 0.0 && critical < duration_seconds &&
          EvaluatePolynomialDerivative(segment, critical) <
              -kGeometryTolerance) {
        issues.Add(segment_path, "nonmonotone_time_scaling",
                   "time-scaling derivative must be non-negative");
      }
    }
  }
  if (!NearlyEqual(EvaluatePolynomial(scaling.segments.front(), 0.0),
                   0.0)) {
    issues.Add(std::string(path), "time_scaling_start_value_mismatch",
               "time scaling must start at zero");
  }
  const CubicPolynomialSegment& last = scaling.segments.back();
  const double last_duration =
      std::chrono::duration<double>(
          last.end_offset.value - last.start_offset.value)
          .count();
  if (!NearlyEqual(EvaluatePolynomial(last, last_duration), 1.0)) {
    issues.Add(std::string(path), "time_scaling_end_value_mismatch",
               "time scaling must end at one");
  }
}

void CheckGeometricPath(const GeometricPath& path,
                        std::string_view field_path,
                        IssueCollector& issues) {
  std::visit(
      [&](const auto& value) {
        using Path = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Path, ClampedCubicBSplinePath>) {
          for (std::size_t index = 0; index < value.knots.size();
               ++index) {
            CheckFinite(value.knots[index],
                        IndexedPath(
                            ChildPath(field_path, "knots"), index),
                        issues);
            if (index > 0U &&
                value.knots[index] < value.knots[index - 1U]) {
              issues.Add(ChildPath(field_path, "knots"),
                         "nonmonotone_spline_knots",
                         "B-spline knots must be nondecreasing");
            }
          }
          for (std::size_t index = 0;
               index < value.control_points.size(); ++index) {
            CheckPose(
                value.control_points[index],
                IndexedPath(
                    ChildPath(field_path, "control_points"), index),
                issues);
          }
          if (value.control_points.size() < 4U ||
              value.knots.size() != value.control_points.size() + 4U) {
            issues.Add(std::string(field_path),
                       "invalid_cubic_bspline_cardinality",
                       "cubic B-spline needs at least four control points "
                       "and knots=control_points+4");
          } else {
            const bool clamped_start =
                std::all_of(
                    value.knots.begin(), value.knots.begin() + 4,
                    [&](const double knot) {
                      return NearlyEqual(knot, value.knots.front());
                    });
            const bool clamped_end =
                std::all_of(
                    value.knots.end() - 4, value.knots.end(),
                    [&](const double knot) {
                      return NearlyEqual(knot, value.knots.back());
                    });
            if (!clamped_start || !clamped_end) {
              issues.Add(ChildPath(field_path, "knots"),
                         "clamped_cubic_spline_required",
                         "first and last four knots must be clamped");
            }
          }
        } else {
          if (value.primitives.empty()) {
            issues.Add(ChildPath(field_path, "primitives"),
                       "validated_primitive_required",
                       "validated primitive chain must not be empty");
          }
          for (std::size_t index = 0;
               index < value.primitives.size(); ++index) {
            const ValidatedPrimitive& primitive =
                value.primitives[index];
            const std::string primitive_path =
                IndexedPath(
                    ChildPath(field_path, "primitives"), index);
            CheckPose(primitive.start_pose,
                      ChildPath(primitive_path, "start_pose"), issues);
            CheckPose(primitive.end_pose,
                      ChildPath(primitive_path, "end_pose"), issues);
            CheckDuration(
                primitive.nominal_duration,
                ChildPath(primitive_path, "nominal_duration"), issues,
                true);
            CheckContentRef(
                primitive.validation_ref,
                ChildPath(primitive_path, "validation_ref"), issues);
            if (index > 0U) {
              const PoseXyzYaw& previous_end =
                  value.primitives[index - 1U].end_pose;
              if (Norm(Subtract(previous_end.position_m,
                                primitive.start_pose.position_m)) >
                      kGeometryTolerance ||
                  !NearlyEqual(previous_end.yaw_rad,
                               primitive.start_pose.yaw_rad)) {
                issues.Add(primitive_path,
                           "primitive_chain_discontinuity",
                           "validated primitives must connect exactly");
              }
            }
          }
        }
      },
      path);
}

void CheckPosePolygon(const std::vector<Vec2>& vertices,
                      std::string_view path,
                      IssueCollector& issues) {
  ConvexPolygonUv polygon{.vertices_uv = vertices};
  CheckConvexPolygon(polygon, path, issues);
}

void CheckWheelCapability(const WheeledCapability& capability,
                          std::string_view path,
                          IssueCollector& issues) {
  if (capability.frame_id.empty()) {
    issues.Add(ChildPath(path, "frame_id"), "frame_id_required",
               "capability frame ID must not be empty");
  }
  CheckPosePolygon(
      capability.collision_envelope.vertices_xy_m,
      ChildPath(path, "collision_envelope"), issues);
  CheckFinite(
      capability.collision_envelope.minimum_z_m,
      ChildPath(path, "collision_envelope.minimum_z_m"), issues);
  CheckFinite(
      capability.collision_envelope.maximum_z_m,
      ChildPath(path, "collision_envelope.maximum_z_m"), issues);
  if (std::isfinite(capability.collision_envelope.minimum_z_m) &&
      std::isfinite(capability.collision_envelope.maximum_z_m) &&
      capability.collision_envelope.minimum_z_m >
          capability.collision_envelope.maximum_z_m) {
    issues.Add(ChildPath(path, "collision_envelope"),
               "interval_bounds_reversed",
               "minimum_z_m must not exceed maximum_z_m");
  }
  CheckContentRef(capability.motion_model_ref,
                  ChildPath(path, "motion_model_ref"), issues);
  CheckContentRef(capability.analytic_cost_model_ref,
                  ChildPath(path, "analytic_cost_model_ref"), issues);

  const WheelHardLimits& limits = capability.hard_limits;
  CheckPositive(limits.maximum_forward_speed_mps,
                ChildPath(path, "hard_limits.maximum_forward_speed_mps"),
                issues);
  CheckPositive(limits.maximum_reverse_speed_mps,
                ChildPath(path, "hard_limits.maximum_reverse_speed_mps"),
                issues);
  CheckPositive(limits.maximum_spin_rate_radps,
                ChildPath(path, "hard_limits.maximum_spin_rate_radps"),
                issues);
  CheckPositive(
      limits.maximum_forward_acceleration_mps2,
      ChildPath(path,
                "hard_limits.maximum_forward_acceleration_mps2"),
      issues);
  CheckPositive(
      limits.maximum_braking_deceleration_mps2,
      ChildPath(path,
                "hard_limits.maximum_braking_deceleration_mps2"),
      issues);
  CheckPositive(
      limits.maximum_yaw_acceleration_radps2,
      ChildPath(path, "hard_limits.maximum_yaw_acceleration_radps2"),
      issues);
  CheckPositive(
      limits.maximum_lateral_acceleration_mps2,
      ChildPath(path,
                "hard_limits.maximum_lateral_acceleration_mps2"),
      issues);
  CheckPositive(
      limits.maximum_drive_curvature_per_m,
      ChildPath(path, "hard_limits.maximum_drive_curvature_per_m"),
      issues);
  CheckNonNegative(limits.maximum_slope_rad,
                   ChildPath(path, "hard_limits.maximum_slope_rad"),
                   issues);
  if (std::isfinite(limits.maximum_slope_rad) &&
      limits.maximum_slope_rad > kPi / 2.0) {
    issues.Add(ChildPath(path, "hard_limits.maximum_slope_rad"),
               "slope_limit_out_of_range",
               "maximum slope must not exceed pi/2");
  }
  CheckNonNegative(limits.minimum_clearance_m,
                   ChildPath(path, "hard_limits.minimum_clearance_m"),
                   issues);
  CheckWheeledOrLeggedError(
      capability.certified_state_error_bounds,
      ChildPath(path, "certified_state_error_bounds"), issues);

  if (capability.motion_primitives.size() < 6U) {
    issues.Add(ChildPath(path, "motion_primitives"),
               "insufficient_motion_primitives",
               "wheeled capability requires at least six primitives");
  }
  std::set<PrimitiveId> primitive_ids;
  for (std::size_t index = 0;
       index < capability.motion_primitives.size(); ++index) {
    const WheelMotionPrimitive& primitive =
        capability.motion_primitives[index];
    const std::string primitive_path =
        IndexedPath(ChildPath(path, "motion_primitives"), index);
    if (primitive.primitive_id.empty()) {
      issues.Add(ChildPath(primitive_path, "primitive_id"),
                 "identifier_required",
                 "primitive ID must not be empty");
    } else if (!primitive_ids.insert(primitive.primitive_id).second) {
      issues.Add(ChildPath(primitive_path, "primitive_id"),
                 "duplicate_primitive_id",
                 "primitive IDs must be unique");
    }
    CheckPose(primitive.relative_end_pose,
              ChildPath(primitive_path, "relative_end_pose"), issues);
    CheckDuration(
        primitive.nominal_duration,
        ChildPath(primitive_path, "nominal_duration"), issues, true);
    CheckContentRef(
        primitive.swept_geometry_ref,
        ChildPath(primitive_path, "swept_geometry_ref"), issues);
  }
}

void CheckLeggedCapability(const LeggedCapability& capability,
                           std::string_view path,
                           IssueCollector& issues) {
  if (capability.frame_id.empty()) {
    issues.Add(ChildPath(path, "frame_id"), "frame_id_required",
               "capability frame ID must not be empty");
  }
  if (capability.reference_point_id.empty()) {
    issues.Add(ChildPath(path, "reference_point_id"),
               "reference_point_id_required",
               "legged capability needs a fixed reference point ID");
  }
  CheckHalfspacePolytope(
      capability.collision_envelope.body_frame_halfspaces,
      ChildPath(path, "collision_envelope.body_frame_halfspaces"),
      issues);
  CheckContentRef(capability.motion_model_ref,
                  ChildPath(path, "motion_model_ref"), issues);
  CheckContentRef(capability.analytic_cost_model_ref,
                  ChildPath(path, "analytic_cost_model_ref"), issues);

  const LeggedTerrainThresholds& terrain =
      capability.terrain_thresholds;
  CheckNonNegative(
      terrain.maximum_slope_rad,
      ChildPath(path, "terrain_thresholds.maximum_slope_rad"), issues);
  if (std::isfinite(terrain.maximum_slope_rad) &&
      terrain.maximum_slope_rad > kPi / 2.0) {
    issues.Add(
        ChildPath(path, "terrain_thresholds.maximum_slope_rad"),
        "slope_limit_out_of_range",
        "maximum slope must not exceed pi/2");
  }
  CheckNonNegative(
      terrain.maximum_roughness_m,
      ChildPath(path, "terrain_thresholds.maximum_roughness_m"),
      issues);
  CheckNonNegative(
      terrain.maximum_step_height_m,
      ChildPath(path, "terrain_thresholds.maximum_step_height_m"),
      issues);
  CheckNonNegative(
      terrain.maximum_gap_width_m,
      ChildPath(path, "terrain_thresholds.maximum_gap_width_m"),
      issues);
  CheckFinite(
      terrain.minimum_confidence,
      ChildPath(path, "terrain_thresholds.minimum_confidence"), issues);
  if (std::isfinite(terrain.minimum_confidence) &&
      (terrain.minimum_confidence < 0.0 ||
       terrain.minimum_confidence > 1.0)) {
    issues.Add(
        ChildPath(path, "terrain_thresholds.minimum_confidence"),
        "confidence_out_of_range",
        "minimum confidence must be in [0, 1]");
  }
  CheckNonNegative(
      terrain.minimum_body_clearance_m,
      ChildPath(path, "terrain_thresholds.minimum_body_clearance_m"),
      issues);
  CheckFinite(
      terrain.minimum_body_height_m,
      ChildPath(path, "terrain_thresholds.minimum_body_height_m"),
      issues);
  CheckFinite(
      terrain.maximum_body_height_m,
      ChildPath(path, "terrain_thresholds.maximum_body_height_m"),
      issues);
  if (std::isfinite(terrain.minimum_body_height_m) &&
      std::isfinite(terrain.maximum_body_height_m) &&
      terrain.minimum_body_height_m > terrain.maximum_body_height_m) {
    issues.Add(ChildPath(path, "terrain_thresholds.body_height"),
               "interval_bounds_reversed",
               "minimum body height must not exceed maximum");
  }

  const LeggedBodyVelocityLimits& velocity =
      capability.body_velocity_limits;
  CheckInterval(velocity.forward_mps,
                ChildPath(path, "body_velocity_limits.forward_mps"),
                issues);
  CheckInterval(velocity.lateral_mps,
                ChildPath(path, "body_velocity_limits.lateral_mps"),
                issues);
  CheckInterval(velocity.vertical_mps,
                ChildPath(path, "body_velocity_limits.vertical_mps"),
                issues);
  CheckInterval(velocity.yaw_rate_radps,
                ChildPath(path, "body_velocity_limits.yaw_rate_radps"),
                issues);
  CheckPositive(
      velocity.linear_acceleration_mps2,
      ChildPath(path,
                "body_velocity_limits.linear_acceleration_mps2"),
      issues);
  CheckPositive(
      velocity.yaw_acceleration_radps2,
      ChildPath(path, "body_velocity_limits.yaw_acceleration_radps2"),
      issues);
  CheckWheeledOrLeggedError(
      capability.certified_state_error_bounds,
      ChildPath(path, "certified_state_error_bounds"), issues);
  if (capability.motion_primitives.size() < 4U) {
    issues.Add(ChildPath(path, "motion_primitives"),
               "insufficient_motion_primitives",
               "legged capability requires at least four primitives");
  }
  std::set<PrimitiveId> primitive_ids;
  for (std::size_t index = 0;
       index < capability.motion_primitives.size(); ++index) {
    const LeggedBodyPrimitive& primitive =
        capability.motion_primitives[index];
    const std::string primitive_path =
        IndexedPath(ChildPath(path, "motion_primitives"), index);
    if (primitive.primitive_id.empty()) {
      issues.Add(ChildPath(primitive_path, "primitive_id"),
                 "identifier_required",
                 "primitive ID must not be empty");
    } else if (!primitive_ids.insert(primitive.primitive_id).second) {
      issues.Add(ChildPath(primitive_path, "primitive_id"),
                 "duplicate_primitive_id",
                 "primitive IDs must be unique");
    }
    CheckFinite(
        primitive.body_frame_displacement_m,
        ChildPath(primitive_path, "body_frame_displacement_m"), issues);
    CheckFinite(primitive.yaw_change_rad,
                ChildPath(primitive_path, "yaw_change_rad"), issues);
    CheckDuration(
        primitive.nominal_duration,
        ChildPath(primitive_path, "nominal_duration"), issues, true);
    CheckContentRef(
        primitive.sampled_body_sweep_ref,
        ChildPath(primitive_path, "sampled_body_sweep_ref"), issues);
  }
  if (capability.footstep_feasibility_guaranteed) {
    issues.Add(ChildPath(path, "footstep_feasibility_guaranteed"),
               "footstep_guarantee_forbidden",
               "v3 legged capability does not guarantee footsteps");
  }
}

void CheckHopperCapability(const HopperCapability& capability,
                           std::string_view path,
                           IssueCollector& issues) {
  if (capability.frame_id.empty()) {
    issues.Add(ChildPath(path, "frame_id"), "frame_id_required",
               "capability frame ID must not be empty");
  }
  CheckHalfspacePolytope(
      capability.collision_envelope.body_frame_halfspaces,
      ChildPath(path, "collision_envelope.body_frame_halfspaces"),
      issues);
  CheckContentRef(capability.motion_model_ref,
                  ChildPath(path, "motion_model_ref"), issues);
  CheckContentRef(capability.analytic_cost_model_ref,
                  ChildPath(path, "analytic_cost_model_ref"), issues);
  CheckContentRef(capability.gravity_model_ref,
                  ChildPath(path, "gravity_model_ref"), issues);
  if (capability.attitude_tightening_table_ref.has_value()) {
    CheckContentRef(
        *capability.attitude_tightening_table_ref,
        ChildPath(path, "attitude_tightening_table_ref"), issues);
  }

  const HopperLandingTerrainThresholds& terrain =
      capability.landing_terrain_thresholds;
  CheckNonNegative(
      terrain.maximum_slope_rad,
      ChildPath(path,
                "landing_terrain_thresholds.maximum_slope_rad"),
      issues);
  if (std::isfinite(terrain.maximum_slope_rad) &&
      terrain.maximum_slope_rad > kPi / 2.0) {
    issues.Add(
        ChildPath(path,
                  "landing_terrain_thresholds.maximum_slope_rad"),
        "slope_limit_out_of_range",
        "maximum slope must not exceed pi/2");
  }
  CheckNonNegative(
      terrain.maximum_roughness_m,
      ChildPath(path,
                "landing_terrain_thresholds.maximum_roughness_m"),
      issues);
  CheckNonNegative(
      terrain.maximum_plane_residual_m,
      ChildPath(path,
                "landing_terrain_thresholds.maximum_plane_residual_m"),
      issues);
  CheckNonNegative(
      terrain.minimum_overhead_clearance_m,
      ChildPath(path,
                "landing_terrain_thresholds.minimum_overhead_clearance_m"),
      issues);
  CheckNonNegative(
      terrain.minimum_lateral_clearance_m,
      ChildPath(path,
                "landing_terrain_thresholds.minimum_lateral_clearance_m"),
      issues);
  CheckPositive(
      terrain.minimum_landing_region_area_m2,
      ChildPath(path,
                "landing_terrain_thresholds.minimum_landing_region_area_m2"),
      issues);

  const HopperLaunchLimits& launch = capability.launch_limits;
  CheckPositive(
      launch.maximum_launch_speed_mps,
      ChildPath(path, "launch_limits.maximum_launch_speed_mps"),
      issues);
  CheckPositive(
      launch.maximum_launch_impulse_newton_seconds,
      ChildPath(path,
                "launch_limits.maximum_launch_impulse_newton_seconds"),
      issues);
  CheckDuration(
      launch.minimum_flight_time,
      ChildPath(path, "launch_limits.minimum_flight_time"), issues);
  CheckDuration(
      launch.maximum_flight_time,
      ChildPath(path, "launch_limits.maximum_flight_time"), issues,
      true);
  if (launch.minimum_flight_time.value >
      launch.maximum_flight_time.value) {
    issues.Add(ChildPath(path, "launch_limits.flight_time"),
               "capability_flight_interval_reversed",
               "minimum flight time must not exceed positive maximum");
  }
  CheckPositive(
      launch.maximum_landing_speed_mps,
      ChildPath(path, "launch_limits.maximum_landing_speed_mps"),
      issues);
  CheckNonNegative(
      launch.minimum_downward_impact_speed_mps,
      ChildPath(path,
                "launch_limits.minimum_downward_impact_speed_mps"),
      issues);
  CheckNonNegative(
      launch.minimum_landing_clearance_m,
      ChildPath(path, "launch_limits.minimum_landing_clearance_m"),
      issues);

  const ArbitraryAxisAttitudeEnvelope& attitude =
      capability.attitude_envelope;
  CheckPositive(
      attitude.maximum_angular_speed_radps,
      ChildPath(path,
                "attitude_envelope.maximum_angular_speed_radps"),
      issues);
  CheckPositive(
      attitude.maximum_angular_acceleration_radps2,
      ChildPath(path,
                "attitude_envelope.maximum_angular_acceleration_radps2"),
      issues);
  CheckNonNegative(
      attitude.maximum_initial_angular_speed_radps,
      ChildPath(path,
                "attitude_envelope.maximum_initial_angular_speed_radps"),
      issues);
  CheckDuration(
      attitude.minimum_settle_guard,
      ChildPath(path, "attitude_envelope.minimum_settle_guard"),
      issues);
  CheckHopperError(
      capability.certified_state_error_bounds,
      ChildPath(path, "certified_state_error_bounds"), issues);
}

void CheckCapabilityProfile(const SafetyCapabilityProfile& capability,
                            IssueCollector& issues) {
  CheckContentRef(capability.content_ref, "content_ref", issues);
  std::visit(
      [&](const auto& content) {
        using Capability = std::decay_t<decltype(content)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          CheckWheelCapability(content, "content", issues);
        } else if constexpr (std::is_same_v<Capability,
                                            LeggedCapability>) {
          CheckLeggedCapability(content, "content", issues);
        } else {
          CheckHopperCapability(content, "content", issues);
        }
      },
      capability.content);
}

void CheckResourceCaps(const ResourceCaps& caps,
                       std::string_view path,
                       IssueCollector& issues) {
  if (caps.maximum_expanded_states < 1U) {
    issues.Add(ChildPath(path, "maximum_expanded_states"),
               "resource_cap_too_small",
               "maximum_expanded_states must be at least one");
  }
  if (caps.maximum_generated_candidates < 1U) {
    issues.Add(ChildPath(path, "maximum_generated_candidates"),
               "resource_cap_too_small",
               "maximum_generated_candidates must be at least one");
  }
  if (caps.maximum_open_states < 1U) {
    issues.Add(ChildPath(path, "maximum_open_states"),
               "resource_cap_too_small",
               "maximum_open_states must be at least one");
  }
  if (caps.maximum_memory_bytes < 1'048'576U) {
    issues.Add(ChildPath(path, "maximum_memory_bytes"),
               "resource_cap_too_small",
               "maximum_memory_bytes must be at least 1048576");
  }
}

void CheckCorridorConfig(const CorridorConfig& config,
                         std::string_view path,
                         IssueCollector& issues) {
  if (config.maximum_regions < 1U) {
    issues.Add(ChildPath(path, "maximum_regions"),
               "resource_cap_too_small",
               "maximum_regions must be at least one");
  }
  if (config.maximum_inflation_iterations < 1U) {
    issues.Add(ChildPath(path, "maximum_inflation_iterations"),
               "resource_cap_too_small",
               "maximum_inflation_iterations must be at least one");
  }
  if (config.maximum_halfplanes_per_region < 3U) {
    issues.Add(ChildPath(path, "maximum_halfplanes_per_region"),
               "resource_cap_too_small",
               "a region needs at least three halfplanes");
  }
  CheckNonNegative(
      config.minimum_overlap_m,
      ChildPath(path, "minimum_overlap_m"), issues);
  CheckPositive(
      config.sampling_spacing_m,
      ChildPath(path, "sampling_spacing_m"), issues);
}

void CheckSmoothingConfig(const SmoothingConfig& config,
                          std::string_view path,
                          IssueCollector& issues) {
  if (config.maximum_scp_iterations < 1U) {
    issues.Add(ChildPath(path, "maximum_scp_iterations"),
               "resource_cap_too_small",
               "maximum_scp_iterations must be at least one");
  }
  CheckPositive(
      config.initial_trust_region_m,
      ChildPath(path, "initial_trust_region_m"), issues);
  CheckPositive(
      config.minimum_trust_region_m,
      ChildPath(path, "minimum_trust_region_m"), issues);
  if (std::isfinite(config.initial_trust_region_m) &&
      std::isfinite(config.minimum_trust_region_m) &&
      config.minimum_trust_region_m >
          config.initial_trust_region_m) {
    issues.Add(std::string(path), "trust_region_bounds_reversed",
               "minimum trust region must not exceed initial");
  }
  CheckPositive(
      config.constraint_tolerance,
      ChildPath(path, "constraint_tolerance"), issues);
  CheckDuration(
      config.maximum_time_increase,
      ChildPath(path, "maximum_time_increase"), issues);
}

void CheckTimeScalingConfig(const TimeScalingConfig& config,
                            std::string_view path,
                            IssueCollector& issues) {
  if (config.maximum_adaptive_samples < 2U) {
    issues.Add(ChildPath(path, "maximum_adaptive_samples"),
               "resource_cap_too_small",
               "maximum_adaptive_samples must be at least two");
  }
  CheckPositive(
      config.minimum_parameter_step,
      ChildPath(path, "minimum_parameter_step"), issues);
  if (std::isfinite(config.minimum_parameter_step) &&
      config.minimum_parameter_step > 1.0) {
    issues.Add(ChildPath(path, "minimum_parameter_step"),
               "parameter_step_out_of_range",
               "minimum parameter step must not exceed one");
  }
  if (config.maximum_forward_passes < 1U) {
    issues.Add(ChildPath(path, "maximum_forward_passes"),
               "resource_cap_too_small",
               "maximum_forward_passes must be at least one");
  }
  if (config.maximum_backward_passes < 1U) {
    issues.Add(ChildPath(path, "maximum_backward_passes"),
               "resource_cap_too_small",
               "maximum_backward_passes must be at least one");
  }
}

void CheckGridConfig(const GridConfig& config,
                     std::string_view path,
                     IssueCollector& issues) {
  CheckPositive(config.xy_resolution_m,
                ChildPath(path, "xy_resolution_m"), issues);
  if (config.yaw_bin_count < 4U) {
    issues.Add(ChildPath(path, "yaw_bin_count"),
               "resource_cap_too_small",
               "yaw_bin_count must be at least four");
  }
  if (config.maximum_terminal_candidates < 1U) {
    issues.Add(ChildPath(path, "maximum_terminal_candidates"),
               "resource_cap_too_small",
               "maximum_terminal_candidates must be at least one");
  }
}

void CheckPlannerConfig(const PlannerAlgorithmConfig& config,
                        IssueCollector& issues) {
  CheckContentRef(config.content_ref, "content_ref", issues);
  CheckDuration(config.time_equivalence_tolerance,
                "time_equivalence_tolerance", issues);
  CheckDuration(config.max_input_skew, "max_input_skew", issues);
  if (config.error_bound_model_id.empty()) {
    issues.Add("error_bound_model_id", "identifier_required",
               "error-bound model ID must not be empty");
  }
  if (config.projection_cache_capacity < 1U) {
    issues.Add("projection_cache_capacity", "resource_cap_too_small",
               "projection cache capacity must be at least one");
  }

  CheckFinite(config.ara_star.initial_epsilon,
              "ara_star.initial_epsilon", issues);
  CheckFinite(config.ara_star.target_epsilon,
              "ara_star.target_epsilon", issues);
  CheckFinite(config.ara_star.epsilon_decrement,
              "ara_star.epsilon_decrement", issues);
  if (std::isfinite(config.ara_star.initial_epsilon) &&
      std::isfinite(config.ara_star.target_epsilon) &&
      (!(config.ara_star.target_epsilon >= 1.0) ||
       config.ara_star.initial_epsilon <
           config.ara_star.target_epsilon)) {
    issues.Add("ara_star", "invalid_epsilon_schedule",
               "initial_epsilon must be >= target_epsilon >= 1");
  }
  if (std::isfinite(config.ara_star.epsilon_decrement) &&
      !(config.ara_star.epsilon_decrement > 0.0)) {
    issues.Add("ara_star.epsilon_decrement",
               "positive_epsilon_decrement_required",
               "epsilon decrement must be strictly positive");
  }
  CheckResourceCaps(config.ara_star.resource_caps,
                    "ara_star.resource_caps", issues);

  CheckGridConfig(config.wheeled.state_lattice,
                  "wheeled.state_lattice", issues);
  CheckCorridorConfig(config.wheeled.corridor,
                      "wheeled.corridor", issues);
  CheckSmoothingConfig(config.wheeled.smoothing,
                       "wheeled.smoothing", issues);
  CheckTimeScalingConfig(config.wheeled.time_scaling,
                         "wheeled.time_scaling", issues);
  if (config.wheeled.continuous_validation_maximum_subdivisions < 1U) {
    issues.Add("wheeled.continuous_validation_maximum_subdivisions",
               "resource_cap_too_small",
               "continuous validation needs at least one subdivision");
  }

  CheckGridConfig(config.legged.pose_lattice,
                  "legged.pose_lattice", issues);
  if (config.legged.maximum_height_interval_splits < 1U) {
    issues.Add("legged.maximum_height_interval_splits",
               "resource_cap_too_small",
               "height interval splits must be at least one");
  }
  CheckCorridorConfig(config.legged.corridor,
                      "legged.corridor", issues);
  CheckSmoothingConfig(config.legged.smoothing,
                       "legged.smoothing", issues);
  CheckTimeScalingConfig(config.legged.time_scaling,
                         "legged.time_scaling", issues);
  if (config.legged.continuous_validation_maximum_subdivisions < 1U) {
    issues.Add("legged.continuous_validation_maximum_subdivisions",
               "resource_cap_too_small",
               "continuous validation needs at least one subdivision");
  }

  const HopperAlgorithmConfig& hopper = config.hopper;
  const std::array<std::pair<std::size_t, std::string_view>, 10>
      positive_hopper_caps{{
          {hopper.maximum_landing_regions, "maximum_landing_regions"},
          {hopper.maximum_graph_nodes, "maximum_graph_nodes"},
          {hopper.maximum_graph_out_degree, "maximum_graph_out_degree"},
          {hopper.yaw_partition_count, "yaw_partition_count"},
          {hopper.maximum_nominal_aim_points_per_region,
           "maximum_nominal_aim_points_per_region"},
          {hopper.maximum_full_certification_attempts,
           "maximum_full_certification_attempts"},
          {hopper.maximum_root_iterations, "maximum_root_iterations"},
          {hopper.maximum_flight_tube_sections,
           "maximum_flight_tube_sections"},
          {hopper.landing_region_inflation_iterations,
           "landing_region_inflation_iterations"},
          {config.deterministic_execution.fixed_thread_count,
           "deterministic_execution.fixed_thread_count"},
      }};
  for (const auto& [value, name] : positive_hopper_caps) {
    if (value < 1U) {
      issues.Add(
          name.starts_with("deterministic")
              ? std::string(name)
              : ChildPath("hopper", name),
          "resource_cap_too_small",
          "resource cap must be at least one");
    }
  }
  if (hopper.support_direction_count < 4U) {
    issues.Add("hopper.support_direction_count",
               "resource_cap_too_small",
               "support direction count must be at least four");
  }
  if (hopper.landing_region_maximum_vertices < 3U ||
      hopper.landing_region_maximum_vertices > 128U) {
    issues.Add("hopper.landing_region_maximum_vertices",
               "resource_cap_out_of_range",
               "landing-region vertex cap must be in [3, 128]");
  }

  CheckNonNegative(
      config.learned_cost_policy.maximum_absolute_energy_correction,
      "learned_cost_policy.maximum_absolute_energy_correction", issues);
  CheckNonNegative(
      config.learned_cost_policy
          .maximum_absolute_nonfatal_risk_correction,
      "learned_cost_policy.maximum_absolute_nonfatal_risk_correction",
      issues);
  const bool learned_enabled =
      config.learned_cost_policy.mode ==
      LearnedCostPolicy::Mode::kOptionalBoundedSoftCost;
  if (learned_enabled && !config.learned_cost_model_ref.has_value()) {
    issues.Add("learned_cost_model_ref",
               "learned_cost_model_ref_required",
               "enabled learned soft cost needs a fixed model ref");
  }
  if (!learned_enabled && config.learned_cost_model_ref.has_value()) {
    issues.Add("learned_cost_model_ref",
               "disabled_learned_cost_forbids_model_ref",
               "disabled learned cost must not carry a model ref");
  }
  if (config.learned_cost_model_ref.has_value()) {
    CheckContentRef(*config.learned_cost_model_ref,
                    "learned_cost_model_ref", issues);
  }
  if (!config.deterministic_execution.stable_candidate_order) {
    issues.Add("deterministic_execution.stable_candidate_order",
               "stable_candidate_order_required",
               "deterministic execution requires stable ordering");
  }
  if (!config.deterministic_execution.preallocated_memory_pools) {
    issues.Add("deterministic_execution.preallocated_memory_pools",
               "preallocated_memory_pools_required",
               "deterministic execution requires preallocated pools");
  }
}

[[nodiscard]] PlatformType CapabilityPlatform(
    const SafetyCapabilityProfile& capability) {
  return std::visit(
      [](const auto& content) {
        using Capability = std::decay_t<decltype(content)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          return PlatformType::kWheeled;
        } else if constexpr (std::is_same_v<Capability,
                                            LeggedCapability>) {
          return PlatformType::kLegged;
        } else {
          return PlatformType::kHopper;
        }
      },
      capability.content);
}

[[nodiscard]] std::string_view CapabilityFrame(
    const SafetyCapabilityProfile& capability) {
  return std::visit(
      [](const auto& content) -> std::string_view {
        return content.frame_id;
      },
      capability.content);
}

[[nodiscard]] const ContentRef& CapabilityMotionModelRef(
    const SafetyCapabilityProfile& capability) {
  return std::visit(
      [](const auto& content) -> const ContentRef& {
        return content.motion_model_ref;
      },
      capability.content);
}

[[nodiscard]] const ContentRef& CapabilityAnalyticCostModelRef(
    const SafetyCapabilityProfile& capability) {
  return std::visit(
      [](const auto& content) -> const ContentRef& {
        return content.analytic_cost_model_ref;
      },
      capability.content);
}

template <class T>
void CheckResolvedBinding(const ResolvedBinding<T>& binding,
                          std::string_view path,
                          IssueCollector& issues) {
  CheckContentRef(binding.content_ref,
                  ChildPath(path, "content_ref"), issues);
  if (!binding.object) {
    issues.Add(ChildPath(path, "object"),
               "missing_resolved_capability_object",
               "capability dependency must be resolved before the call");
  }
}

void CheckGoal(const GoalRegion& goal, IssueCollector& issues) {
  if (goal.goal_id.empty()) {
    issues.Add("goal.goal_id", "identifier_required",
               "goal ID must not be empty");
  }
  std::visit(
      [&](const auto& target) {
        using Goal = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Goal, PointGoal>) {
          CheckFinite(target.position_m, "goal.target.position_m",
                      issues);
          CheckNonNegative(target.position_tolerance_m,
                           "goal.target.position_tolerance_m", issues);
        } else {
          CheckLandingPlane(target.plane, "goal.target.plane", issues);
          CheckConvexPolygon(target.polygon, "goal.target.polygon",
                             issues);
          CheckNonNegative(target.normal_tolerance_m,
                           "goal.target.normal_tolerance_m", issues);
        }
      },
      goal.target);
  if (goal.optional_yaw_interval.has_value()) {
    CheckCircularYawInterval(*goal.optional_yaw_interval,
                             "goal.optional_yaw_interval", issues);
  }
  if (goal.mission_direction_hint.has_value()) {
    CheckFinite(*goal.mission_direction_hint,
                "goal.mission_direction_hint", issues);
  }
  std::set<std::string> metadata_keys;
  for (std::size_t index = 0; index < goal.task_metadata.size();
       ++index) {
    const MetadataEntry& entry = goal.task_metadata[index];
    const std::string entry_path =
        IndexedPath("goal.task_metadata", index);
    if (entry.key.empty()) {
      issues.Add(ChildPath(entry_path, "key"), "metadata_key_required",
                 "metadata key must not be empty");
    } else if (!metadata_keys.insert(entry.key).second) {
      issues.Add(ChildPath(entry_path, "key"), "duplicate_metadata_key",
                 "goal metadata keys must be unique");
    }
    if (std::holds_alternative<double>(entry.value)) {
      CheckFinite(std::get<double>(entry.value),
                  ChildPath(entry_path, "value"), issues);
    }
  }
}

void CheckPreviousExecutionContext(
    const PreviousExecutionContext& context,
    PlatformType platform,
    IssueCollector& issues) {
  CheckContentRef(context.active_bundle_ref,
                  "previous_execution_context.active_bundle_ref", issues);
  if (context.active_bundle_handle.empty()) {
    issues.Add("previous_execution_context.active_bundle_handle",
               "registry_handle_required",
               "active bundle handle must not be empty");
  }
  CheckContentRef(
      context.source_map_snapshot_ref,
      "previous_execution_context.source_map_snapshot_ref", issues);
  CheckContentRef(
      context.source_capability_ref,
      "previous_execution_context.source_capability_ref", issues);

  const bool hopper = platform == PlatformType::kHopper;
  if (hopper !=
      std::holds_alternative<JumpCommitBoundary>(
          context.commit_boundary)) {
    issues.Add("previous_execution_context.commit_boundary",
               "platform_commit_boundary_mismatch",
               "commit-boundary variant must match the platform");
  }
  if (hopper !=
      std::holds_alternative<JumpExecutionCursor>(
          context.execution_cursor)) {
    issues.Add("previous_execution_context.execution_cursor",
               "platform_execution_cursor_mismatch",
               "execution-cursor variant must match the platform");
  }
  if (const auto* time_boundary =
          std::get_if<TimeCommitBoundary>(&context.commit_boundary);
      time_boundary != nullptr) {
    CheckDuration(
        time_boundary->committed_until_offset,
        "previous_execution_context.commit_boundary."
        "committed_until_offset",
        issues);
  }
  if (const auto* time_cursor =
          std::get_if<TimeExecutionCursor>(&context.execution_cursor);
      time_cursor != nullptr) {
    CheckDuration(time_cursor->offset,
                  "previous_execution_context.execution_cursor.offset",
                  issues);
  }
  if (const auto* jump_boundary =
          std::get_if<JumpCommitBoundary>(&context.commit_boundary);
      jump_boundary != nullptr && jump_boundary->boundary_id.empty()) {
    issues.Add(
        "previous_execution_context.commit_boundary.boundary_id",
        "boundary_id_required", "jump commit boundary needs an ID");
  }
  if (const auto* jump_cursor =
          std::get_if<JumpExecutionCursor>(&context.execution_cursor);
      jump_cursor != nullptr) {
    const bool boundary_required =
        jump_cursor->jump_state == JumpExecutionState::kJumpReady ||
        jump_cursor->jump_state == JumpExecutionState::kJumpCommitted ||
        jump_cursor->jump_state == JumpExecutionState::kInFlight;
    if (boundary_required &&
        (!jump_cursor->boundary_id.has_value() ||
         jump_cursor->boundary_id->empty())) {
      issues.Add(
          "previous_execution_context.execution_cursor.boundary_id",
          "boundary_id_required",
          "ready, committed and in-flight states need a boundary ID");
    }
  }
}

void CheckResolvedCapabilities(const PlanningRequest& request,
                               IssueCollector& issues) {
  if (!request.safety_capability) {
    return;
  }
  const SafetyCapabilityProfile& profile = *request.safety_capability;
  const ResolvedCapabilityBindings& bindings =
      request.capability_bindings;
  CheckResolvedBinding(bindings.motion_model,
                       "capability_bindings.motion_model", issues);
  CheckResolvedBinding(bindings.analytic_cost_model,
                       "capability_bindings.analytic_cost_model", issues);
  if (bindings.motion_model.content_ref !=
      CapabilityMotionModelRef(profile)) {
    issues.Add("capability_bindings.motion_model.content_ref",
               "motion_model_binding_mismatch",
               "resolved motion model must match the fixed capability ref");
  }
  if (bindings.analytic_cost_model.content_ref !=
      CapabilityAnalyticCostModelRef(profile)) {
    issues.Add(
        "capability_bindings.analytic_cost_model.content_ref",
        "analytic_cost_model_binding_mismatch",
        "resolved analytic cost model must match the fixed capability ref");
  }

  if (request.platform_type == PlatformType::kHopper) {
    const auto* capability =
        std::get_if<HopperCapability>(&profile.content);
    if (!bindings.gravity_model.has_value()) {
      issues.Add("capability_bindings.gravity_model",
                 "hopper_binding_required",
                 "Hopper requires a resolved gravity model");
    } else {
      CheckResolvedBinding(*bindings.gravity_model,
                           "capability_bindings.gravity_model", issues);
      if (capability != nullptr &&
          bindings.gravity_model->content_ref !=
              capability->gravity_model_ref) {
        issues.Add(
            "capability_bindings.gravity_model.content_ref",
            "gravity_model_binding_mismatch",
            "resolved gravity model must match the capability");
      }
    }
    if (!bindings.error_model.has_value()) {
      issues.Add("capability_bindings.error_model",
                 "hopper_binding_required",
                 "Hopper requires a resolved deterministic error model");
    } else {
      CheckResolvedBinding(*bindings.error_model,
                           "capability_bindings.error_model", issues);
    }
    if (!bindings.actuator_or_impulse_profile.has_value()) {
      issues.Add("capability_bindings.actuator_or_impulse_profile",
                 "hopper_binding_required",
                 "Hopper requires a resolved actuator/impulse profile");
    } else {
      CheckResolvedBinding(
          *bindings.actuator_or_impulse_profile,
          "capability_bindings.actuator_or_impulse_profile", issues);
    }
    if (!bindings.body_rotation_envelope.has_value()) {
      issues.Add("capability_bindings.body_rotation_envelope",
                 "hopper_binding_required",
                 "Hopper requires a resolved body-rotation envelope");
    } else {
      CheckResolvedBinding(
          *bindings.body_rotation_envelope,
          "capability_bindings.body_rotation_envelope", issues);
    }
    const bool table_expected =
        capability != nullptr &&
        capability->attitude_tightening_table_ref.has_value();
    if (table_expected !=
        bindings.attitude_tightening_table.has_value()) {
      issues.Add("capability_bindings.attitude_tightening_table",
                 "attitude_tightening_binding_mismatch",
                 "optional tightening-table binding must match capability");
    } else if (bindings.attitude_tightening_table.has_value()) {
      CheckResolvedBinding(
          *bindings.attitude_tightening_table,
          "capability_bindings.attitude_tightening_table", issues);
      if (capability != nullptr &&
          bindings.attitude_tightening_table->content_ref !=
              *capability->attitude_tightening_table_ref) {
        issues.Add(
            "capability_bindings.attitude_tightening_table.content_ref",
            "attitude_tightening_binding_mismatch",
            "tightening-table binding must match capability");
      }
    }
  } else {
    if (bindings.gravity_model.has_value() ||
        bindings.error_model.has_value() ||
        bindings.actuator_or_impulse_profile.has_value() ||
        bindings.body_rotation_envelope.has_value() ||
        bindings.attitude_tightening_table.has_value()) {
      issues.Add("capability_bindings",
                 "unexpected_hopper_capability_binding",
                 "ground platforms must not carry Hopper-only bindings");
    }
  }
}

void CheckLearnedCostBinding(const PlanningRequest& request,
                             IssueCollector& issues) {
  if (!request.algorithm_config) {
    return;
  }
  const PlannerAlgorithmConfig& config = *request.algorithm_config;
  const bool enabled =
      config.learned_cost_policy.mode ==
      LearnedCostPolicy::Mode::kOptionalBoundedSoftCost;
  if (!enabled && request.learned_cost_snapshot.has_value()) {
    issues.Add("learned_cost_snapshot",
               "disabled_learned_cost_forbids_snapshot",
               "disabled learned cost must not bind a snapshot");
  }
  if (request.learned_cost_snapshot.has_value()) {
    const LearnedCostSnapshotBinding& binding =
        *request.learned_cost_snapshot;
    CheckContentRef(binding.snapshot_ref,
                    "learned_cost_snapshot.snapshot_ref", issues);
    if (binding.registry_handle.empty()) {
      issues.Add("learned_cost_snapshot.registry_handle",
                 "registry_handle_required",
                 "learned-cost registry handle must not be empty");
    }
    if (!binding.resolved_snapshot) {
      issues.Add("learned_cost_snapshot.resolved_snapshot",
                 "missing_resolved_learned_snapshot",
                 "learned-cost snapshot must be resolved before the call");
    }
  }
}

[[nodiscard]] std::string_view LayerKindFieldName(
    const LayerKind kind) {
  switch (kind) {
    case LayerKind::kKnownMask:
      return "KNOWN_MASK";
    case LayerKind::kElevation:
      return "ELEVATION";
    case LayerKind::kTerrainNormal:
      return "TERRAIN_NORMAL";
    case LayerKind::kRoughness:
      return "ROUGHNESS";
    case LayerKind::kHardObstacle:
      return "HARD_OBSTACLE";
    case LayerKind::kConfidence:
      return "CONFIDENCE";
    case LayerKind::kEsdf:
      return "ESDF";
    case LayerKind::kStaticSpeedLimit:
      return "STATIC_SPEED_LIMIT";
  }
  return "UNKNOWN_LAYER_KIND";
}

void CheckMapSnapshot(const ImmutableMapSnapshot& snapshot,
                      IssueCollector& issues) {
  CheckContentRef(snapshot.snapshot_ref(),
                  "map_snapshot.snapshot_ref", issues);
  if (snapshot.map_revision() == 0U ||
      snapshot.map_revision() > kMaximumRevision) {
    issues.Add("map_snapshot.map_revision", "revision_out_of_range",
               "map revision must be in [1, 2147483647]");
  }
  if (snapshot.immutable_data_handle().empty()) {
    issues.Add("map_snapshot.immutable_data_handle",
               "registry_handle_required",
               "map snapshot needs an opaque registry handle");
  }
  if (snapshot.frame_id().empty()) {
    issues.Add("map_snapshot.frame_id", "frame_id_required",
               "map snapshot frame ID must not be empty");
  }
  CheckClockStamp(snapshot.source_time(), "map_snapshot.source_time",
                  issues);
  CheckFinite(snapshot.bounds().minimum_m,
              "map_snapshot.bounds.minimum_m", issues);
  CheckFinite(snapshot.bounds().maximum_m,
              "map_snapshot.bounds.maximum_m", issues);
  if (snapshot.bounds().minimum_m.x >
          snapshot.bounds().maximum_m.x ||
      snapshot.bounds().minimum_m.y >
          snapshot.bounds().maximum_m.y ||
      snapshot.bounds().minimum_m.z >
          snapshot.bounds().maximum_m.z) {
    issues.Add("map_snapshot.bounds", "map_bounds_reversed",
               "map minimum bounds must not exceed maximum bounds");
  }
  CheckPositive(snapshot.geometry().resolution_m,
                "map_snapshot.geometry.resolution_m", issues);
  CheckFinite(snapshot.geometry().origin_m,
              "map_snapshot.geometry.origin_m", issues);
  if (snapshot.geometry().width == 0U ||
      snapshot.geometry().height == 0U ||
      snapshot.geometry().CellCount() == 0U) {
    issues.Add("map_snapshot.geometry", "invalid_grid_geometry",
               "map grid dimensions must be positive and non-overflowing");
  }
  if (snapshot.geometry().frame_id != snapshot.frame_id()) {
    issues.Add("map_snapshot.geometry.frame_id",
               "map_internal_frame_mismatch",
               "map geometry and snapshot must use one frame");
  }

  std::set<LayerKind> kinds;
  std::set<std::pair<Identifier, std::uint32_t>> revisions;
  const auto manifest = snapshot.layer_manifest();
  for (std::size_t index = 0; index < manifest.size(); ++index) {
    const LayerManifestEntry& entry = manifest[index];
    const std::string entry_path =
        IndexedPath("map_snapshot.layer_manifest", index);
    CheckContentRef(entry.content_ref,
                    ChildPath(entry_path, "content_ref"), issues);
    if (!kinds.insert(entry.layer_kind).second) {
      issues.Add(ChildPath(entry_path, "layer_kind"),
                 "duplicate_map_layer_kind",
                 "map layer kinds must be unique");
    }
    if (!revisions
             .insert(
                 {entry.content_ref.id, entry.content_ref.revision})
             .second) {
      issues.Add(ChildPath(entry_path, "content_ref"),
                 "duplicate_map_layer_identity",
                 "map layer id/revision identities must be unique");
    }
  }

  const auto known = snapshot.KnownMask();
  const auto confidence = snapshot.Confidence();
  const auto esdf = snapshot.EsdfMeters();
  const auto static_speed = snapshot.StaticSpeedLimitMps();
  for (std::size_t index = 0; index < known.size(); ++index) {
    if (known[index] != 0U) {
      continue;
    }
    if (index < confidence.size() && confidence[index] > 0.0F) {
      issues.Add(IndexedPath("map_snapshot.confidence", index),
                 "unknown_cell_has_hard_feasible_hint",
                 "unknown cells cannot carry positive confidence as a "
                 "hard-feasibility hint");
    }
    if (index < esdf.size() && esdf[index] > 0.0F) {
      issues.Add(IndexedPath("map_snapshot.esdf_m", index),
                 "unknown_cell_has_hard_feasible_hint",
                 "unknown cells cannot carry positive clearance as a "
                 "hard-feasibility hint");
    }
    if (index < static_speed.size() && static_speed[index] > 0.0F) {
      issues.Add(
          IndexedPath("map_snapshot.static_speed_limit_mps", index),
          "unknown_cell_has_hard_feasible_hint",
          "unknown cells cannot carry a positive executable speed hint");
    }
  }
}

void CheckPlanningRequest(const PlanningRequest& request,
                          IssueCollector& issues) {
  if (request.request_id.empty()) {
    issues.Add("request_id", "identifier_required",
               "request ID must not be empty");
  }
  if (request.frame_id.empty()) {
    issues.Add("frame_id", "frame_id_required",
               "request frame ID must not be empty");
  }
  CheckClockStamp(request.request_time, "request_time", issues);
  CheckClockStamp(request.state_time, "state_time", issues);
  if (!request.request_time.clock_id.empty() &&
      !request.state_time.clock_id.empty() &&
      request.request_time.clock_id != request.state_time.clock_id) {
    issues.Add("state_time.clock_id", "clock_id_mismatch",
               "request and state times must use the same clock");
  }

  const bool state_matches =
      (request.platform_type == PlatformType::kHopper) ==
      std::holds_alternative<HopperState>(request.current_state);
  if (!state_matches) {
    issues.Add("current_state", "platform_state_mismatch",
               "current-state variant must match platform_type");
  }
  std::visit(
      [&](const auto& state) {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, WheeledOrLeggedState>) {
          CheckWheeledOrLeggedState(state, "current_state", issues);
        } else {
          CheckHopperState(state, "current_state", issues);
        }
      },
      request.current_state);
  CheckGoal(request.goal, issues);

  if (!request.map_snapshot) {
    issues.Add("map_snapshot", "missing_registry_object",
               "request map snapshot must be resolved and immutable");
  } else {
    CheckMapSnapshot(*request.map_snapshot, issues);
    if (request.map_snapshot->frame_id() != request.frame_id) {
      issues.Add("map_snapshot.frame_id", "map_frame_mismatch",
                 "map frame must match request frame");
    }
    const ClockStamp map_time = request.map_snapshot->source_time();
    if (!request.state_time.clock_id.empty() &&
        !map_time.clock_id.empty() &&
        request.state_time.clock_id != map_time.clock_id) {
      issues.Add("map_snapshot.source_time.clock_id",
                 "clock_id_mismatch",
                 "map source and state times must use one clock");
    }
  }
  if (!request.safety_capability) {
    issues.Add("safety_capability", "missing_registry_object",
               "safety capability must be resolved before the call");
  } else {
    IssueCollector nested;
    CheckCapabilityProfile(*request.safety_capability, nested);
    issues.Merge(nested.Finish(), "safety_capability");
    if (CapabilityPlatform(*request.safety_capability) !=
        request.platform_type) {
      issues.Add("safety_capability.content",
                 "platform_capability_mismatch",
                 "safety-capability variant must match platform_type");
    }
    if (CapabilityFrame(*request.safety_capability) !=
        request.frame_id) {
      issues.Add("safety_capability.content.frame_id",
                 "capability_frame_mismatch",
                 "capability frame must match request frame");
    }
  }
  if (!request.algorithm_config) {
    issues.Add("algorithm_config", "missing_registry_object",
               "algorithm config must be resolved before the call");
  } else {
    IssueCollector nested;
    CheckPlannerConfig(*request.algorithm_config, nested);
    issues.Merge(nested.Finish(), "algorithm_config");
    if (request.request_time.clock_id ==
        request.state_time.clock_id) {
      const long double skew =
          std::abs(static_cast<long double>(
                       request.request_time.tick.count()) -
                   static_cast<long double>(
                       request.state_time.tick.count()));
      if (skew > static_cast<long double>(
                     request.algorithm_config->max_input_skew.value
                         .count())) {
        issues.Add("state_time", "input_time_skew_exceeded",
                   "request and state time exceed configured skew");
      }
    }
    if (request.map_snapshot &&
        request.map_snapshot->source_time().clock_id ==
            request.state_time.clock_id) {
      const long double map_skew =
          std::abs(
              static_cast<long double>(
                  request.map_snapshot->source_time().tick.count()) -
              static_cast<long double>(
                  request.state_time.tick.count()));
      if (map_skew >
          static_cast<long double>(
              request.algorithm_config->max_input_skew.value.count())) {
        issues.Add("map_snapshot.source_time",
                   "input_time_skew_exceeded",
                   "map source and state time exceed configured skew");
      }
    }
  }

  CheckResolvedCapabilities(request, issues);
  CheckLearnedCostBinding(request, issues);
  if (request.previous_execution_context.has_value()) {
    CheckPreviousExecutionContext(
        *request.previous_execution_context, request.platform_type,
        issues);
  }
}

void CheckSafeStopAnchor(const SafeStopAnchor& anchor,
                         std::string_view path,
                         IssueCollector& issues) {
  if (anchor.anchor_id.empty()) {
    issues.Add(ChildPath(path, "anchor_id"), "identifier_required",
               "safe-stop anchor ID must not be empty");
  }
  CheckPose(anchor.pose, ChildPath(path, "pose"), issues);
  CheckFinite(anchor.target_linear_velocity_mps,
              ChildPath(path, "target_linear_velocity_mps"), issues);
  CheckFinite(anchor.target_yaw_rate_radps,
              ChildPath(path, "target_yaw_rate_radps"), issues);
  if (anchor.target_linear_velocity_mps != 0.0 ||
      anchor.target_yaw_rate_radps != 0.0 ||
      std::signbit(anchor.target_linear_velocity_mps) ||
      std::signbit(anchor.target_yaw_rate_radps)) {
    issues.Add(std::string(path), "stationary_anchor_required",
               "safe-stop anchor velocities must be canonical zero");
  }
  CheckContentRef(anchor.terrain_certification_ref,
                  ChildPath(path, "terrain_certification_ref"), issues);
}

void CheckWheeledReference(const WheeledReference& reference,
                           std::string_view path,
                           IssueCollector& issues) {
  if (reference.reference_id.empty()) {
    issues.Add(ChildPath(path, "reference_id"), "identifier_required",
               "reference ID must not be empty");
  }
  CheckHash(reference.reference_hash,
            ChildPath(path, "reference_hash"), issues);
  CheckClockStamp(reference.reference_time_origin,
                  ChildPath(path, "reference_time_origin"), issues);
  if (reference.segments.empty()) {
    issues.Add(ChildPath(path, "segments"), "reference_segment_required",
               "wheeled reference must contain a segment");
  }
  std::set<SegmentId> segment_ids;
  std::optional<std::chrono::nanoseconds> previous_end;
  for (std::size_t index = 0; index < reference.segments.size();
       ++index) {
    const std::string segment_path =
        IndexedPath(ChildPath(path, "segments"), index);
    std::visit(
        [&](const auto& segment) {
          using Segment = std::decay_t<decltype(segment)>;
          if (segment.segment_id.empty()) {
            issues.Add(ChildPath(segment_path, "segment_id"),
                       "identifier_required",
                       "segment ID must not be empty");
          } else if (!segment_ids.insert(segment.segment_id).second) {
            issues.Add(ChildPath(segment_path, "segment_id"),
                       "duplicate_segment_id",
                       "segment IDs must be unique");
          }
          CheckTimeInterval(segment.time_interval,
                            ChildPath(segment_path, "time_interval"),
                            issues, true);
          if (previous_end.has_value() &&
              segment.time_interval.start_offset.value !=
                  *previous_end) {
            issues.Add(ChildPath(segment_path, "time_interval"),
                       "reference_time_discontinuity",
                       "wheeled segment intervals must be contiguous");
          }
          previous_end = segment.time_interval.end_offset.value;
          if constexpr (std::is_same_v<Segment, DriveSegment>) {
            CheckGeometricPath(
                segment.geometric_path,
                ChildPath(segment_path, "geometric_path"), issues);
            CheckTimeScaling(
                segment.time_scaling,
                ChildPath(segment_path, "time_scaling"), issues);
            if (!segment.time_scaling.segments.empty() &&
                (segment.time_scaling.segments.front()
                         .start_offset.value !=
                     segment.time_interval.start_offset.value ||
                 segment.time_scaling.segments.back().end_offset.value !=
                     segment.time_interval.end_offset.value)) {
              issues.Add(ChildPath(segment_path, "time_scaling"),
                         "time_scaling_interval_mismatch",
                         "time scaling must span the drive interval");
            }
            if (segment.derived_caches.has_value()) {
              const DerivedKinematicCaches& caches =
                  *segment.derived_caches;
              CheckNonNegative(
                  caches.consistency_tolerance,
                  ChildPath(segment_path,
                            "derived_caches.consistency_tolerance"),
                  issues);
              CheckPiecewiseTrajectory(
                  caches.signed_body_forward_speed_mps,
                  ChildPath(
                      segment_path,
                      "derived_caches.signed_body_forward_speed_mps"),
                  issues);
              CheckPiecewiseTrajectory(
                  caches.yaw_rate_radps,
                  ChildPath(segment_path,
                            "derived_caches.yaw_rate_radps"),
                  issues);
            }
          } else {
            CheckFinite(segment.fixed_position_m,
                        ChildPath(segment_path, "fixed_position_m"),
                        issues);
            CheckPiecewiseTrajectory(
                segment.unwrapped_yaw_rad,
                ChildPath(segment_path, "unwrapped_yaw_rad"), issues);
            if (!segment.unwrapped_yaw_rad.segments.empty() &&
                (segment.unwrapped_yaw_rad.segments.front()
                         .start_offset.value !=
                     segment.time_interval.start_offset.value ||
                 segment.unwrapped_yaw_rad.segments.back()
                         .end_offset.value !=
                     segment.time_interval.end_offset.value)) {
              issues.Add(
                  ChildPath(segment_path, "unwrapped_yaw_rad"),
                  "yaw_trajectory_interval_mismatch",
                  "spin yaw trajectory must span its segment interval");
            }
          }
        },
        reference.segments[index]);
  }
  CheckSafeStopAnchor(reference.safe_stop_anchor,
                      ChildPath(path, "safe_stop_anchor"), issues);
}

void CheckLeggedReference(const LeggedBodyReference& reference,
                          std::string_view path,
                          IssueCollector& issues) {
  if (reference.reference_id.empty()) {
    issues.Add(ChildPath(path, "reference_id"), "identifier_required",
               "reference ID must not be empty");
  }
  CheckHash(reference.reference_hash,
            ChildPath(path, "reference_hash"), issues);
  if (reference.reference_point_id.empty()) {
    issues.Add(ChildPath(path, "reference_point_id"),
               "reference_point_id_required",
               "legged reference needs a fixed reference-point ID");
  }
  CheckClockStamp(reference.reference_time_origin,
                  ChildPath(path, "reference_time_origin"), issues);
  CheckGeometricPath(reference.geometric_path,
                     ChildPath(path, "geometric_path"), issues);
  CheckTimeScaling(reference.time_scaling,
                   ChildPath(path, "time_scaling"), issues);
  CheckInterval(
      reference.body_frame_velocity_envelope.forward_mps,
      ChildPath(path, "body_frame_velocity_envelope.forward_mps"),
      issues);
  CheckInterval(
      reference.body_frame_velocity_envelope.lateral_mps,
      ChildPath(path, "body_frame_velocity_envelope.lateral_mps"),
      issues);
  CheckInterval(
      reference.body_frame_velocity_envelope.vertical_mps,
      ChildPath(path, "body_frame_velocity_envelope.vertical_mps"),
      issues);
  CheckInterval(
      reference.body_frame_velocity_envelope.yaw_rate_radps,
      ChildPath(path, "body_frame_velocity_envelope.yaw_rate_radps"),
      issues);
  CheckFinite(
      reference.terrain_normal_envelope.maximum_normal_deviation_rad,
      ChildPath(
          path,
          "terrain_normal_envelope.maximum_normal_deviation_rad"),
      issues);
  if (std::isfinite(
          reference.terrain_normal_envelope.maximum_normal_deviation_rad) &&
      (reference.terrain_normal_envelope.maximum_normal_deviation_rad <
           0.0 ||
       reference.terrain_normal_envelope.maximum_normal_deviation_rad >
           kPi / 2.0)) {
    issues.Add(
        ChildPath(
            path,
            "terrain_normal_envelope.maximum_normal_deviation_rad"),
        "normal_deviation_out_of_range",
        "terrain-normal deviation must be in [0, pi/2]");
  }
  CheckContentRef(
      reference.terrain_normal_envelope
          .source_terrain_certification_ref,
      ChildPath(
          path,
          "terrain_normal_envelope.source_terrain_certification_ref"),
      issues);
  CheckInterval(
      reference.roll_pitch_diagnostic_envelope.roll_rad,
      ChildPath(path, "roll_pitch_diagnostic_envelope.roll_rad"),
      issues);
  CheckInterval(
      reference.roll_pitch_diagnostic_envelope.pitch_rad,
      ChildPath(path, "roll_pitch_diagnostic_envelope.pitch_rad"),
      issues);
  CheckSafeStopAnchor(reference.safe_stop_anchor,
                      ChildPath(path, "safe_stop_anchor"), issues);
  if (reference.feasibility_scope !=
      "body_geometry_and_terrain_thresholds_only") {
    issues.Add(ChildPath(path, "feasibility_scope"),
               "invalid_feasibility_scope",
               "legged feasibility scope is frozen by v3");
  }
  if (reference.footstep_feasibility_guaranteed) {
    issues.Add(ChildPath(path, "footstep_feasibility_guaranteed"),
               "footstep_guarantee_forbidden",
               "body reference does not guarantee footstep feasibility");
  }
}

void CheckHopperKinematicState(const HopperKinematicState& state,
                               std::string_view path,
                               IssueCollector& issues) {
  CheckFinite(state.position_m, ChildPath(path, "position_m"), issues);
  CheckQuaternion(state.orientation_body_to_frame,
                  ChildPath(path, "orientation_body_to_frame"), issues);
  CheckFinite(state.linear_velocity_mps,
              ChildPath(path, "linear_velocity_mps"), issues);
  CheckFinite(state.angular_velocity_radps,
              ChildPath(path, "angular_velocity_radps"), issues);
}

[[nodiscard]] double MaximumSetVectorNorm(
    const DeterministicVectorSet3& set) {
  return std::visit(
      [](const auto& value) {
        using Set = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Set, AxisAlignedBox3>) {
          return Norm(Vec3{
              std::abs(value.center.x) + value.half_extent.x,
              std::abs(value.center.y) + value.half_extent.y,
              std::abs(value.center.z) + value.half_extent.z,
          });
        } else {
          return Norm(value.center) + value.radius;
        }
      },
      set);
}

[[nodiscard]] double MaximumBoundsNorm(const Vector3Bounds& bounds) {
  return Norm(Vec3{
      std::max(std::abs(bounds.lower.x), std::abs(bounds.upper.x)),
      std::max(std::abs(bounds.lower.y), std::abs(bounds.upper.y)),
      std::max(std::abs(bounds.lower.z), std::abs(bounds.upper.z)),
  });
}

void CheckHopperReference(const HopperReference& reference,
                          std::string_view path,
                          IssueCollector& issues) {
  if (reference.reference_id.empty()) {
    issues.Add(ChildPath(path, "reference_id"), "identifier_required",
               "reference ID must not be empty");
  }
  CheckHash(reference.reference_hash,
            ChildPath(path, "reference_hash"), issues);
  CheckClockStamp(reference.reference_time_origin,
                  ChildPath(path, "reference_time_origin"), issues);

  const GroundHoldAnchor& hold = reference.ground_hold_anchor;
  const std::string hold_path =
      ChildPath(path, "ground_hold_anchor");
  if (hold.anchor_id.empty()) {
    issues.Add(ChildPath(hold_path, "anchor_id"), "identifier_required",
               "ground-hold anchor ID must not be empty");
  }
  CheckHopperKinematicState(hold.hold_state,
                            ChildPath(hold_path, "hold_state"), issues);
  CheckHopperError(
      hold.allowed_hold_state_error_set,
      ChildPath(hold_path, "allowed_hold_state_error_set"), issues);
  CheckContentRef(hold.terrain_certification_ref,
                  ChildPath(hold_path, "terrain_certification_ref"),
                  issues);
  const Vec3& hold_linear = hold.hold_state.linear_velocity_mps;
  const Vec3& hold_angular = hold.hold_state.angular_velocity_radps;
  if (hold_linear.x != 0.0 || hold_linear.y != 0.0 ||
      hold_linear.z != 0.0 || hold_angular.x != 0.0 ||
      hold_angular.y != 0.0 || hold_angular.z != 0.0 ||
      std::signbit(hold_linear.x) || std::signbit(hold_linear.y) ||
      std::signbit(hold_linear.z) || std::signbit(hold_angular.x) ||
      std::signbit(hold_angular.y) || std::signbit(hold_angular.z)) {
    issues.Add(ChildPath(hold_path, "hold_state"),
               "ground_hold_not_stationary",
               "ground-hold linear and angular velocity must be zero");
  }

  const NextLandingRegion& region =
      reference.next_landing_region;
  const std::string region_path =
      ChildPath(path, "next_landing_region");
  if (region.region_id.empty()) {
    issues.Add(ChildPath(region_path, "region_id"),
               "identifier_required",
               "landing-region ID must not be empty");
  }
  if (region.frame_id.empty()) {
    issues.Add(ChildPath(region_path, "frame_id"),
               "frame_id_required",
               "landing-region frame must not be empty");
  }
  CheckLandingPlane(region.landing_plane,
                    ChildPath(region_path, "landing_plane"), issues);
  CheckConvexPolygon(region.convex_polygon,
                     ChildPath(region_path, "convex_polygon"), issues);
  CheckCircularYawInterval(
      region.allowed_yaw_interval,
      ChildPath(region_path, "allowed_yaw_interval"), issues);
  CheckContentRef(region.terrain_certification_ref,
                  ChildPath(region_path, "terrain_certification_ref"),
                  issues);
  CheckNonNegative(
      region.inward_safety_margin_m,
      ChildPath(region_path, "inward_safety_margin_m"), issues);

  const JumpBoundary& boundary = reference.jump_boundary;
  const std::string boundary_path =
      ChildPath(path, "jump_boundary");
  if (boundary.boundary_id.empty()) {
    issues.Add(ChildPath(boundary_path, "boundary_id"),
               "identifier_required",
               "jump-boundary ID must not be empty");
  }
  CheckHopperKinematicState(
      boundary.nominal_launch_state,
      ChildPath(boundary_path, "nominal_launch_state"), issues);
  CheckHopperError(
      boundary.allowed_launch_state_error_set,
      ChildPath(boundary_path, "allowed_launch_state_error_set"),
      issues);
  CheckContentRef(boundary.gravity_model_ref,
                  ChildPath(boundary_path, "gravity_model_ref"), issues);
  CheckContentRef(
      boundary.actuator_or_impulse_profile_ref,
      ChildPath(boundary_path, "actuator_or_impulse_profile_ref"),
      issues);
  CheckDuration(
      boundary.ballistic_flight_time,
      ChildPath(boundary_path, "ballistic_flight_time"), issues);
  if (boundary.ballistic_flight_time.value.count() <= 0) {
    issues.Add(ChildPath(boundary_path, "ballistic_flight_time"),
               "positive_flight_time_required",
               "ballistic flight time must be strictly positive");
  }
  if (Norm(Subtract(
          hold.hold_state.position_m,
          boundary.nominal_launch_state.position_m)) >
          kGeometryTolerance ||
      QuaternionAngularDistance(
          hold.hold_state.orientation_body_to_frame,
          boundary.nominal_launch_state.orientation_body_to_frame) >
          kGeometryTolerance) {
    issues.Add(ChildPath(boundary_path, "nominal_launch_state"),
               "ground_hold_launch_discontinuity",
               "ground hold and nominal launch position/orientation "
               "must be continuous");
  }

  const PredictedLandingFootprint& footprint =
      reference.predicted_landing_footprint;
  const std::string footprint_path =
      ChildPath(path, "predicted_landing_footprint");
  CheckLandingPlane(footprint.landing_plane,
                    ChildPath(footprint_path, "landing_plane"), issues);
  CheckConvexPolygon(
      footprint.convex_center_landing_polygon,
      ChildPath(footprint_path, "convex_center_landing_polygon"),
      issues);
  CheckTimeInterval(footprint.landing_time_window,
                    ChildPath(footprint_path, "landing_time_window"),
                    issues);
  CheckVectorBounds(
      footprint.landing_velocity_bounds,
      ChildPath(footprint_path, "landing_velocity_bounds"), issues);
  CheckCircularYawInterval(
      footprint.landing_yaw_interval,
      ChildPath(footprint_path, "landing_yaw_interval"), issues);
  CheckContentRef(
      footprint.source_error_model_ref,
      ChildPath(footprint_path, "source_error_model_ref"), issues);
  CheckNonNegative(
      footprint.outer_approximation_margin_m,
      ChildPath(footprint_path, "outer_approximation_margin_m"),
      issues);
  if (!SamePlane(footprint.landing_plane, region.landing_plane)) {
    issues.Add(ChildPath(footprint_path, "landing_plane"),
               "landing_plane_mismatch",
               "predicted footprint and landing region must share "
               "one plane and basis");
  }
  const double total_margin =
      footprint.outer_approximation_margin_m +
      region.inward_safety_margin_m;
  if (std::isfinite(total_margin) &&
      !PolygonWithMarginContained(
          footprint.convex_center_landing_polygon,
          footprint.landing_plane, region.convex_polygon,
          region.landing_plane, total_margin)) {
    issues.Add(
        ChildPath(footprint_path, "convex_center_landing_polygon"),
        "landing_footprint_not_contained",
        "margin-expanded footprint must be contained in landing region");
  }
  const auto flight_time = boundary.ballistic_flight_time.value;
  if (footprint.landing_time_window.start_offset.value > flight_time ||
      footprint.landing_time_window.end_offset.value < flight_time) {
    issues.Add(ChildPath(footprint_path, "landing_time_window"),
               "landing_time_window_excludes_nominal_impact",
               "landing time window must contain nominal impact time");
  }

  const CertifiedFlightTube& tube =
      reference.certified_flight_tube;
  const std::string tube_path =
      ChildPath(path, "certified_flight_tube");
  if (tube.frame_id.empty()) {
    issues.Add(ChildPath(tube_path, "frame_id"), "frame_id_required",
               "flight-tube frame must not be empty");
  }
  if (tube.frame_id != region.frame_id) {
    issues.Add(ChildPath(tube_path, "frame_id"),
               "flight_tube_frame_mismatch",
               "flight tube and landing region must use one frame");
  }
  CheckContentRef(tube.source_map_snapshot_ref,
                  ChildPath(tube_path, "source_map_snapshot_ref"),
                  issues);
  CheckContentRef(tube.body_rotation_envelope_ref,
                  ChildPath(tube_path, "body_rotation_envelope_ref"),
                  issues);
  CheckContentRef(tube.error_model_ref,
                  ChildPath(tube_path, "error_model_ref"), issues);
  CheckNonNegative(
      tube.minimum_certified_clearance_m,
      ChildPath(tube_path, "minimum_certified_clearance_m"), issues);
  if (tube.sections.empty()) {
    issues.Add(ChildPath(tube_path, "sections"),
               "flight_tube_section_required",
               "certified flight tube must contain a section");
  }
  std::chrono::nanoseconds covered_until{0};
  std::chrono::nanoseconds previous_start{0};
  bool first_section = true;
  for (std::size_t index = 0; index < tube.sections.size(); ++index) {
    const FlightTubeSection& section = tube.sections[index];
    const std::string section_path =
        IndexedPath(ChildPath(tube_path, "sections"), index);
    CheckTimeInterval(section.time_interval,
                      ChildPath(section_path, "time_interval"), issues,
                      true);
    CheckHalfspacePolytope(
        section.envelope, ChildPath(section_path, "envelope"), issues);
    const auto start = section.time_interval.start_offset.value;
    const auto end = section.time_interval.end_offset.value;
    if (first_section) {
      if (start != std::chrono::nanoseconds{0}) {
        issues.Add(ChildPath(section_path, "time_interval"),
                   "flight_tube_does_not_start_at_launch",
                   "first flight-tube section must begin at launch time");
      }
      first_section = false;
    } else {
      if (start < previous_start) {
        issues.Add(ChildPath(section_path, "time_interval"),
                   "flight_tube_sections_unordered",
                   "flight-tube section starts must be ordered");
      }
      if (start > covered_until) {
        issues.Add(ChildPath(section_path, "time_interval"),
                   "flight_tube_coverage_gap",
                   "flight-tube sections must cover without gaps");
      }
    }
    previous_start = start;
    covered_until = std::max(covered_until, end);
  }
  if (covered_until < flight_time) {
    issues.Add(ChildPath(tube_path, "sections"),
               "flight_tube_incomplete_coverage",
               "flight tube must continuously cover [0, flight_time]");
  }

  const AttitudeBoundary& attitude = reference.attitude_boundary;
  const std::string attitude_path =
      ChildPath(path, "attitude_boundary");
  CheckRotationVectorBall(
      attitude.initial_orientation_error_set,
      ChildPath(attitude_path, "initial_orientation_error_set"),
      issues);
  CheckDeterministicSet(
      attitude.initial_angular_velocity_error_set_radps,
      ChildPath(attitude_path,
                "initial_angular_velocity_error_set_radps"),
      issues);
  CheckQuaternion(
      attitude.target_attitude_set.nominal_orientation_body_to_frame,
      ChildPath(attitude_path,
                "target_attitude_set."
                "nominal_orientation_body_to_frame"),
      issues);
  CheckRotationVectorBall(
      attitude.target_attitude_set.orientation_error_set,
      ChildPath(attitude_path,
                "target_attitude_set.orientation_error_set"),
      issues);
  CheckCircularYawInterval(
      attitude.target_attitude_set.allowed_yaw_interval,
      ChildPath(attitude_path,
                "target_attitude_set.allowed_yaw_interval"),
      issues);
  CheckDeterministicSet(
      attitude.landing_angular_velocity_bounds_radps,
      ChildPath(attitude_path,
                "landing_angular_velocity_bounds_radps"),
      issues);
  CheckDuration(attitude.settle_guard,
                ChildPath(attitude_path, "settle_guard"), issues);
  CheckContentRef(attitude.certification_ref,
                  ChildPath(attitude_path, "certification_ref"), issues);
  if (!IsYawSubset(
          footprint.landing_yaw_interval,
          attitude.target_attitude_set.allowed_yaw_interval)) {
    issues.Add(ChildPath(footprint_path, "landing_yaw_interval"),
               "landing_yaw_not_subset_of_target_attitude",
               "footprint yaw must be a subset of target-attitude yaw");
  }
  if (!IsYawSubset(
          attitude.target_attitude_set.allowed_yaw_interval,
          region.allowed_yaw_interval)) {
    issues.Add(ChildPath(
                   attitude_path,
                   "target_attitude_set.allowed_yaw_interval"),
               "target_attitude_yaw_not_subset_of_landing_region",
               "target-attitude yaw must be a subset of region yaw");
  }

  CheckFinite(reference.nominal_aim_point.position_m,
              ChildPath(path, "nominal_aim_point.position_m"), issues);
  CheckContentRef(reference.physical_certification_ref,
                  ChildPath(path, "physical_certification_ref"), issues);
  if (reference.future_route_preview.has_value()) {
    const FutureRoutePreview& preview =
        *reference.future_route_preview;
    if (preview.reason_code.empty()) {
      issues.Add(ChildPath(path, "future_route_preview.reason_code"),
                 "reason_code_required",
                 "future-route preview needs a reason code");
    }
  }
}

[[nodiscard]] PlatformType ReferencePlatform(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& value) {
        using Reference = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Reference, WheeledReference>) {
          return PlatformType::kWheeled;
        } else if constexpr (std::is_same_v<Reference,
                                            LeggedBodyReference>) {
          return PlatformType::kLegged;
        } else {
          return PlatformType::kHopper;
        }
      },
      reference);
}

[[nodiscard]] std::string_view ReferenceIdOf(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& value) -> std::string_view {
        return value.reference_id;
      },
      reference);
}

[[nodiscard]] std::string_view ReferenceHashOf(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& value) -> std::string_view {
        return value.reference_hash;
      },
      reference);
}

[[nodiscard]] const ClockStamp& ReferenceTimeOf(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& value) -> const ClockStamp& {
        return value.reference_time_origin;
      },
      reference);
}

void CheckPlatformReference(const PlatformReference& reference,
                            std::string_view path,
                            IssueCollector& issues) {
  std::visit(
      [&](const auto& value) {
        using Reference = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Reference, WheeledReference>) {
          CheckWheeledReference(value, path, issues);
        } else if constexpr (std::is_same_v<Reference,
                                            LeggedBodyReference>) {
          CheckLeggedReference(value, path, issues);
        } else {
          CheckHopperReference(value, path, issues);
        }
      },
      reference);
}

[[nodiscard]] std::optional<nlohmann::json> EncodeBundleJson(
    const ReferenceBundle& bundle,
    IssueCollector& issues) {
  try {
    PlanningResponse carrier{};
    carrier.request_id = bundle.source_request_id.empty()
                             ? "hash-carrier-request"
                             : bundle.source_request_id;
    carrier.response_time = ReferenceTimeOf(bundle.platform_reference);
    carrier.planning_outcome = PlanningOutcome::kNewReferenceReady;
    carrier.execution_directive =
        ExecutionDirective::kActivateNewBundle;
    carrier.reason_code = "HASH_CARRIER";
    carrier.new_reference_bundle = bundle;
    return nlohmann::json::parse(
        JsonCodec::EncodePlanningResponse(carrier));
  } catch (const std::exception& error) {
    issues.Add("bundle_hash", "hash_input_unencodable",
               std::string("typed bundle could not be encoded for JCS: ") +
                   error.what());
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<Sha256Digest> HashJson(
    const nlohmann::json& value,
    std::string_view path,
    IssueCollector& issues) {
  const Result<std::string> canonical =
      JcsCanonicalizer::Canonicalize(value);
  if (!IsOk(canonical)) {
    const Error& error = std::get<Error>(canonical);
    issues.Add(std::string(path), "jcs_canonicalization_failed",
               error.message);
    return std::nullopt;
  }
  const Result<Sha256Digest> digest =
      Sha256Hex(std::get<std::string>(canonical));
  if (!IsOk(digest)) {
    const Error& error = std::get<Error>(digest);
    issues.Add(std::string(path), "sha256_failed", error.message);
    return std::nullopt;
  }
  return std::get<Sha256Digest>(digest);
}

void CheckExactBundleHashes(const ReferenceBundle& bundle,
                            IssueCollector& issues) {
  const std::optional<nlohmann::json> carrier =
      EncodeBundleJson(bundle, issues);
  if (!carrier.has_value()) {
    return;
  }
  try {
    const nlohmann::json& bundle_json =
        carrier->at("new_reference_bundle");
    nlohmann::json reference_json =
        bundle_json.at("platform_reference");
    reference_json.erase("reference_hash");
    const auto reference_digest =
        HashJson(reference_json, "platform_reference.reference_hash",
                 issues);
    if (reference_digest.has_value() &&
        *reference_digest != ReferenceHashOf(bundle.platform_reference)) {
      issues.Add("platform_reference.reference_hash",
                 "reference_hash_content_mismatch",
                 "reference_hash must bind the platform reference with "
                 "reference_hash omitted");
    }

    const std::array<std::string_view, 3> component_names{
        "route_skeleton", "committed_prefix", "preview"};
    const std::array<std::string_view, 3> expected_hashes{
        bundle.route_skeleton.component_hash,
        bundle.committed_prefix.component_hash,
        bundle.preview.component_hash,
    };
    for (std::size_t index = 0; index < component_names.size();
         ++index) {
      const nlohmann::json& component =
          bundle_json.at(std::string(component_names[index]));
      const auto digest =
          HashJson(component.at("content"),
                   std::string(component_names[index]) +
                       ".component_hash",
                   issues);
      if (digest.has_value() && *digest != expected_hashes[index]) {
        issues.Add(
            std::string(component_names[index]) + ".component_hash",
            "component_hash_content_mismatch",
            "component_hash must bind exactly the inline content object");
      }
    }

    nlohmann::json bundle_without_hash = bundle_json;
    bundle_without_hash.erase("bundle_hash");
    const auto bundle_digest =
        HashJson(bundle_without_hash, "bundle_hash", issues);
    if (bundle_digest.has_value() &&
        *bundle_digest != bundle.bundle_hash) {
      issues.Add("bundle_hash", "bundle_hash_content_mismatch",
                 "bundle_hash must bind the complete bundle with its "
                 "top-level hash omitted");
    }
  } catch (const nlohmann::json::exception& error) {
    issues.Add("bundle_hash", "hash_input_shape_mismatch",
               std::string("encoded bundle lacked a hash input: ") +
                   error.what());
  }
}

void CheckRouteSkeleton(
    const InlineComponent<RouteSkeletonContent>& component,
    const PlatformReference& reference,
    IssueCollector& issues) {
  if (component.component_id.empty()) {
    issues.Add("route_skeleton.component_id", "identifier_required",
               "component ID must not be empty");
  }
  CheckHash(component.component_hash,
            "route_skeleton.component_hash", issues);
  if (component.content.source_reference_id !=
      ReferenceIdOf(reference)) {
    issues.Add("route_skeleton.content.source_reference_id",
               "component_reference_id_mismatch",
               "route skeleton must bind the inline reference ID");
  }
  if (component.content.source_reference_hash !=
      ReferenceHashOf(reference)) {
    issues.Add("route_skeleton.content.source_reference_hash",
               "component_reference_hash_mismatch",
               "route skeleton must bind the inline reference hash");
  }
  for (std::size_t index = 0;
       index < component.content.waypoints.size(); ++index) {
    CheckPose(
        component.content.waypoints[index],
        IndexedPath("route_skeleton.content.waypoints", index), issues);
  }
  for (std::size_t index = 0;
       index < component.content.unresolved_tail.size(); ++index) {
    CheckFinite(
        component.content.unresolved_tail[index],
        IndexedPath("route_skeleton.content.unresolved_tail", index),
        issues);
  }
}

[[nodiscard]] std::size_t ReferenceSegmentCount(
    const PlatformReference& reference) {
  if (const auto* wheeled =
          std::get_if<WheeledReference>(&reference);
      wheeled != nullptr) {
    return wheeled->segments.size();
  }
  return 0U;
}

void CheckViewSelector(const ReferenceViewContent& view,
                       const PlatformReference& reference,
                       std::string_view path,
                       IssueCollector& issues) {
  std::visit(
      [&](const auto& selector) {
        using Selector = std::decay_t<decltype(selector)>;
        if constexpr (std::is_same_v<Selector, TimeViewSelector>) {
          CheckTimeInterval(selector.time_interval,
                            ChildPath(path, "selector.time_interval"),
                            issues);
          if (ReferencePlatform(reference) == PlatformType::kHopper) {
            issues.Add(ChildPath(path, "selector"),
                       "selector_platform_mismatch",
                       "Hopper views use ground-hold or jump selectors");
          }
        } else if constexpr (std::is_same_v<Selector,
                                            SegmentViewSelector>) {
          if (selector.first_segment_index >
              selector.past_last_segment_index) {
            issues.Add(ChildPath(path, "selector"),
                       "selector_range_reversed",
                       "first segment index must not exceed past-last");
          }
          const std::size_t segment_count =
              ReferenceSegmentCount(reference);
          if (ReferencePlatform(reference) != PlatformType::kWheeled ||
              selector.past_last_segment_index > segment_count) {
            issues.Add(ChildPath(path, "selector"),
                       "selector_outside_reference",
                       "segment selector must address the same wheeled "
                       "reference");
          }
        } else if constexpr (std::is_same_v<
                                 Selector, GroundHoldViewSelector>) {
          const auto* hopper =
              std::get_if<HopperReference>(&reference);
          if (view.role !=
                  ReferenceViewContent::Role::kCommittedPrefix ||
              hopper == nullptr ||
              selector.anchor_id !=
                  hopper->ground_hold_anchor.anchor_id) {
            issues.Add(ChildPath(path, "selector"),
                       "invalid_ground_hold_selector",
                       "ground-hold selector is committed-prefix only "
                       "and must resolve the same reference anchor");
          }
          if (hopper != nullptr) {
            const Vec3& linear =
                hopper->ground_hold_anchor.hold_state
                    .linear_velocity_mps;
            const Vec3& angular =
                hopper->ground_hold_anchor.hold_state
                    .angular_velocity_radps;
            if (Norm(linear) != 0.0 || Norm(angular) != 0.0) {
              issues.Add(ChildPath(path, "selector"),
                         "ground_hold_selector_not_stationary",
                         "selected ground-hold anchor must be stationary");
            }
            CheckContentRef(
                hopper->ground_hold_anchor.terrain_certification_ref,
                ChildPath(path,
                          "selector.terrain_certification_ref"),
                issues);
          }
        } else {
          const auto* hopper =
              std::get_if<HopperReference>(&reference);
          if (hopper == nullptr ||
              selector.boundary_id !=
                  hopper->jump_boundary.boundary_id) {
            issues.Add(ChildPath(path, "selector"),
                       "selector_outside_reference",
                       "jump selector must resolve the same Hopper "
                       "reference boundary");
          }
          if (view.role ==
                  ReferenceViewContent::Role::kCommittedPrefix &&
              selector.scope != JumpViewSelector::Scope::kNextHop) {
            issues.Add(ChildPath(path, "selector.scope"),
                       "committed_jump_scope_mismatch",
                       "committed jump view can select only NEXT_HOP");
          }
        }
      },
      view.selector);
}

void CheckViewComponent(
    const InlineComponent<ReferenceViewContent>& component,
    const PlatformReference& reference,
    const ReferenceViewContent::Role expected_role,
    std::string_view path,
    IssueCollector& issues) {
  if (component.component_id.empty()) {
    issues.Add(ChildPath(path, "component_id"), "identifier_required",
               "component ID must not be empty");
  }
  CheckHash(component.component_hash,
            ChildPath(path, "component_hash"), issues);
  if (component.content.role != expected_role) {
    issues.Add(ChildPath(path, "content.role"),
               "component_role_mismatch",
               "reference-view role must match its component slot");
  }
  if (component.content.source_reference_id !=
      ReferenceIdOf(reference)) {
    issues.Add(ChildPath(path, "content.source_reference_id"),
               "component_reference_id_mismatch",
               "view must bind the inline reference ID");
  }
  if (component.content.source_reference_hash !=
      ReferenceHashOf(reference)) {
    issues.Add(ChildPath(path, "content.source_reference_hash"),
               "component_reference_hash_mismatch",
               "view must bind the inline reference hash");
  }
  CheckViewSelector(component.content, reference,
                    ChildPath(path, "content"), issues);
}

void CheckReferenceValidity(const ReferenceValidity& validity,
                            PlatformType platform,
                            IssueCollector& issues) {
  CheckClockStamp(validity.valid_from, "validity.valid_from", issues);
  if (validity.valid_until.has_value()) {
    CheckClockStamp(*validity.valid_until, "validity.valid_until",
                    issues);
    if (validity.valid_until->clock_id !=
        validity.valid_from.clock_id) {
      issues.Add("validity.valid_until.clock_id", "clock_id_mismatch",
                 "validity endpoints must use one clock");
    } else if (validity.valid_until->tick <
               validity.valid_from.tick) {
      issues.Add("validity.valid_until",
                 "validity_interval_reversed",
                 "valid_until must not precede valid_from");
    }
  }
  CheckContentRef(validity.required_map_snapshot_ref,
                  "validity.required_map_snapshot_ref", issues);
  CheckContentRef(validity.required_capability_ref,
                  "validity.required_capability_ref", issues);
  const bool hopper_deviation =
      std::holds_alternative<HopperErrorBounds>(
          validity.allowed_state_deviation);
  if (hopper_deviation != (platform == PlatformType::kHopper)) {
    issues.Add("validity.allowed_state_deviation",
               "platform_state_deviation_mismatch",
               "allowed-state-deviation variant must match platform");
  }
  std::visit(
      [&](const auto& bounds) {
        using Bounds = std::decay_t<decltype(bounds)>;
        if constexpr (std::is_same_v<Bounds,
                                     WheeledOrLeggedErrorBounds>) {
          CheckWheeledOrLeggedError(
              bounds, "validity.allowed_state_deviation", issues);
        } else {
          CheckHopperError(
              bounds, "validity.allowed_state_deviation", issues);
        }
      },
      validity.allowed_state_deviation);
  if (validity.invalidation_conditions.empty()) {
    issues.Add("validity.invalidation_conditions",
               "invalidation_condition_required",
               "bundle validity needs at least one invalidation condition");
  }
  std::set<InvalidationCondition> unique_conditions;
  for (std::size_t index = 0;
       index < validity.invalidation_conditions.size(); ++index) {
    if (!unique_conditions
             .insert(validity.invalidation_conditions[index])
             .second) {
      issues.Add(
          IndexedPath("validity.invalidation_conditions", index),
          "duplicate_invalidation_condition",
          "invalidation conditions must be unique");
    }
  }
}

void CheckReferenceBundle(const ReferenceBundle& bundle,
                          IssueCollector& issues) {
  if (bundle.bundle_id.empty()) {
    issues.Add("bundle_id", "identifier_required",
               "bundle ID must not be empty");
  }
  if (bundle.bundle_revision == 0U ||
      bundle.bundle_revision > kMaximumRevision) {
    issues.Add("bundle_revision", "revision_out_of_range",
               "bundle revision must be in [1, 2147483647]");
  }
  CheckHash(bundle.bundle_hash, "bundle_hash", issues);
  if (bundle.source_request_id.empty()) {
    issues.Add("source_request_id", "identifier_required",
               "source request ID must not be empty");
  }
  CheckContentRef(bundle.source_map_snapshot_ref,
                  "source_map_snapshot_ref", issues);
  CheckContentRef(bundle.source_safety_capability_ref,
                  "source_safety_capability_ref", issues);
  CheckContentRef(bundle.source_algorithm_config_ref,
                  "source_algorithm_config_ref", issues);
  if (bundle.platform_type !=
      ReferencePlatform(bundle.platform_reference)) {
    issues.Add("platform_reference", "platform_reference_mismatch",
               "platform reference variant must match platform_type");
  }
  CheckPlatformReference(bundle.platform_reference, "", issues);
  CheckRouteSkeleton(bundle.route_skeleton, bundle.platform_reference,
                     issues);
  CheckViewComponent(
      bundle.committed_prefix, bundle.platform_reference,
      ReferenceViewContent::Role::kCommittedPrefix,
      "committed_prefix", issues);
  CheckViewComponent(
      bundle.preview, bundle.platform_reference,
      ReferenceViewContent::Role::kPreview, "preview", issues);
  const std::set<ComponentId> component_ids{
      bundle.route_skeleton.component_id,
      bundle.committed_prefix.component_id,
      bundle.preview.component_id,
  };
  if (component_ids.size() != 3U) {
    issues.Add("route_skeleton.component_id",
               "duplicate_component_id",
               "bundle component IDs must be distinct");
  }

  CheckReferenceValidity(bundle.validity, bundle.platform_type, issues);
  if (bundle.validity.required_map_snapshot_ref !=
      bundle.source_map_snapshot_ref) {
    issues.Add("validity.required_map_snapshot_ref",
               "bundle_map_provenance_mismatch",
               "validity map must equal the bundle source map");
  }
  if (bundle.validity.required_capability_ref !=
      bundle.source_safety_capability_ref) {
    issues.Add("validity.required_capability_ref",
               "bundle_capability_provenance_mismatch",
               "validity capability must equal bundle source capability");
  }
  if (const auto* hopper =
          std::get_if<HopperReference>(&bundle.platform_reference);
      hopper != nullptr &&
      hopper->certified_flight_tube.source_map_snapshot_ref !=
          bundle.source_map_snapshot_ref) {
    issues.Add(
        "certified_flight_tube.source_map_snapshot_ref",
        "bundle_map_provenance_mismatch",
        "Hopper flight tube must bind the bundle source map");
  }

  if (!bundle.validation_summary.hard_constraints_passed) {
    issues.Add("validation_summary.hard_constraints_passed",
               "hard_constraints_not_passed",
               "activatable bundle must pass hard constraints");
  }
  if (!bundle.validation_summary.continuous_validation_passed) {
    issues.Add("validation_summary.continuous_validation_passed",
               "continuous_validation_not_passed",
               "activatable bundle must pass continuous validation");
  }
  if (bundle.validation_summary.certificate_refs.empty()) {
    issues.Add("validation_summary.certificate_refs",
               "validation_certificate_required",
               "validation summary needs at least one certificate");
  }
  for (std::size_t index = 0;
       index < bundle.validation_summary.certificate_refs.size();
       ++index) {
    CheckContentRef(
        bundle.validation_summary.certificate_refs[index],
        IndexedPath("validation_summary.certificate_refs", index),
        issues);
  }
  std::set<std::string> warnings;
  for (std::size_t index = 0;
       index < bundle.validation_summary.warning_codes.size(); ++index) {
    if (!warnings
             .insert(bundle.validation_summary.warning_codes[index])
             .second) {
      issues.Add(
          IndexedPath("validation_summary.warning_codes", index),
          "duplicate_warning_code",
          "warning codes must be unique");
    }
  }
  if (bundle.generation_evidence.selected_candidate_id.empty()) {
    issues.Add("generation_evidence.selected_candidate_id",
               "identifier_required",
               "generation evidence needs a candidate ID");
  }
  if (bundle.generation_evidence.termination_reason.empty()) {
    issues.Add("generation_evidence.termination_reason",
               "termination_reason_required",
               "generation evidence needs a termination reason");
  }
  if (bundle.generation_evidence.evidence_refs.empty()) {
    issues.Add("generation_evidence.evidence_refs",
               "generation_evidence_ref_required",
               "generation evidence needs at least one content ref");
  }
  for (std::size_t index = 0;
       index < bundle.generation_evidence.evidence_refs.size();
       ++index) {
    CheckContentRef(
        bundle.generation_evidence.evidence_refs[index],
        IndexedPath("generation_evidence.evidence_refs", index),
        issues);
  }
  if (bundle.generation_evidence.learned_cost_snapshot_ref.has_value()) {
    CheckContentRef(
        *bundle.generation_evidence.learned_cost_snapshot_ref,
        "generation_evidence.learned_cost_snapshot_ref", issues);
  }
  const bool hopper_generation =
      bundle.generation_evidence.generation_mode ==
      GenerationMode::kCertifiedBallisticReference;
  if (hopper_generation !=
      (bundle.platform_type == PlatformType::kHopper)) {
    issues.Add("generation_evidence.generation_mode",
               "generation_mode_platform_mismatch",
               "certified ballistic generation is Hopper-only");
  }
  CheckExactBundleHashes(bundle, issues);
}

[[nodiscard]] bool IsReadyOutcome(const PlanningOutcome outcome) {
  return outcome == PlanningOutcome::kNewReferenceReady ||
         outcome == PlanningOutcome::kSafeFrontierReferenceReady;
}

[[nodiscard]] bool IsContinueDirective(
    const ExecutionDirective directive) {
  return directive == ExecutionDirective::kContinueActiveBundle ||
         directive == ExecutionDirective::kContinueCommittedJump;
}

void CheckCallDiagnostics(const CallDiagnostics& diagnostics,
                          IssueCollector& issues) {
  CheckDuration(diagnostics.api_latency, "call_diagnostics.api_latency",
                issues);
  if (diagnostics.termination_reason.empty()) {
    issues.Add("call_diagnostics.termination_reason",
               "termination_reason_required",
               "call diagnostics need a termination reason");
  }
  if (diagnostics.final_epsilon.has_value()) {
    CheckFinite(*diagnostics.final_epsilon,
                "call_diagnostics.final_epsilon", issues);
    if (std::isfinite(*diagnostics.final_epsilon) &&
        *diagnostics.final_epsilon < 1.0) {
      issues.Add("call_diagnostics.final_epsilon",
                 "epsilon_below_one",
                 "reported ARA* epsilon must be at least one");
    }
  }
  if (diagnostics.expected_execution_time.has_value()) {
    CheckDuration(
        *diagnostics.expected_execution_time,
        "call_diagnostics.expected_execution_time", issues);
  }
  if (diagnostics.secondary_costs.has_value()) {
    const SecondaryCosts& costs = *diagnostics.secondary_costs;
    if (costs.energy.has_value()) {
      CheckNonNegative(*costs.energy,
                       "call_diagnostics.secondary_costs.energy",
                       issues);
    }
    if (costs.nonfatal_risk.has_value()) {
      CheckNonNegative(
          *costs.nonfatal_risk,
          "call_diagnostics.secondary_costs.nonfatal_risk", issues);
    }
    if (costs.smoothness.has_value()) {
      CheckNonNegative(
          *costs.smoothness,
          "call_diagnostics.secondary_costs.smoothness", issues);
    }
  }
  const bool learned_disabled =
      diagnostics.learned_cost_usage == LearnedCostUsage::kDisabled;
  if (learned_disabled &&
      diagnostics.learned_cost_snapshot_ref.has_value()) {
    issues.Add("call_diagnostics.learned_cost_snapshot_ref",
               "disabled_learned_cost_forbids_snapshot_ref",
               "DISABLED usage must not carry a learned snapshot ref");
  }
  if (!learned_disabled &&
      !diagnostics.learned_cost_snapshot_ref.has_value()) {
    issues.Add("call_diagnostics.learned_cost_snapshot_ref",
               "learned_cost_snapshot_ref_required",
               "USED and FALLBACK usage require the attempted snapshot");
  }
  if (diagnostics.learned_cost_snapshot_ref.has_value()) {
    CheckContentRef(
        *diagnostics.learned_cost_snapshot_ref,
        "call_diagnostics.learned_cost_snapshot_ref", issues);
  }
  std::set<std::string> message_codes;
  for (std::size_t index = 0;
       index < diagnostics.message_codes.size(); ++index) {
    if (!message_codes.insert(diagnostics.message_codes[index]).second) {
      issues.Add(
          IndexedPath("call_diagnostics.message_codes", index),
          "duplicate_message_code",
          "call diagnostic message codes must be unique");
    }
  }
}

void CheckPlanningResponse(const PlanningResponse& response,
                           IssueCollector& issues) {
  if (response.request_id.empty()) {
    issues.Add("request_id", "identifier_required",
               "response request ID must not be empty");
  }
  CheckClockStamp(response.response_time, "response_time", issues);
  if (response.reason_code.empty()) {
    issues.Add("reason_code", "reason_code_required",
               "response reason code must not be empty");
  }
  if (response.active_bundle_ref.has_value()) {
    CheckContentRef(*response.active_bundle_ref, "active_bundle_ref",
                    issues);
  }
  CheckCallDiagnostics(response.call_diagnostics, issues);

  const bool activates =
      response.execution_directive ==
      ExecutionDirective::kActivateNewBundle;
  if (activates && !response.new_reference_bundle.has_value()) {
    issues.Add("new_reference_bundle",
               "activation_requires_validated_bundle",
               "activation requires a locally and contextually validated "
               "bundle");
  }
  if (!activates && response.new_reference_bundle.has_value()) {
    issues.Add("new_reference_bundle",
               "non_activation_forbids_new_bundle",
               "only ACTIVATE_NEW_BUNDLE may carry a new bundle");
  }
  if (response.new_reference_bundle.has_value()) {
    IssueCollector nested;
    CheckReferenceBundle(*response.new_reference_bundle, nested);
    issues.Merge(nested.Finish());
    if (response.request_id !=
        response.new_reference_bundle->source_request_id) {
      issues.Add("new_reference_bundle.source_request_id",
                 "response_bundle_request_mismatch",
                 "response and bundle request IDs must match");
    }
  }

  if (IsReadyOutcome(response.planning_outcome) != activates) {
    issues.Add("execution_directive", "ready_activation_mismatch",
               "READY outcomes and bundle activation must occur together");
  }
  const bool continues =
      IsContinueDirective(response.execution_directive);
  if (continues && !response.active_bundle_ref.has_value()) {
    issues.Add("active_bundle_ref",
               "continue_requires_active_bundle_ref",
               "continue directives require the active bundle ref");
  }
  if (!continues && response.active_bundle_ref.has_value()) {
    issues.Add("active_bundle_ref",
               "non_continue_forbids_active_bundle_ref",
               "only continue directives may carry active_bundle_ref");
  }
  if (response.execution_directive ==
          ExecutionDirective::kHoldStationary &&
      (response.new_reference_bundle.has_value() ||
       response.active_bundle_ref.has_value() ||
       IsReadyOutcome(response.planning_outcome))) {
    issues.Add("execution_directive",
               "stationary_hold_contract_violation",
               "stationary hold carries no bundle ref and no READY outcome");
  }
  if (response.planning_outcome ==
      PlanningOutcome::kActiveReferenceInvalidated) {
    const bool permitted =
        response.execution_directive ==
            ExecutionDirective::kHoldStationary ||
        response.execution_directive ==
            ExecutionDirective::kContinueCommittedJump ||
        response.execution_directive ==
            ExecutionDirective::kNoSafePlannerReference;
    if (!permitted) {
      issues.Add("execution_directive",
                 "invalidated_reference_directive_forbidden",
                 "invalidated references may only hold, continue a "
                 "committed jump, or report no safe reference");
    }
  }
}

[[nodiscard]] bool ContainsRef(const std::vector<ContentRef>& refs,
                               const ContentRef& expected) {
  return std::find(refs.begin(), refs.end(), expected) != refs.end();
}

[[nodiscard]] std::shared_ptr<const ImmutableContractObject>
ResolveExactContractObject(const ContractObjectRegistry& registry,
                           const ContentRef& reference,
                           const ContractObjectKind expected_kind,
                           std::string_view path,
                           IssueCollector& issues) {
  const auto resolved = registry.Resolve(reference, expected_kind);
  if (!IsOk(resolved)) {
    const Error& error = std::get<Error>(resolved);
    issues.Add(std::string(path), "missing_registry_object",
               error.message.empty()
                   ? "referenced immutable object is missing"
                   : error.message);
    return nullptr;
  }
  const auto& object =
      std::get<std::shared_ptr<const ImmutableContractObject>>(resolved);
  if (!object) {
    issues.Add(std::string(path), "missing_registry_object",
               "registry returned a null immutable object");
    return nullptr;
  }
  if (object->content_ref() != reference) {
    issues.Add(std::string(path), "registry_content_ref_mismatch",
               "registry object ID, revision and hash must match exactly");
  }
  if (object->kind() != expected_kind) {
    issues.Add(std::string(path), "registry_object_kind_mismatch",
               "registry object kind must match the requested kind");
  }
  return object;
}

void CheckCertification(
    const ContractObjectRegistry& registry,
    const ContentRef& certificate_ref,
    std::string_view path,
    const ReferenceBundle& bundle,
    const std::vector<ContentRef>& fixed_candidate_inputs,
    IssueCollector& issues) {
  const auto object = ResolveExactContractObject(
      registry, certificate_ref, ContractObjectKind::kCertification,
      path, issues);
  if (!object) {
    return;
  }
  const auto* certificate =
      dynamic_cast<const CertificationObject*>(object.get());
  if (certificate == nullptr) {
    issues.Add(std::string(path),
               "certification_provenance_required",
               "certification objects must expose immutable provenance");
    return;
  }
  const CertificationProvenance& provenance =
      certificate->provenance();
  CheckContentRef(
      provenance.source_map_snapshot_ref,
      ChildPath(path, "provenance.source_map_snapshot_ref"), issues);
  CheckContentRef(
      provenance.source_safety_capability_ref,
      ChildPath(path, "provenance.source_safety_capability_ref"),
      issues);
  CheckContentRef(
      provenance.source_algorithm_config_ref,
      ChildPath(path, "provenance.source_algorithm_config_ref"),
      issues);
  if (provenance.source_map_snapshot_ref !=
      bundle.source_map_snapshot_ref) {
    issues.Add(ChildPath(path, "provenance.source_map_snapshot_ref"),
               "certificate_map_provenance_mismatch",
               "certificate must bind the activation map");
  }
  if (provenance.source_safety_capability_ref !=
      bundle.source_safety_capability_ref) {
    issues.Add(
        ChildPath(path,
                  "provenance.source_safety_capability_ref"),
        "certificate_capability_provenance_mismatch",
        "certificate must bind the activation capability");
  }
  if (provenance.source_algorithm_config_ref !=
      bundle.source_algorithm_config_ref) {
    issues.Add(
        ChildPath(path, "provenance.source_algorithm_config_ref"),
        "certificate_algorithm_provenance_mismatch",
        "certificate must bind the activation algorithm config");
  }
  if (provenance.certification_purpose.empty()) {
    issues.Add(ChildPath(path, "provenance.certification_purpose"),
               "certification_purpose_required",
               "certificate provenance needs a stable purpose");
  }
  for (std::size_t index = 0; index < provenance.input_refs.size();
       ++index) {
    CheckContentRef(
        provenance.input_refs[index],
        IndexedPath(ChildPath(path, "provenance.input_refs"), index),
        issues);
  }
  for (const ContentRef& required_input : fixed_candidate_inputs) {
    if (!ContainsRef(provenance.input_refs, required_input)) {
      issues.Add(ChildPath(path, "provenance.input_refs"),
                 "certificate_fixed_input_missing",
                 "certificate must repeat every fixed candidate model "
                 "and envelope input");
    }
  }
}

[[nodiscard]] std::vector<std::pair<ContentRef, std::string>>
CollectCertificateRefs(const ReferenceBundle& bundle) {
  std::vector<std::pair<ContentRef, std::string>> certificates;
  std::visit(
      [&](const auto& reference) {
        using Reference = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Reference, WheeledReference>) {
          certificates.emplace_back(
              reference.safe_stop_anchor.terrain_certification_ref,
              "safe_stop_anchor.terrain_certification_ref");
          for (std::size_t segment_index = 0;
               segment_index < reference.segments.size();
               ++segment_index) {
            const auto* drive =
                std::get_if<DriveSegment>(
                    &reference.segments[segment_index]);
            if (drive == nullptr) {
              continue;
            }
            const auto* chain =
                std::get_if<ValidatedPrimitiveChain>(
                    &drive->geometric_path);
            if (chain == nullptr) {
              continue;
            }
            for (std::size_t primitive_index = 0;
                 primitive_index < chain->primitives.size();
                 ++primitive_index) {
              certificates.emplace_back(
                  chain->primitives[primitive_index].validation_ref,
                  IndexedPath(
                      ChildPath(
                          IndexedPath("segments", segment_index),
                          "geometric_path.primitives"),
                      primitive_index) +
                      ".validation_ref");
            }
          }
        } else if constexpr (std::is_same_v<Reference,
                                            LeggedBodyReference>) {
          certificates.emplace_back(
              reference.terrain_normal_envelope
                  .source_terrain_certification_ref,
              "terrain_normal_envelope."
              "source_terrain_certification_ref");
          certificates.emplace_back(
              reference.safe_stop_anchor.terrain_certification_ref,
              "safe_stop_anchor.terrain_certification_ref");
          if (const auto* chain =
                  std::get_if<ValidatedPrimitiveChain>(
                      &reference.geometric_path);
              chain != nullptr) {
            for (std::size_t primitive_index = 0;
                 primitive_index < chain->primitives.size();
                 ++primitive_index) {
              certificates.emplace_back(
                  chain->primitives[primitive_index].validation_ref,
                  IndexedPath("geometric_path.primitives",
                              primitive_index) +
                      ".validation_ref");
            }
          }
        } else {
          certificates.emplace_back(
              reference.ground_hold_anchor.terrain_certification_ref,
              "ground_hold_anchor.terrain_certification_ref");
          certificates.emplace_back(
              reference.next_landing_region.terrain_certification_ref,
              "next_landing_region.terrain_certification_ref");
          certificates.emplace_back(
              reference.attitude_boundary.certification_ref,
              "attitude_boundary.certification_ref");
          certificates.emplace_back(
              reference.physical_certification_ref,
              "physical_certification_ref");
        }
      },
      bundle.platform_reference);
  for (std::size_t index = 0;
       index < bundle.validation_summary.certificate_refs.size();
       ++index) {
    certificates.emplace_back(
        bundle.validation_summary.certificate_refs[index],
        IndexedPath("validation_summary.certificate_refs", index));
  }
  return certificates;
}

template <class T>
[[nodiscard]] bool BindingIdentityEquals(
    const ResolvedBinding<T>& lhs,
    const ResolvedBinding<T>& rhs) {
  return lhs.content_ref == rhs.content_ref &&
         static_cast<bool>(lhs.object) == static_cast<bool>(rhs.object);
}

template <class T>
[[nodiscard]] bool OptionalBindingIdentityEquals(
    const std::optional<ResolvedBinding<T>>& lhs,
    const std::optional<ResolvedBinding<T>>& rhs) {
  return lhs.has_value() == rhs.has_value() &&
         (!lhs.has_value() ||
          BindingIdentityEquals(*lhs, *rhs));
}

void CheckRegistryCapabilityBindings(
    const ReferenceActivationContext& context,
    IssueCollector& issues) {
  if (!context.request.safety_capability) {
    return;
  }
  const auto resolved = context.registry.ResolveCapabilityBindings(
      *context.request.safety_capability);
  if (!IsOk(resolved)) {
    const Error& error = std::get<Error>(resolved);
    issues.Add("capability_bindings", "missing_registry_object",
               error.message.empty()
                   ? "registry could not reproduce fixed capability "
                     "bindings"
                   : error.message);
    return;
  }
  const ResolvedCapabilityBindings& registry_bindings =
      std::get<ResolvedCapabilityBindings>(resolved);
  const ResolvedCapabilityBindings& request_bindings =
      context.request.capability_bindings;
  if (!BindingIdentityEquals(registry_bindings.motion_model,
                             request_bindings.motion_model)) {
    issues.Add("capability_bindings.motion_model",
               "activation_binding_registry_mismatch",
               "activation must reproduce the request-fixed motion model");
  }
  if (!BindingIdentityEquals(registry_bindings.analytic_cost_model,
                             request_bindings.analytic_cost_model)) {
    issues.Add(
        "capability_bindings.analytic_cost_model",
        "activation_binding_registry_mismatch",
        "activation must reproduce the request-fixed analytic cost model");
  }
  if (!OptionalBindingIdentityEquals(
          registry_bindings.gravity_model,
          request_bindings.gravity_model)) {
    issues.Add("capability_bindings.gravity_model",
               "activation_binding_registry_mismatch",
               "activation must reproduce the request-fixed gravity model");
  }
  if (!OptionalBindingIdentityEquals(
          registry_bindings.error_model,
          request_bindings.error_model)) {
    issues.Add(
        "capability_bindings.error_model",
        "activation_binding_registry_mismatch",
        "activation must reproduce the request-fixed error model");
  }
  if (!OptionalBindingIdentityEquals(
          registry_bindings.actuator_or_impulse_profile,
          request_bindings.actuator_or_impulse_profile)) {
    issues.Add(
        "capability_bindings.actuator_or_impulse_profile",
        "activation_binding_registry_mismatch",
        "activation must reproduce the request-fixed actuator profile");
  }
  if (!OptionalBindingIdentityEquals(
          registry_bindings.body_rotation_envelope,
          request_bindings.body_rotation_envelope)) {
    issues.Add(
        "capability_bindings.body_rotation_envelope",
        "activation_binding_registry_mismatch",
        "activation must reproduce the fixed body-rotation envelope");
  }
  if (!OptionalBindingIdentityEquals(
          registry_bindings.attitude_tightening_table,
          request_bindings.attitude_tightening_table)) {
    issues.Add(
        "capability_bindings.attitude_tightening_table",
        "activation_binding_registry_mismatch",
        "activation must reproduce the fixed tightening table");
  }
}

[[nodiscard]] Vec3 SetCenter(const DeterministicVectorSet3& set) {
  return std::visit([](const auto& value) { return value.center; }, set);
}

[[nodiscard]] bool DeterministicSetContains(
    const Vec3& outer_nominal,
    const DeterministicVectorSet3& outer,
    const Vec3& inner_nominal,
    const DeterministicVectorSet3& inner) {
  const Vec3 outer_center = Add(outer_nominal, SetCenter(outer));
  const Vec3 inner_center = Add(inner_nominal, SetCenter(inner));
  return std::visit(
      [&](const auto& outer_set, const auto& inner_set) {
        using Outer = std::decay_t<decltype(outer_set)>;
        using Inner = std::decay_t<decltype(inner_set)>;
        const Vec3 delta = Subtract(inner_center, outer_center);
        if constexpr (std::is_same_v<Outer, AxisAlignedBox3> &&
                      std::is_same_v<Inner, AxisAlignedBox3>) {
          return std::abs(delta.x) + inner_set.half_extent.x <=
                     outer_set.half_extent.x + kGeometryTolerance &&
                 std::abs(delta.y) + inner_set.half_extent.y <=
                     outer_set.half_extent.y + kGeometryTolerance &&
                 std::abs(delta.z) + inner_set.half_extent.z <=
                     outer_set.half_extent.z + kGeometryTolerance;
        } else if constexpr (std::is_same_v<Outer, EuclideanBall3> &&
                             std::is_same_v<Inner, EuclideanBall3>) {
          return Norm(delta) + inner_set.radius <=
                 outer_set.radius + kGeometryTolerance;
        } else if constexpr (std::is_same_v<Outer, AxisAlignedBox3> &&
                             std::is_same_v<Inner, EuclideanBall3>) {
          return std::abs(delta.x) + inner_set.radius <=
                     outer_set.half_extent.x + kGeometryTolerance &&
                 std::abs(delta.y) + inner_set.radius <=
                     outer_set.half_extent.y + kGeometryTolerance &&
                 std::abs(delta.z) + inner_set.radius <=
                     outer_set.half_extent.z + kGeometryTolerance;
        } else {
          const Vec3 farthest{
              std::abs(delta.x) + inner_set.half_extent.x,
              std::abs(delta.y) + inner_set.half_extent.y,
              std::abs(delta.z) + inner_set.half_extent.z,
          };
          return Norm(farthest) <=
                 outer_set.radius + kGeometryTolerance;
        }
      },
      outer, inner);
}

void CheckRequestStateContainedInGroundHold(
    const HopperState& request_state,
    const GroundHoldAnchor& hold,
    IssueCollector& issues) {
  const bool position_contained = DeterministicSetContains(
      hold.hold_state.position_m,
      hold.allowed_hold_state_error_set.position_bound_m,
      request_state.position_m, request_state.error_bounds.position_bound_m);
  const bool linear_velocity_contained = DeterministicSetContains(
      hold.hold_state.linear_velocity_mps,
      hold.allowed_hold_state_error_set.linear_velocity_bound_mps,
      request_state.linear_velocity_mps,
      request_state.error_bounds.linear_velocity_bound_mps);
  const bool angular_velocity_contained = DeterministicSetContains(
      hold.hold_state.angular_velocity_radps,
      hold.allowed_hold_state_error_set.angular_velocity_bound_radps,
      request_state.angular_velocity_radps,
      request_state.error_bounds.angular_velocity_bound_radps);
  const double orientation_center_distance =
      QuaternionAngularDistance(
          hold.hold_state.orientation_body_to_frame,
          request_state.orientation_body_to_frame);
  const bool orientation_contained =
      orientation_center_distance +
          request_state.error_bounds.orientation_bound.radius_rad <=
      hold.allowed_hold_state_error_set.orientation_bound.radius_rad +
          kGeometryTolerance;
  if (!position_contained || !linear_velocity_contained ||
      !angular_velocity_contained || !orientation_contained) {
    issues.Add("current_state",
               "request_state_not_contained_in_ground_hold",
               "request Hopper state set must be contained in the "
               "ground-hold set");
  }
}

[[nodiscard]] double PolygonArea(const ConvexPolygonUv& polygon) {
  double twice_area = 0.0;
  for (std::size_t index = 0; index < polygon.vertices_uv.size();
       ++index) {
    const Vec2& current = polygon.vertices_uv[index];
    const Vec2& next =
        polygon.vertices_uv[(index + 1U) % polygon.vertices_uv.size()];
    twice_area += current.x * next.y - current.y * next.x;
  }
  return 0.5 * std::abs(twice_area);
}

void CheckHopperCapabilityLimits(const HopperReference& reference,
                                 const HopperCapability& capability,
                                 IssueCollector& issues) {
  const HopperLaunchLimits& limits = capability.launch_limits;
  const auto flight_time =
      reference.jump_boundary.ballistic_flight_time.value;
  if (flight_time < limits.minimum_flight_time.value ||
      flight_time > limits.maximum_flight_time.value) {
    issues.Add("jump_boundary.ballistic_flight_time",
               "flight_time_outside_capability",
               "ballistic duration must lie in the certified capability "
               "interval");
  }
  const double launch_speed =
      Norm(reference.jump_boundary.nominal_launch_state
               .linear_velocity_mps) +
      MaximumSetVectorNorm(
          reference.jump_boundary.allowed_launch_state_error_set
              .linear_velocity_bound_mps);
  if (std::isfinite(launch_speed) &&
      launch_speed >
          limits.maximum_launch_speed_mps + kGeometryTolerance) {
    issues.Add("jump_boundary.nominal_launch_state.linear_velocity_mps",
               "launch_speed_limit_exceeded",
               "error-expanded launch speed exceeds capability");
  }
  const double landing_speed = MaximumBoundsNorm(
      reference.predicted_landing_footprint.landing_velocity_bounds);
  if (std::isfinite(landing_speed) &&
      landing_speed >
          limits.maximum_landing_speed_mps + kGeometryTolerance) {
    issues.Add(
        "predicted_landing_footprint.landing_velocity_bounds",
        "landing_speed_limit_exceeded",
        "worst-case landing speed exceeds capability");
  }
  const double maximum_vertical_velocity =
      reference.predicted_landing_footprint
          .landing_velocity_bounds.upper.z;
  if (std::isfinite(maximum_vertical_velocity) &&
      -maximum_vertical_velocity +
              kGeometryTolerance <
          limits.minimum_downward_impact_speed_mps) {
    issues.Add(
        "predicted_landing_footprint.landing_velocity_bounds.z",
        "downward_impact_speed_limit_violated",
        "every certified landing velocity must satisfy the minimum "
        "downward impact speed");
  }
  const double required_clearance =
      std::max(
          {limits.minimum_landing_clearance_m,
           capability.landing_terrain_thresholds
               .minimum_overhead_clearance_m,
           capability.landing_terrain_thresholds
               .minimum_lateral_clearance_m});
  if (reference.certified_flight_tube.minimum_certified_clearance_m +
          kGeometryTolerance <
      required_clearance) {
    issues.Add(
        "certified_flight_tube.minimum_certified_clearance_m",
        "clearance_limit_violated",
        "certified tube clearance is below the fixed capability limit");
  }
  if (reference.next_landing_region.landing_plane.residual_bound_m >
          capability.landing_terrain_thresholds
                  .maximum_plane_residual_m +
              kGeometryTolerance) {
    issues.Add("next_landing_region.landing_plane.residual_bound_m",
               "landing_plane_residual_limit_exceeded",
               "landing plane residual exceeds capability");
  }
  if (PolygonArea(reference.next_landing_region.convex_polygon) +
          kGeometryTolerance <
      capability.landing_terrain_thresholds
          .minimum_landing_region_area_m2) {
    issues.Add("next_landing_region.convex_polygon",
               "landing_region_area_limit_violated",
               "landing region is smaller than the capability minimum");
  }

  const ArbitraryAxisAttitudeEnvelope& envelope =
      capability.attitude_envelope;
  const double initial_angular_speed =
      Norm(reference.jump_boundary.nominal_launch_state
               .angular_velocity_radps) +
      MaximumSetVectorNorm(
          reference.attitude_boundary
              .initial_angular_velocity_error_set_radps);
  if (std::isfinite(initial_angular_speed) &&
      initial_angular_speed >
          envelope.maximum_initial_angular_speed_radps +
              kGeometryTolerance) {
    issues.Add(
        "attitude_boundary.initial_angular_velocity_error_set_radps",
        "initial_angular_speed_limit_exceeded",
        "error-expanded initial angular speed exceeds capability");
  }
  const double landing_angular_speed =
      MaximumSetVectorNorm(
          reference.attitude_boundary
              .landing_angular_velocity_bounds_radps);
  if (std::isfinite(landing_angular_speed) &&
      landing_angular_speed >
          envelope.maximum_angular_speed_radps + kGeometryTolerance) {
    issues.Add(
        "attitude_boundary.landing_angular_velocity_bounds_radps",
        "landing_angular_speed_limit_exceeded",
        "landing angular-velocity set exceeds capability");
  }
  if (reference.attitude_boundary.settle_guard.value <
      envelope.minimum_settle_guard.value) {
    issues.Add("attitude_boundary.settle_guard",
               "settle_guard_below_capability",
               "settle guard must meet the certified minimum");
  }
}

void CheckActivationRegistryRoots(
    const ReferenceBundle& bundle,
    const ReferenceActivationContext& context,
    IssueCollector& issues) {
  const PlanningRequest& request = context.request;
  if (request.map_snapshot) {
    if (request.map_snapshot->snapshot_ref() !=
        bundle.source_map_snapshot_ref) {
      issues.Add("source_map_snapshot_ref",
                 "bundle_map_request_mismatch",
                 "bundle map must equal the request-fixed snapshot");
    }
    const auto registered_map = context.registry.FindMapSnapshot(
        bundle.source_map_snapshot_ref,
        request.map_snapshot->immutable_data_handle());
    if (!registered_map) {
      issues.Add("source_map_snapshot_ref", "missing_registry_object",
                 "activation map is absent from the registry");
    } else {
      if (registered_map->snapshot_ref() !=
          bundle.source_map_snapshot_ref) {
        issues.Add("source_map_snapshot_ref",
                   "registry_content_ref_mismatch",
                   "registry map identity must match bundle exactly");
      }
      for (const LayerManifestEntry& request_layer :
           request.map_snapshot->layer_manifest()) {
        const auto registered_layer =
            registered_map->LayerIdentity(request_layer.layer_kind);
        if (!registered_layer.has_value() ||
            registered_layer->get() != request_layer.content_ref) {
          issues.Add(
              std::string("map_snapshot.layer_manifest.") +
                  std::string(
                      LayerKindFieldName(request_layer.layer_kind)),
              "registry_map_layer_ref_mismatch",
              "registry map layer identity must match the request "
              "snapshot exactly");
        }
      }
      for (const LayerManifestEntry& registered_layer :
           registered_map->layer_manifest()) {
        if (!request.map_snapshot
                 ->LayerIdentity(registered_layer.layer_kind)
                 .has_value()) {
          issues.Add(
              std::string("map_snapshot.layer_manifest.") +
                  std::string(
                      LayerKindFieldName(registered_layer.layer_kind)),
              "registry_map_layer_ref_mismatch",
              "registry map must not add a layer absent from the "
              "request snapshot");
        }
      }
      if (registered_map->LayerManifestHash() !=
          request.map_snapshot->LayerManifestHash()) {
        issues.Add("map_snapshot.layer_manifest",
                   "registry_map_manifest_mismatch",
                   "registry and request map manifests must match exactly");
      }
    }
  }
  if (request.safety_capability) {
    const auto registered_capability =
        context.registry.FindSafetyCapability(
            bundle.source_safety_capability_ref);
    if (!registered_capability) {
      issues.Add("source_safety_capability_ref",
                 "missing_registry_object",
                 "activation capability is absent from the registry");
    } else if (registered_capability->content_ref !=
               bundle.source_safety_capability_ref) {
      issues.Add("source_safety_capability_ref",
                 "registry_content_ref_mismatch",
                 "registry capability identity must match exactly");
    }
    if (request.safety_capability->content_ref !=
        bundle.source_safety_capability_ref) {
      issues.Add("source_safety_capability_ref",
                 "bundle_capability_request_mismatch",
                 "bundle capability must equal the request-fixed profile");
    }
  }
  if (request.algorithm_config) {
    const auto registered_config =
        context.registry.FindAlgorithmConfig(
            bundle.source_algorithm_config_ref);
    if (!registered_config) {
      issues.Add("source_algorithm_config_ref",
                 "missing_registry_object",
                 "activation algorithm config is absent from registry");
    } else if (registered_config->content_ref !=
               bundle.source_algorithm_config_ref) {
      issues.Add("source_algorithm_config_ref",
                 "registry_content_ref_mismatch",
                 "registry algorithm config identity must match exactly");
    }
    if (request.algorithm_config->content_ref !=
        bundle.source_algorithm_config_ref) {
      issues.Add("source_algorithm_config_ref",
                 "bundle_algorithm_request_mismatch",
                 "bundle algorithm config must equal the request-fixed "
                 "config");
    }
  }
}

void CheckHopperActivationBindings(
    const HopperReference& reference,
    const ReferenceActivationContext& context,
    std::vector<ContentRef>& fixed_candidate_inputs,
    IssueCollector& issues) {
  const ResolvedCapabilityBindings& bindings =
      context.request.capability_bindings;
  if (bindings.gravity_model.has_value()) {
    fixed_candidate_inputs.push_back(
        bindings.gravity_model->content_ref);
    if (reference.jump_boundary.gravity_model_ref !=
        bindings.gravity_model->content_ref) {
      issues.Add("jump_boundary.gravity_model_ref",
                 "hopper_gravity_binding_mismatch",
                 "jump boundary gravity must equal request-fixed gravity");
    }
    static_cast<void>(ResolveExactContractObject(
        context.registry, reference.jump_boundary.gravity_model_ref,
        ContractObjectKind::kGravityModel,
        "jump_boundary.gravity_model_ref", issues));
  }
  if (bindings.actuator_or_impulse_profile.has_value()) {
    fixed_candidate_inputs.push_back(
        bindings.actuator_or_impulse_profile->content_ref);
    if (reference.jump_boundary.actuator_or_impulse_profile_ref !=
        bindings.actuator_or_impulse_profile->content_ref) {
      issues.Add(
          "jump_boundary.actuator_or_impulse_profile_ref",
          "hopper_actuator_binding_mismatch",
          "jump boundary actuator/impulse ref must equal the "
          "request-fixed profile");
    }
    static_cast<void>(ResolveExactContractObject(
        context.registry,
        reference.jump_boundary.actuator_or_impulse_profile_ref,
        ContractObjectKind::kActuatorOrImpulseProfile,
        "jump_boundary.actuator_or_impulse_profile_ref", issues));
  }
  if (bindings.error_model.has_value()) {
    fixed_candidate_inputs.push_back(
        bindings.error_model->content_ref);
    if (reference.predicted_landing_footprint.source_error_model_ref !=
        bindings.error_model->content_ref) {
      issues.Add(
          "predicted_landing_footprint.source_error_model_ref",
          "hopper_error_model_binding_mismatch",
          "landing footprint error model must equal request binding");
    }
    if (reference.certified_flight_tube.error_model_ref !=
        bindings.error_model->content_ref) {
      issues.Add("certified_flight_tube.error_model_ref",
                 "hopper_error_model_binding_mismatch",
                 "flight-tube error model must equal request binding");
    }
    static_cast<void>(ResolveExactContractObject(
        context.registry,
        reference.predicted_landing_footprint.source_error_model_ref,
        ContractObjectKind::kDeterministicErrorModel,
        "predicted_landing_footprint.source_error_model_ref", issues));
    static_cast<void>(ResolveExactContractObject(
        context.registry,
        reference.certified_flight_tube.error_model_ref,
        ContractObjectKind::kDeterministicErrorModel,
        "certified_flight_tube.error_model_ref", issues));
  }
  if (bindings.body_rotation_envelope.has_value()) {
    fixed_candidate_inputs.push_back(
        bindings.body_rotation_envelope->content_ref);
    if (reference.certified_flight_tube.body_rotation_envelope_ref !=
        bindings.body_rotation_envelope->content_ref) {
      issues.Add(
          "certified_flight_tube.body_rotation_envelope_ref",
          "hopper_body_rotation_binding_mismatch",
          "flight tube body-rotation envelope must equal request binding");
    }
    static_cast<void>(ResolveExactContractObject(
        context.registry,
        reference.certified_flight_tube.body_rotation_envelope_ref,
        ContractObjectKind::kBodyRotationEnvelope,
        "certified_flight_tube.body_rotation_envelope_ref", issues));
  }
  if (bindings.attitude_tightening_table.has_value()) {
    static_cast<void>(ResolveExactContractObject(
        context.registry,
        bindings.attitude_tightening_table->content_ref,
        ContractObjectKind::kAttitudeTighteningTable,
        "capability_bindings.attitude_tightening_table", issues));
  }
}

void CheckBundleLearnedBinding(
    const ReferenceBundle& bundle,
    const ReferenceActivationContext& context,
    IssueCollector& issues) {
  const auto& evidence_ref =
      bundle.generation_evidence.learned_cost_snapshot_ref;
  if (!evidence_ref.has_value()) {
    return;
  }
  if (!context.request.learned_cost_snapshot.has_value()) {
    issues.Add("generation_evidence.learned_cost_snapshot_ref",
               "learned_snapshot_not_fixed_by_request",
               "generation evidence may copy only the request-fixed "
               "snapshot");
    return;
  }
  const LearnedCostSnapshotBinding& request_binding =
      *context.request.learned_cost_snapshot;
  if (*evidence_ref != request_binding.snapshot_ref) {
    issues.Add("generation_evidence.learned_cost_snapshot_ref",
               "learned_snapshot_request_mismatch",
               "generation evidence snapshot must match request exactly");
  }
  const auto registered = context.registry.FindLearnedCost(
      request_binding.snapshot_ref, request_binding.registry_handle);
  if (!registered) {
    issues.Add("generation_evidence.learned_cost_snapshot_ref",
               "missing_registry_object",
               "learned-cost snapshot is absent from the registry");
  } else if (request_binding.resolved_snapshot &&
             registered.get() !=
                 request_binding.resolved_snapshot.get()) {
    issues.Add("generation_evidence.learned_cost_snapshot_ref",
               "learned_snapshot_registry_mismatch",
               "activation must use the same immutable learned snapshot");
  }
}

void CheckReferenceActivation(const ReferenceBundle& bundle,
                              const ReferenceActivationContext& context,
                              IssueCollector& issues) {
  const PlanningRequest& request = context.request;
  if (bundle.source_request_id != request.request_id) {
    issues.Add("source_request_id", "bundle_request_provenance_mismatch",
               "bundle source request must equal activation request");
  }
  if (bundle.platform_type != request.platform_type) {
    issues.Add("platform_type", "bundle_platform_provenance_mismatch",
               "bundle platform must equal activation request");
  }
  if (!request.request_time.clock_id.empty() &&
      ReferenceTimeOf(bundle.platform_reference).clock_id !=
          request.request_time.clock_id) {
    issues.Add("reference_time_origin.clock_id", "clock_id_mismatch",
               "reference and request must use the same clock");
  }
  if (bundle.validity.valid_from.clock_id !=
      ReferenceTimeOf(bundle.platform_reference).clock_id) {
    issues.Add("validity.valid_from.clock_id", "clock_id_mismatch",
               "validity and reference origin must use the same clock");
  }
  if (bundle.validity.valid_from.tick !=
      ReferenceTimeOf(bundle.platform_reference).tick) {
    issues.Add("validity.valid_from",
               "reference_time_provenance_mismatch",
               "valid_from must equal the inline reference time origin");
  }

  CheckActivationRegistryRoots(bundle, context, issues);
  CheckRegistryCapabilityBindings(context, issues);
  CheckBundleLearnedBinding(bundle, context, issues);

  std::vector<ContentRef> fixed_candidate_inputs;
  if (const auto* hopper =
          std::get_if<HopperReference>(&bundle.platform_reference);
      hopper != nullptr) {
    if (hopper->next_landing_region.frame_id != request.frame_id ||
        hopper->certified_flight_tube.frame_id != request.frame_id) {
      issues.Add("next_landing_region.frame_id",
                 "bundle_frame_provenance_mismatch",
                 "Hopper region and tube frame must match request");
    }
    if (const auto* request_state =
            std::get_if<HopperState>(&request.current_state);
        request_state != nullptr) {
      CheckRequestStateContainedInGroundHold(
          *request_state, hopper->ground_hold_anchor, issues);
    }
    CheckHopperActivationBindings(
        *hopper, context, fixed_candidate_inputs, issues);
    if (request.safety_capability) {
      if (const auto* capability =
              std::get_if<HopperCapability>(
                  &request.safety_capability->content);
          capability != nullptr) {
        CheckHopperCapabilityLimits(*hopper, *capability, issues);
      }
    }
  }

  const auto certificates = CollectCertificateRefs(bundle);
  for (const auto& [certificate_ref, field_path] : certificates) {
    CheckCertification(
        context.registry, certificate_ref, field_path, bundle,
        fixed_candidate_inputs, issues);
  }
}

void CheckResponseLearnedActivation(
    const PlanningResponse& response,
    const ReferenceActivationContext& context,
    IssueCollector& issues) {
  if (!response.new_reference_bundle.has_value()) {
    return;
  }
  const LearnedCostUsage usage =
      response.call_diagnostics.learned_cost_usage;
  const auto& diagnostic_ref =
      response.call_diagnostics.learned_cost_snapshot_ref;
  const auto& evidence_ref =
      response.new_reference_bundle->generation_evidence
          .learned_cost_snapshot_ref;
  const bool used =
      usage == LearnedCostUsage::kUsedBoundedSoftCost;
  if (usage == LearnedCostUsage::kDisabled) {
    if (diagnostic_ref.has_value() || evidence_ref.has_value()) {
      issues.Add("call_diagnostics.learned_cost_snapshot_ref",
                 "disabled_learned_cost_provenance_forbidden",
                 "DISABLED activation carries no learned-cost ref");
    }
    return;
  }
  if (!context.request.learned_cost_snapshot.has_value()) {
    issues.Add("call_diagnostics.learned_cost_snapshot_ref",
               "learned_snapshot_not_fixed_by_request",
               "USED/FALLBACK requires a request-fixed snapshot");
    return;
  }
  const ContentRef& request_ref =
      context.request.learned_cost_snapshot->snapshot_ref;
  if (!diagnostic_ref.has_value() ||
      *diagnostic_ref != request_ref) {
    issues.Add("call_diagnostics.learned_cost_snapshot_ref",
               "learned_snapshot_request_mismatch",
               "per-call learned usage must join the request snapshot");
  }
  if (used) {
    if (!evidence_ref.has_value() || *evidence_ref != request_ref) {
      issues.Add("generation_evidence.learned_cost_snapshot_ref",
                 "used_candidate_learned_snapshot_mismatch",
                 "activated USED candidate must copy the exact request "
                 "snapshot into generation evidence");
    }
  } else if (evidence_ref.has_value()) {
    issues.Add("generation_evidence.learned_cost_snapshot_ref",
               "fallback_candidate_forbids_learned_snapshot",
               "analytic fallback must not credit the learned snapshot");
  }
}

}  // namespace

bool ValidationReport::Contains(
    const std::string_view field_path,
    const std::string_view reason_code) const {
  return std::any_of(
      issues.begin(), issues.end(),
      [&](const ValidationIssue& issue) {
        return issue.field_path == field_path &&
               issue.reason_code == reason_code;
      });
}

ValidationReport SemanticValidator::Validate(
    const PlanningRequest& request) const {
  IssueCollector issues;
  CheckPlanningRequest(request, issues);
  return issues.Finish();
}

ValidationReport SemanticValidator::Validate(
    const PlanningResponse& response) const {
  IssueCollector issues;
  CheckPlanningResponse(response, issues);
  return issues.Finish();
}

ValidationReport SemanticValidator::Validate(
    const PlatformReference& reference) const {
  IssueCollector issues;
  CheckPlatformReference(reference, "", issues);
  return issues.Finish();
}

ValidationReport SemanticValidator::Validate(
    const ReferenceBundle& bundle) const {
  IssueCollector issues;
  CheckReferenceBundle(bundle, issues);
  return issues.Finish();
}

ValidationReport SemanticValidator::Validate(
    const SafetyCapabilityProfile& capability) const {
  IssueCollector issues;
  CheckCapabilityProfile(capability, issues);
  return issues.Finish();
}

ValidationReport SemanticValidator::Validate(
    const PlannerAlgorithmConfig& config) const {
  IssueCollector issues;
  CheckPlannerConfig(config, issues);
  return issues.Finish();
}

ValidationReport SemanticValidator::ValidateForActivation(
    const ReferenceBundle& bundle,
    const ReferenceActivationContext& context) const {
  IssueCollector issues;
  CheckReferenceBundle(bundle, issues);
  CheckReferenceActivation(bundle, context, issues);
  return issues.Finish();
}

ValidationReport SemanticValidator::ValidateForActivation(
    const PlanningResponse& response,
    const ReferenceActivationContext& context) const {
  IssueCollector issues;
  CheckPlanningResponse(response, issues);
  if (response.execution_directive !=
      ExecutionDirective::kActivateNewBundle) {
    issues.Add("execution_directive", "activation_directive_required",
               "activation validation requires ACTIVATE_NEW_BUNDLE");
  }
  if (response.request_id != context.request.request_id) {
    issues.Add("request_id", "activation_request_id_mismatch",
               "response request ID must equal activation request");
  }
  if (!response.response_time.clock_id.empty() &&
      response.response_time.clock_id !=
          context.request.request_time.clock_id) {
    issues.Add("response_time.clock_id", "clock_id_mismatch",
               "response and request must use the same clock");
  }
  if (response.new_reference_bundle.has_value()) {
    CheckReferenceActivation(
        *response.new_reference_bundle, context, issues);
  }
  CheckResponseLearnedActivation(response, context, issues);
  return issues.Finish();
}

}  // namespace lunar::planning::v3
