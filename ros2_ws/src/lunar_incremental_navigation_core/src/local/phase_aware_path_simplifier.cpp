#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "local/grid_supercover.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] bool Interrupted(const SearchControl& control) {
  return control.canceled() || control.expired();
}

[[nodiscard]] bool Allowed(const RequestLocalPlanningView& view,
                           const GridIndex index, const StartPhase phase) {
  return view.Source(index) ==
         (phase == StartPhase::kStartPrefix
              ? LocalCellSource::kStartAssumedFree
              : LocalCellSource::kEvidenceFree);
}

// Local cell coordinates keep the distance comparison independent of world
// offsets and resolution. No physical clearance threshold is introduced.
struct Point {
  long double x, y;
};
struct Segment {
  Point a, b;
};

[[nodiscard]] long double PointDistanceSquared(Point p, Segment s) {
  const long double dx = s.b.x - s.a.x, dy = s.b.y - s.a.y;
  const long double length = dx * dx + dy * dy;
  const long double t = length > 0
      ? std::clamp(((p.x - s.a.x) * dx + (p.y - s.a.y) * dy) / length,
                   0.0L, 1.0L)
      : 0;
  const long double ex = p.x - s.a.x - t * dx;
  const long double ey = p.y - s.a.y - t * dy;
  return ex * ex + ey * ey;
}

// LOS has excluded intersections. For disjoint segments, the minimum distance
// is attained at an endpoint of at least one segment.
[[nodiscard]] long double DistanceSquared(Segment a, Segment b) {
  return std::min({PointDistanceSquared(a.a, b), PointDistanceSquared(a.b, b),
                   PointDistanceSquared(b.a, a), PointDistanceSquared(b.b, a)});
}

// Per-call geometry of the current phase's allowed-cell union, including the
// local-window boundary. This does not modify maps or their source semantics.
class PhaseGeometry final {
 public:
  PhaseGeometry(const RequestLocalPlanningView& view, StartPhase phase,
                const SearchControl& control)
      : view_(view), phase_(phase), control_(control) {}

  [[nodiscard]] bool Build() {
    const auto& geometry = view_.geometry();
    const auto lo = geometry.min_inclusive();
    const std::size_t width = geometry.width(), height = geometry.height();
    std::vector<unsigned char> allowed(geometry.CellCount());
    for (std::size_t y = 0; y < height; ++y) {
      if (Interrupted(control_)) return false;
      for (std::size_t x = 0; x < width; ++x) {
        allowed[y * width + x] = Allowed(
            view_, {lo.x + static_cast<std::int64_t>(x),
                    lo.y + static_cast<std::int64_t>(y)}, phase_);
      }
    }
    // Merge consecutive exposed cell faces into exact boundary segments.
    for (std::size_t y = 0; y <= height; ++y) {
      if (Interrupted(control_)) return false;
      std::size_t start = 0;
      bool active = false;
      for (std::size_t x = 0; x <= width; ++x) {
        const bool face = x < width &&
            ((y > 0 && allowed[(y - 1) * width + x]) !=
             (y < height && allowed[y * width + x]));
        if (face && !active) {
          start = x;
          active = true;
        }
        if (!face && active) {
          boundary_.push_back(
              {{static_cast<long double>(start), static_cast<long double>(y)},
               {static_cast<long double>(x), static_cast<long double>(y)}});
          active = false;
        }
      }
    }
    for (std::size_t x = 0; x <= width; ++x) {
      if (Interrupted(control_)) return false;
      std::size_t start = 0;
      bool active = false;
      for (std::size_t y = 0; y <= height; ++y) {
        const bool face = y < height &&
            ((x > 0 && allowed[y * width + x - 1]) !=
             (x < width && allowed[y * width + x]));
        if (face && !active) {
          start = y;
          active = true;
        }
        if (!face && active) {
          boundary_.push_back(
              {{static_cast<long double>(x), static_cast<long double>(start)},
               {static_cast<long double>(x), static_cast<long double>(y)}});
          active = false;
        }
      }
    }
    return !Interrupted(control_);
  }

  [[nodiscard]] bool Legal(const PathPoint& a, const PathPoint& b) const {
    return local::VisitSupercoverCells(
        view_.geometry(), a.pose.position_m, b.pose.position_m,
        [&](GridIndex index) {
          return !Interrupted(control_) && Allowed(view_, index, phase_);
        });
  }

  // Negative means illegal/interrupted. Original and replacement segments use
  // exactly the same evaluation; there are no shortcut boundary exceptions.
  [[nodiscard]] long double Evaluate(const PathPoint& a,
                                     const PathPoint& b) const {
    if (!Legal(a, b)) return -1;
    const Segment line{Convert(a.pose.position_m), Convert(b.pose.position_m)};
    long double minimum = std::numeric_limits<long double>::infinity();
    for (std::size_t i = 0; i < boundary_.size(); ++i) {
      if (i % 64 == 0 && Interrupted(control_)) return -1;
      const auto& edge = boundary_[i];
      const long double dx = std::max({
          0.0L, std::min(line.a.x, line.b.x) - std::max(edge.a.x, edge.b.x),
          std::min(edge.a.x, edge.b.x) - std::max(line.a.x, line.b.x)});
      const long double dy = std::max({
          0.0L, std::min(line.a.y, line.b.y) - std::max(edge.a.y, edge.b.y),
          std::min(edge.a.y, edge.b.y) - std::max(line.a.y, line.b.y)});
      if (dx * dx + dy * dy < minimum) {
        minimum = std::min(minimum, DistanceSquared(line, edge));
      }
    }
    return minimum;
  }

 private:
  [[nodiscard]] Point Convert(Vec3 p) const {
    const auto& geometry = view_.geometry();
    const auto origin = geometry.origin_m();
    const auto lo = geometry.min_inclusive();
    return {(static_cast<long double>(p.x) - origin.x) /
                geometry.resolution_m() - lo.x,
            (static_cast<long double>(p.y) - origin.y) /
                geometry.resolution_m() - lo.y};
  }

  const RequestLocalPlanningView& view_;
  StartPhase phase_;
  const SearchControl& control_;
  std::vector<Segment> boundary_;
};

[[nodiscard]] bool SameDirection(const PathPoint& a, const PathPoint& b,
                                 const PathPoint& c) {
  const long double dx = static_cast<long double>(b.pose.position_m.x) - a.pose.position_m.x;
  const long double dy = static_cast<long double>(b.pose.position_m.y) - a.pose.position_m.y;
  const long double ex = static_cast<long double>(c.pose.position_m.x) - b.pose.position_m.x;
  const long double ey = static_cast<long double>(c.pose.position_m.y) - b.pose.position_m.y;
  // Exact same-direction collinearity changes sampling, not geometry.
  return dx * ey == dy * ex && dx * ex + dy * ey > 0;
}

[[nodiscard]] bool SimplifyPhase(
    const RequestLocalPlanningView& view, std::span<const PathPoint> raw,
    std::vector<PathPoint>& output, const SearchControl& control) {
  std::vector<PathPoint> path;
  path.reserve(raw.size());
  for (const auto& point : raw) {
    if (Interrupted(control)) return false;
    if (path.size() >= 2 &&
        SameDirection(path[path.size() - 2], path.back(), point)) {
      path.back() = point;
    } else {
      path.push_back(point);
    }
  }
  PhaseGeometry geometry(view, raw.front().phase, control);
  if (path.size() < 3) {
    if (path.size() == 2 && !geometry.Legal(path[0], path[1])) return false;
    output.insert(output.end(), path.begin(), path.end());
    return true;
  }
  if (!geometry.Build()) return false;
  std::vector<long double> clearance(path.size() - 1), minimum(path.size());
  for (std::size_t i = 0; i < clearance.size(); ++i) {
    clearance[i] = geometry.Evaluate(path[i], path[i + 1]);
    if (clearance[i] < 0) return false;
  }
  output.push_back(path.front());
  std::size_t anchor = 0;
  while (anchor + 1 < path.size()) {
    if (Interrupted(control)) return false;
    minimum[anchor] = std::numeric_limits<long double>::infinity();
    for (std::size_t j = anchor + 1; j < path.size(); ++j) {
      minimum[j] = std::min(minimum[j - 1], clearance[j - 1]);
    }
    std::size_t selected = anchor + 1;
    for (std::size_t j = path.size() - 1; j > anchor + 1; --j) {
      const long double candidate = geometry.Evaluate(path[anchor], path[j]);
      if (Interrupted(control)) return false;
      // Numerical roundoff in squared cell units, not a physical margin.
      // Reference values always come from the original geometry, so repeated
      // shortcuts cannot accumulate a sequence of tolerance relaxations.
      const long double tolerance = 64 * std::numeric_limits<double>::epsilon() *
                                    std::max(1.0L, minimum[j]);
      if (candidate >= 0 && candidate + tolerance >= minimum[j]) {
        selected = j;
        break;
      }
    }
    output.push_back(path[selected]);
    anchor = selected;
  }
  return true;
}

}  // namespace

std::vector<PathPoint> SimplifyPhaseAwarePath(
    const RequestLocalPlanningView& view, std::span<const PathPoint> raw,
    const SearchControl& control) {
  if (Interrupted(control)) return {};
  if (raw.size() < 3) return {raw.begin(), raw.end()};
  std::vector<PathPoint> output;
  output.reserve(raw.size());
  for (std::size_t begin = 0; begin < raw.size();) {
    if (Interrupted(control)) return {};
    std::size_t end = begin + 1;
    while (end < raw.size() && raw[end].phase == raw[begin].phase) ++end;
    if (!SimplifyPhase(view, raw.subspan(begin, end - begin), output, control)) {
      return {};
    }
    begin = end;
  }
  return output;
}

std::vector<PathPoint> SimplifyPhaseAwarePath(
    const RequestLocalPlanningView& view, std::span<const PathPoint> raw) {
  return SimplifyPhaseAwarePath(
      view, raw, SearchControl{.deadline = SteadyClock::time_point::max()});
}

}  // namespace lunar::incremental_navigation
