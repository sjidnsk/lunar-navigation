#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "lunar_incremental_navigation_core/elevation_map.hpp"

namespace lunar::incremental_navigation::local {

[[nodiscard]] inline std::optional<GridIndex> SupercoverWorldToCell(
    const SparseGridGeometry& geometry, const Vec3 point) noexcept {
  if (!geometry.valid() || !std::isfinite(point.x) ||
      !std::isfinite(point.y)) {
    return std::nullopt;
  }
  const Vec3 origin = geometry.origin_m();
  const double x = std::floor((point.x - origin.x) / geometry.resolution_m());
  const double y = std::floor((point.y - origin.y) / geometry.resolution_m());
  if (!std::isfinite(x) || !std::isfinite(y) ||
      x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      x >= static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      y >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  const GridIndex index{.x = static_cast<std::int64_t>(x),
                        .y = static_cast<std::int64_t>(y)};
  return geometry.Contains(index) ? std::optional<GridIndex>(index)
                                  : std::nullopt;
}

// Closed-segment coverage: every intersected cell contains an endpoint or a
// grid-line crossing. Enumerate those contacts directly instead of accumulating
// DDA event times (whose rounding depended on direction and subdivision).
// Duplicate visits are permitted; callers may stop at any contact. O(dx+dy)
// grid crossings, constant memory. One scaled tolerance handles all contacts.
template <typename Visitor>
[[nodiscard]] bool VisitSupercoverCells(
    const SparseGridGeometry& geometry, Vec3 from, Vec3 to, Visitor&& visitor) {
  if (!SupercoverWorldToCell(geometry, from) ||
      !SupercoverWorldToCell(geometry, to)) return false;
  // Canonical arithmetic makes reversal use exactly the same calculations.
  if (from.x > to.x || (from.x == to.x && from.y > to.y)) std::swap(from,to);
  const Vec3 origin=geometry.origin_m();
  const long double resolution=geometry.resolution_m();
  const long double x0=(static_cast<long double>(from.x)-origin.x)/resolution;
  const long double y0=(static_cast<long double>(from.y)-origin.y)/resolution;
  const long double x1=(static_cast<long double>(to.x)-origin.x)/resolution;
  const long double y1=(static_cast<long double>(to.y)-origin.y)/resolution;
  const auto tolerance=[](long double value) {
    return 64*std::numeric_limits<double>::epsilon()*std::max(1.0L,std::abs(value));
  };
  const auto visit_point=[&](long double x,long double y) {
    const auto limits=[&](long double coordinate) {
      const long double nearest=std::round(coordinate);
      return std::abs(coordinate-nearest) <= tolerance(coordinate)
          ? std::pair{nearest-1,nearest}
          : std::pair{std::floor(coordinate),std::floor(coordinate)};
    };
    const auto [xmin,xmax]=limits(x);
    const auto [ymin,ymax]=limits(y);
    constexpr auto minimum=std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum=std::numeric_limits<std::int64_t>::max();
    if (xmin < static_cast<long double>(minimum) || xmax > static_cast<long double>(maximum) ||
        ymin < static_cast<long double>(minimum) || ymax > static_cast<long double>(maximum)) return false;
    const auto ixmin=static_cast<std::int64_t>(xmin), ixmax=static_cast<std::int64_t>(xmax);
    const auto iymin=static_cast<std::int64_t>(ymin), iymax=static_cast<std::int64_t>(ymax);
    for (auto ix=ixmin;;++ix) {
      for (auto iy=iymin;;++iy) {
        const GridIndex cell{ix,iy};
        if (!geometry.Contains(cell) || !visitor(cell)) return false;
        if (iy==iymax) break;
      }
      if (ix==ixmax) break;
    }
    return true;
  };
  if (!visit_point(x0,y0) || !visit_point(x1,y1)) return false;
  const auto crossings=[&](long double a,long double b,long double c,long double d,bool vertical) {
    if (a==b) return true;
    const long double lower=std::min(a,b), upper=std::max(a,b);
    const auto first=std::ceil(lower), last=std::floor(upper);
    if (first>last) return true;
    const auto begin=static_cast<std::int64_t>(first), end=static_cast<std::int64_t>(last);
    for (auto index=begin;;++index) {
      const long double boundary=index;
      const long double t=(boundary-a)/(b-a);
      const long double other=std::lerp(c,d,t);
      if (!(vertical ? visit_point(boundary,other) : visit_point(other,boundary))) return false;
      if (index==end) break;
    }
    return true;
  };
  return crossings(x0,x1,y0,y1,true) && crossings(y0,y1,x0,x1,false);
}

}  // namespace lunar::incremental_navigation::local
