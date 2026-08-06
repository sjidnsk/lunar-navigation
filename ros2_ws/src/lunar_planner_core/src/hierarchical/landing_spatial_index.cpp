#include "hierarchical/landing_spatial_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stop_token>
#include <utility>
#include <vector>

namespace lunar::planning::hierarchical {
namespace {

constexpr double kTolerance = 1.0e-9;

[[nodiscard]] bool Finite(const Vec2 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

} // namespace

LandingSpatialIndex::LandingSpatialIndex(const LandingSupportField &field,
                                         const double maximum_reach_m,
                                         const std::stop_token stop_token)
    : map_(field.SourceMap()), maximum_reach_m_(maximum_reach_m),
      bucket_width_m_(map_ == nullptr
                          ? maximum_reach_m
                          : std::max(maximum_reach_m, map_->resolution_m())) {
  if (stop_token.stop_requested()) {
    reason_code_ = "REQUEST_CANCELED";
    return;
  }
  if (map_ == nullptr || !std::isfinite(maximum_reach_m_) ||
      maximum_reach_m_ <= 0.0 || !std::isfinite(bucket_width_m_) ||
      bucket_width_m_ <= 0.0) {
    reason_code_ = "HOPPER_LANDING_SPATIAL_INDEX_INVALID";
    return;
  }
  for (const LandingNodeId id : field.SafeCenterIds()) {
    if (stop_token.stop_requested()) {
      buckets_.clear();
      reason_code_ = "REQUEST_CANCELED";
      return;
    }
    if (id >= map_->cell_count()) {
      buckets_.clear();
      reason_code_ = "HOPPER_LANDING_SPATIAL_INDEX_INVALID";
      return;
    }
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(id % map_->width()),
        .y = static_cast<std::int32_t>(id / map_->width()),
    };
    const Vec3 center = map_->CellCenter(cell);
    buckets_[BucketFor({center.x, center.y})].push_back(id);
  }
}

bool LandingSpatialIndex::ok() const noexcept {
  return map_ != nullptr && reason_code_.empty();
}

std::string_view LandingSpatialIndex::reason_code() const noexcept {
  return reason_code_;
}

LandingSpatialIndex::BucketCoordinate
LandingSpatialIndex::BucketFor(const Vec2 point_m) const noexcept {
  const auto coordinate = [&](const double value, const double origin) {
    const double bucket = std::floor((value - origin) / bucket_width_m_);
    if (bucket <=
        static_cast<double>(std::numeric_limits<std::int64_t>::min())) {
      return std::numeric_limits<std::int64_t>::min();
    }
    if (bucket >=
        static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(bucket);
  };
  return {
      coordinate(point_m.x, map_->origin_m().x),
      coordinate(point_m.y, map_->origin_m().y),
  };
}

std::vector<LandingNodeId>
LandingSpatialIndex::Query(const Vec2 source_m,
                           const std::stop_token stop_token) const {
  std::vector<LandingNodeId> result;
  if (!ok() || !Finite(source_m) || stop_token.stop_requested()) {
    return result;
  }
  const BucketCoordinate minimum =
      BucketFor({source_m.x - maximum_reach_m_, source_m.y - maximum_reach_m_});
  const BucketCoordinate maximum =
      BucketFor({source_m.x + maximum_reach_m_, source_m.y + maximum_reach_m_});
  for (std::int64_t bucket_y = minimum.second; bucket_y <= maximum.second;
       ++bucket_y) {
    if (stop_token.stop_requested()) {
      return {};
    }
    for (std::int64_t bucket_x = minimum.first; bucket_x <= maximum.first;
         ++bucket_x) {
      const auto found = buckets_.find({bucket_x, bucket_y});
      if (found == buckets_.end()) {
        continue;
      }
      for (const LandingNodeId id : found->second) {
        if (stop_token.stop_requested()) {
          return {};
        }
        const shared::GridCell cell{
            .x = static_cast<std::int32_t>(id % map_->width()),
            .y = static_cast<std::int32_t>(id / map_->width()),
        };
        const Vec3 center = map_->CellCenter(cell);
        if (std::hypot(center.x - source_m.x, center.y - source_m.y) <=
            maximum_reach_m_ + kTolerance) {
          result.push_back(id);
        }
      }
    }
  }
  std::ranges::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::size_t LandingSpatialIndex::EstimatedWorkMemoryBytes() const noexcept {
  std::size_t bytes = 0U;
  for (const auto &[coordinate, ids] : buckets_) {
    static_cast<void>(coordinate);
    bytes += sizeof(BucketCoordinate) + ids.capacity() * sizeof(LandingNodeId);
  }
  return bytes;
}

} // namespace lunar::planning::hierarchical
