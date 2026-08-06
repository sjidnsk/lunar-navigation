#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hierarchical/landing_support_field.hpp"
#include "lunar_planner_core/types/geometry.hpp"

namespace lunar::planning::hierarchical {

class LandingSpatialIndex final {
public:
  LandingSpatialIndex(const LandingSupportField &field, double maximum_reach_m,
                      std::stop_token stop_token = {});

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] std::string_view reason_code() const noexcept;
  [[nodiscard]] std::vector<LandingNodeId>
  Query(Vec2 source_m, std::stop_token stop_token = {}) const;
  [[nodiscard]] std::size_t EstimatedWorkMemoryBytes() const noexcept;

private:
  using BucketCoordinate = std::pair<std::int64_t, std::int64_t>;

  [[nodiscard]] BucketCoordinate BucketFor(Vec2 point_m) const noexcept;

  std::shared_ptr<const shared::MapSnapshot> map_;
  double maximum_reach_m_{};
  double bucket_width_m_{};
  std::map<BucketCoordinate, std::vector<LandingNodeId>> buckets_;
  std::string reason_code_;
};

} // namespace lunar::planning::hierarchical
