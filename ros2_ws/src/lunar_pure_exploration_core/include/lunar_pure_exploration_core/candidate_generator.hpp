#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lunar_pure_exploration_core/frontier_detector.hpp"

namespace lunar::pure_exploration {

struct PlatformGeometry {
  std::string platform_id;
  std::string platform_type;
  std::string base_frame_id;
  std::vector<Vec2> footprint_vertices;
  double minimum_clearance_m;
};

struct CandidateParameters {
  std::array<double, 5> yaw_offsets_rad;
};

struct CandidateKey {
  std::int64_t x_mm;
  std::int64_t y_mm;
  std::int64_t yaw_tenth_deg;
  auto operator<=>(const CandidateKey&) const = default;
};

struct CandidateView {
  std::uint64_t id;
  std::uint64_t frontier_id;
  std::size_t frontier_index;
  CandidateKey key;
  Pose2 pose;
  double frontier_distance_m;
  std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key{};
};

using CandidatePositionAcceptance = std::function<bool(Pose2)>;

class CandidateGenerator {
 public:
  struct Limits {
    std::size_t maximum_position_probes;
    std::size_t maximum_candidate_views;
    std::size_t maximum_collision_work_units;
  };

  CandidateGenerator(PlatformGeometry platform,
                     CandidateParameters parameters, Limits limits);
  std::vector<CandidateView> Generate(
      const TaskRaster& raster,
      std::span<const FrontierCluster> frontiers,
      const CandidatePositionAcceptance& accept_position = {}) const;

  double platform_length_m() const;
  double platform_width_m() const;
  double footprint_circumscribed_radius_m() const;
  double minimum_spacing_m(double resolution_m) const;
  double minimum_standoff_m() const;
  double maximum_extra_search_m() const;
  std::size_t maximum_search_step(double resolution_m) const;

 private:
  PlatformGeometry platform_;
  CandidateParameters parameters_;
  Limits limits_;
  double platform_length_m_;
  double platform_width_m_;
  double footprint_circumscribed_radius_m_;
};

}  // namespace lunar::pure_exploration
