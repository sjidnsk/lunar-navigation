#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <grid_map_msgs/msg/grid_map.hpp>

#include "lunar_observed_map/map_types.hpp"
#include "lunar_observed_map/obstacle_classifier.hpp"
#include "lunar_observed_map/sparse_observed_map.hpp"

namespace lunar::observed_map {

inline constexpr std::array<std::size_t, 6U> kGlobalLevelFactors{
    1U, 2U, 4U, 8U, 16U, 20U};

struct MapPoint2D final {
  double x_m{};
  double y_m{};
};

struct ProductCell final {
  float elevation{};
  float valid_mask{};
  float obstacle{};
  float obstacle_height{};
  float observation_age_s{};
  float observation_quality{};
  float elevation_variance{};
  float obstacle_variance{};
  float observation_count{};
  float forbidden{};
};

struct GridMapProductConfig final {
  std::size_t local_axis_cells{320U};
  std::size_t target_axis_cells{256U};
  std::size_t maximum_axis_cells{4096U};
  std::size_t maximum_total_cells{1'048'576U};
  double maximum_global_axis_m{1024.0};
  std::size_t planning_boundary_cells{2U};
};

struct ProductRequest final {
  MapPoint2D robot_map;
  MapPoint2D robot_odom;
  MapPoint2D l0_origin_map;
  std::optional<MapPoint2D> goal_map;
  std::int64_t simulation_time_ns{};
};

struct MapProductResult final {
  std::optional<grid_map_msgs::msg::GridMap> message;
  std::string reason_code;
  std::size_t level_factor{};
  ObservedBounds l0_bounds;

  [[nodiscard]] bool ok() const noexcept {
    return message.has_value();
  }
};

class GridMapProducts final {
 public:
  explicit GridMapProducts(
      GridMapProductConfig config = {},
      ObstacleClassifier classifier = ObstacleClassifier{});

  [[nodiscard]] MapProductResult BuildLocal(
      const SparseObservedMap& map, const ProductRequest& request) const;
  [[nodiscard]] MapProductResult BuildGlobal(
      const SparseObservedMap& map, const ProductRequest& request) const;

  [[nodiscard]] std::optional<std::size_t> SelectGlobalLevel(
      std::size_t l0_width, std::size_t l0_height) const noexcept;
  [[nodiscard]] static ProductCell Aggregate(
      const ProductCell* children, std::size_t child_count,
      std::size_t expected_child_count);

 private:
  GridMapProductConfig config_;
  ObstacleClassifier classifier_;
};

}  // namespace lunar::observed_map
