#include "lunar_observed_map/grid_map_products.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>

namespace lunar::observed_map {
namespace {

constexpr std::array<const char*, 10U> kLayerNames{
    "elevation",
    "valid_mask",
    "obstacle",
    "obstacle_height",
    "observation_age_s",
    "observation_quality",
    "elevation_variance",
    "obstacle_variance",
    "observation_count",
    "forbidden",
};

struct GridCellIndexHash final {
  std::size_t operator()(const GridCellIndex& cell) const noexcept {
    return TileKeyHash{}(TileKey{cell.x, cell.y});
  }
};

MapProductResult Failure(std::string reason_code) {
  return MapProductResult{
      .message = std::nullopt,
      .reason_code = std::move(reason_code),
      .level_factor = 0U,
      .l0_bounds = {},
  };
}

std::optional<GridCellIndex> WorldToCell(
    const MapPoint2D point, const MapPoint2D l0_origin) noexcept {
  if (!std::isfinite(point.x_m) || !std::isfinite(point.y_m) ||
      !std::isfinite(l0_origin.x_m) ||
      !std::isfinite(l0_origin.y_m)) {
    return std::nullopt;
  }
  const double x =
      std::floor((point.x_m - l0_origin.x_m) / kL0ResolutionM);
  const double y =
      std::floor((point.y_m - l0_origin.y_m) / kL0ResolutionM);
  if (x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      x > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      y > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  return GridCellIndex{
      static_cast<std::int64_t>(x), static_cast<std::int64_t>(y)};
}

std::optional<std::size_t> InclusiveLength(
    const std::int64_t minimum, const std::int64_t maximum) noexcept {
  if (maximum < minimum) {
    return std::nullopt;
  }
  const auto difference = static_cast<std::uint64_t>(maximum) -
      static_cast<std::uint64_t>(minimum);
  if (difference == std::numeric_limits<std::uint64_t>::max() ||
      difference + 1U >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(difference + 1U);
}

std::int64_t SaturatingSubtract(
    const std::int64_t value, const std::size_t amount) noexcept {
  if (amount > static_cast<std::size_t>(
                   std::numeric_limits<std::int64_t>::max()) ||
      value < std::numeric_limits<std::int64_t>::min() +
              static_cast<std::int64_t>(amount)) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return value - static_cast<std::int64_t>(amount);
}

std::int64_t SaturatingAdd(
    const std::int64_t value, const std::size_t amount) noexcept {
  if (amount > static_cast<std::size_t>(
                   std::numeric_limits<std::int64_t>::max()) ||
      value > std::numeric_limits<std::int64_t>::max() -
              static_cast<std::int64_t>(amount)) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return value + static_cast<std::int64_t>(amount);
}

builtin_interfaces::msg::Time Stamp(const std::int64_t nanoseconds) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
  stamp.nanosec =
      static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
  return stamp;
}

std_msgs::msg::Float32MultiArray EncodeLayer(
    const std::vector<float>& row_major,
    const std::size_t width, const std::size_t height) {
  std_msgs::msg::Float32MultiArray output;
  output.layout.dim.resize(2U);
  output.layout.dim[0].label = "column_index";
  output.layout.dim[0].size = height;
  output.layout.dim[0].stride = width * height;
  output.layout.dim[1].label = "row_index";
  output.layout.dim[1].size = width;
  output.layout.dim[1].stride = width;
  output.layout.data_offset = 0U;
  output.data.resize(width * height);
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t physical_x = width - 1U - x;
      const std::size_t physical_y = height - 1U - y;
      output.data[physical_y * width + physical_x] =
          row_major[y * width + x];
    }
  }
  return output;
}

ProductCell ProductFromEvidence(
    const SparseObservedMap& map, const ObstacleClassifier& classifier,
    const GridCellIndex index, const std::int64_t simulation_time_ns) {
  const CellEvidence* evidence = map.Find(index);
  const CellClassification classification = classifier.Classify(map, index);
  ProductCell product;
  product.forbidden = classification.forbidden ? 1.0F : 0.0F;
  if (evidence == nullptr || !evidence->valid) {
    return product;
  }
  product.elevation = static_cast<float>(evidence->elevation_m);
  product.valid_mask = 1.0F;
  product.obstacle = classification.obstacle ? 1.0F : 0.0F;
  product.obstacle_height =
      static_cast<float>(classification.obstacle_height_m);
  product.observation_age_s =
      static_cast<float>(evidence->observation_age_s(simulation_time_ns));
  product.observation_quality = 1.0F;
  product.elevation_variance =
      static_cast<float>(evidence->elevation_variance());
  product.obstacle_variance =
      static_cast<float>(classification.obstacle_variance);
  const std::uint64_t count = std::min<std::uint64_t>(
      evidence->observation_count,
      std::numeric_limits<std::uint32_t>::max());
  product.observation_count = static_cast<float>(count);
  return product;
}

bool Traversable(
    const SparseObservedMap& map, const ObstacleClassifier& classifier,
    const GridCellIndex cell) {
  const CellEvidence* evidence = map.Find(cell);
  if (evidence == nullptr || !evidence->valid) {
    return false;
  }
  const CellClassification classification = classifier.Classify(map, cell);
  return !classification.obstacle && !classification.forbidden;
}

std::optional<ObservedBounds> ConnectedPathBounds(
    const SparseObservedMap& map, const ObstacleClassifier& classifier,
    const GridCellIndex start, const GridCellIndex goal) {
  if (!Traversable(map, classifier, start) ||
      !Traversable(map, classifier, goal)) {
    return std::nullopt;
  }
  std::deque<GridCellIndex> frontier;
  std::unordered_set<GridCellIndex, GridCellIndexHash> visited;
  std::unordered_map<
      GridCellIndex, GridCellIndex, GridCellIndexHash> parent;
  frontier.push_back(start);
  visited.insert(start);
  constexpr std::array<GridCellIndex, 4U> directions{
      GridCellIndex{1, 0}, GridCellIndex{-1, 0},
      GridCellIndex{0, 1}, GridCellIndex{0, -1}};
  bool found = start == goal;
  while (!frontier.empty() && !found) {
    const GridCellIndex current = frontier.front();
    frontier.pop_front();
    for (const GridCellIndex direction : directions) {
      if ((direction.x < 0 &&
           current.x == std::numeric_limits<std::int64_t>::min()) ||
          (direction.x > 0 &&
           current.x == std::numeric_limits<std::int64_t>::max()) ||
          (direction.y < 0 &&
           current.y == std::numeric_limits<std::int64_t>::min()) ||
          (direction.y > 0 &&
           current.y == std::numeric_limits<std::int64_t>::max())) {
        continue;
      }
      const GridCellIndex next{
          current.x + direction.x, current.y + direction.y};
      if (visited.contains(next) ||
          !Traversable(map, classifier, next)) {
        continue;
      }
      visited.insert(next);
      parent.emplace(next, current);
      if (next == goal) {
        found = true;
        break;
      }
      frontier.push_back(next);
    }
  }
  if (!found) {
    return std::nullopt;
  }

  ObservedBounds bounds{goal, goal};
  GridCellIndex current = goal;
  while (!(current == start)) {
    bounds.minimum.x = std::min(bounds.minimum.x, current.x);
    bounds.minimum.y = std::min(bounds.minimum.y, current.y);
    bounds.maximum.x = std::max(bounds.maximum.x, current.x);
    bounds.maximum.y = std::max(bounds.maximum.y, current.y);
    current = parent.at(current);
  }
  bounds.minimum.x = std::min(bounds.minimum.x, start.x);
  bounds.minimum.y = std::min(bounds.minimum.y, start.y);
  bounds.maximum.x = std::max(bounds.maximum.x, start.x);
  bounds.maximum.y = std::max(bounds.maximum.y, start.y);
  return bounds;
}

MapProductResult BuildMessage(
    const SparseObservedMap& map, const ObstacleClassifier& classifier,
    const ObservedBounds bounds, const std::size_t factor,
    const std::string& frame_id, const MapPoint2D l0_origin,
    const MapPoint2D frame_offset,
    const std::int64_t simulation_time_ns) {
  const auto l0_width = InclusiveLength(bounds.minimum.x, bounds.maximum.x);
  const auto l0_height = InclusiveLength(bounds.minimum.y, bounds.maximum.y);
  if (!l0_width || !l0_height || factor == 0U ||
      simulation_time_ns <= 0) {
    return Failure("MAP_PRODUCT_REQUEST_INVALID");
  }
  const std::size_t width = (*l0_width + factor - 1U) / factor;
  const std::size_t height = (*l0_height + factor - 1U) / factor;
  if (width == 0U || height == 0U ||
      width > std::numeric_limits<std::size_t>::max() / height ||
      width > std::numeric_limits<std::uint16_t>::max() ||
      height > std::numeric_limits<std::uint16_t>::max()) {
    return Failure("MAP_PRODUCT_GEOMETRY_INVALID");
  }

  std::array<std::vector<float>, 10U> layers;
  for (auto& layer : layers) {
    layer.resize(width * height, 0.0F);
  }
  std::vector<ProductCell> children;
  children.reserve(factor * factor);
  for (std::size_t output_y = 0U; output_y < height; ++output_y) {
    for (std::size_t output_x = 0U; output_x < width; ++output_x) {
      children.clear();
      for (std::size_t child_y = 0U; child_y < factor; ++child_y) {
        const std::size_t l0_y = output_y * factor + child_y;
        if (l0_y >= *l0_height) {
          continue;
        }
        for (std::size_t child_x = 0U; child_x < factor; ++child_x) {
          const std::size_t l0_x = output_x * factor + child_x;
          if (l0_x >= *l0_width) {
            continue;
          }
          children.push_back(ProductFromEvidence(
              map, classifier,
              GridCellIndex{
                  bounds.minimum.x + static_cast<std::int64_t>(l0_x),
                  bounds.minimum.y + static_cast<std::int64_t>(l0_y)},
              simulation_time_ns));
        }
      }
      const ProductCell product = factor == 1U
          ? children.front()
          : GridMapProducts::Aggregate(
                children.data(), children.size(), factor * factor);
      const std::size_t index = output_y * width + output_x;
      layers[0][index] = product.elevation;
      layers[1][index] = product.valid_mask;
      layers[2][index] = product.obstacle;
      layers[3][index] = product.obstacle_height;
      layers[4][index] = product.observation_age_s;
      layers[5][index] = product.observation_quality;
      layers[6][index] = product.elevation_variance;
      layers[7][index] = product.obstacle_variance;
      layers[8][index] = product.observation_count;
      layers[9][index] = product.forbidden;
    }
  }

  grid_map_msgs::msg::GridMap message;
  message.header.stamp = Stamp(simulation_time_ns);
  message.header.frame_id = frame_id;
  message.info.resolution =
      kL0ResolutionM * static_cast<double>(factor);
  message.info.length_x =
      static_cast<double>(width) * message.info.resolution;
  message.info.length_y =
      static_cast<double>(height) * message.info.resolution;
  const double origin_x =
      static_cast<double>(bounds.minimum.x) * kL0ResolutionM +
      l0_origin.x_m + frame_offset.x_m;
  const double origin_y =
      static_cast<double>(bounds.minimum.y) * kL0ResolutionM +
      l0_origin.y_m + frame_offset.y_m;
  message.info.pose.position.x = origin_x + message.info.length_x * 0.5;
  message.info.pose.position.y = origin_y + message.info.length_y * 0.5;
  message.info.pose.orientation.w = 1.0;
  message.layers.assign(kLayerNames.begin(), kLayerNames.end());
  message.basic_layers = {"elevation", "valid_mask"};
  message.data.reserve(layers.size());
  for (const auto& layer : layers) {
    message.data.push_back(EncodeLayer(layer, width, height));
  }
  message.outer_start_index = 0U;
  message.inner_start_index = 0U;
  return MapProductResult{
      .message = std::move(message),
      .reason_code = "ACCEPTED",
      .level_factor = factor,
      .l0_bounds = bounds,
  };
}

}  // namespace

GridMapProducts::GridMapProducts(
    GridMapProductConfig config, ObstacleClassifier classifier)
    : config_(config), classifier_(std::move(classifier)) {
  if (config_.local_axis_cells == 0U ||
      config_.target_axis_cells == 0U ||
      config_.maximum_axis_cells == 0U ||
      config_.maximum_total_cells == 0U ||
      !std::isfinite(config_.maximum_global_axis_m) ||
      config_.maximum_global_axis_m <= 0.0) {
    throw std::invalid_argument("GRID_MAP_PRODUCT_CONFIG_INVALID");
  }
}

MapProductResult GridMapProducts::BuildLocal(
    const SparseObservedMap& map, const ProductRequest& request) const {
  const auto robot = WorldToCell(
      request.robot_map, request.l0_origin_map);
  if (!robot || request.simulation_time_ns <= 0 ||
      !std::isfinite(request.robot_odom.x_m) ||
      !std::isfinite(request.robot_odom.y_m)) {
    return Failure("LOCAL_MAP_REQUEST_INVALID");
  }
  const std::size_t lower_half = config_.local_axis_cells / 2U;
  const std::size_t upper_half =
      config_.local_axis_cells - lower_half - 1U;
  const ObservedBounds bounds{
      .minimum = {
          SaturatingSubtract(robot->x, lower_half),
          SaturatingSubtract(robot->y, lower_half)},
      .maximum = {
          SaturatingAdd(robot->x, upper_half),
          SaturatingAdd(robot->y, upper_half)},
  };
  const MapPoint2D odom_from_map{
      request.robot_odom.x_m - request.robot_map.x_m,
      request.robot_odom.y_m - request.robot_map.y_m,
  };
  return BuildMessage(
      map, classifier_, bounds, 1U, "odom", request.l0_origin_map,
      odom_from_map,
      request.simulation_time_ns);
}

MapProductResult GridMapProducts::BuildGlobal(
    const SparseObservedMap& map, const ProductRequest& request) const {
  const auto robot = WorldToCell(
      request.robot_map, request.l0_origin_map);
  if (!robot || request.simulation_time_ns <= 0) {
    return Failure("GLOBAL_MAP_REQUEST_INVALID");
  }
  ObservedBounds bounds;
  if (request.goal_map.has_value()) {
    const auto goal = WorldToCell(
        *request.goal_map, request.l0_origin_map);
    if (!goal) {
      return Failure("GOAL_INVALID");
    }
    const CellEvidence* goal_evidence = map.Find(*goal);
    if (goal_evidence == nullptr || !goal_evidence->valid) {
      return Failure("GOAL_NOT_OBSERVED");
    }
    const CellClassification goal_classification =
        classifier_.Classify(map, *goal);
    if (goal_classification.obstacle || goal_classification.forbidden) {
      return Failure("GOAL_NOT_TRAVERSABLE");
    }
    const auto corridor =
        ConnectedPathBounds(map, classifier_, *robot, *goal);
    if (!corridor) {
      return Failure("GOAL_NOT_CONNECTED");
    }
    bounds = *corridor;
  } else {
    if (!Traversable(map, classifier_, *robot)) {
      return Failure("ROBOT_NOT_ON_OBSERVED_FREE_CELL");
    }
    const auto observed = map.observed_bounds();
    if (!observed) {
      return Failure("MAP_NOT_READY");
    }
    bounds = *observed;
  }

  bounds.minimum.x = SaturatingSubtract(
      bounds.minimum.x, config_.planning_boundary_cells);
  bounds.minimum.y = SaturatingSubtract(
      bounds.minimum.y, config_.planning_boundary_cells);
  bounds.maximum.x = SaturatingAdd(
      bounds.maximum.x, config_.planning_boundary_cells);
  bounds.maximum.y = SaturatingAdd(
      bounds.maximum.y, config_.planning_boundary_cells);
  const auto width = InclusiveLength(bounds.minimum.x, bounds.maximum.x);
  const auto height = InclusiveLength(bounds.minimum.y, bounds.maximum.y);
  if (!width || !height) {
    return Failure("GLOBAL_MAP_GEOMETRY_INVALID");
  }
  const auto factor = SelectGlobalLevel(*width, *height);
  if (!factor) {
    return Failure("GLOBAL_MAP_SCALE_UNSUPPORTED");
  }
  return BuildMessage(
      map, classifier_, bounds, *factor, "map",
      request.l0_origin_map, {},
      request.simulation_time_ns);
}

std::optional<std::size_t> GridMapProducts::SelectGlobalLevel(
    const std::size_t l0_width, const std::size_t l0_height) const noexcept {
  if (l0_width == 0U || l0_height == 0U ||
      static_cast<double>(l0_width) * kL0ResolutionM >
          config_.maximum_global_axis_m + 1.0e-9 ||
      static_cast<double>(l0_height) * kL0ResolutionM >
          config_.maximum_global_axis_m + 1.0e-9) {
    return std::nullopt;
  }
  const std::size_t target =
      std::min(config_.target_axis_cells, config_.maximum_axis_cells);
  for (const std::size_t factor : kGlobalLevelFactors) {
    const std::size_t width = (l0_width + factor - 1U) / factor;
    const std::size_t height = (l0_height + factor - 1U) / factor;
    if (width > target || height > target ||
        width > config_.maximum_axis_cells ||
        height > config_.maximum_axis_cells ||
        width > config_.maximum_total_cells / height) {
      continue;
    }
    return factor;
  }
  return std::nullopt;
}

ProductCell GridMapProducts::Aggregate(
    const ProductCell* children, const std::size_t child_count,
    const std::size_t expected_child_count) {
  if (children == nullptr || child_count == 0U ||
      expected_child_count == 0U || child_count > expected_child_count) {
    throw std::invalid_argument("MAP_AGGREGATION_INPUT_INVALID");
  }
  ProductCell parent;
  parent.valid_mask = child_count == expected_child_count ? 1.0F : 0.0F;
  parent.observation_quality = 1.0F;
  parent.observation_count =
      static_cast<float>(std::numeric_limits<std::uint32_t>::max());
  float elevation_sum = 0.0F;
  float maximum_elevation_variance = 0.0F;
  for (std::size_t index = 0U; index < child_count; ++index) {
    const ProductCell& child = children[index];
    parent.valid_mask = std::min(parent.valid_mask, child.valid_mask);
    parent.obstacle = std::max(parent.obstacle, child.obstacle);
    parent.forbidden = std::max(parent.forbidden, child.forbidden);
    parent.obstacle_height =
        std::max(parent.obstacle_height, child.obstacle_height);
    parent.observation_age_s =
        std::max(parent.observation_age_s, child.observation_age_s);
    parent.observation_quality =
        std::min(parent.observation_quality, child.observation_quality);
    parent.observation_count =
        std::min(parent.observation_count, child.observation_count);
    parent.obstacle_variance =
        std::max(parent.obstacle_variance, child.obstacle_variance);
    maximum_elevation_variance =
        std::max(maximum_elevation_variance, child.elevation_variance);
    elevation_sum += child.elevation;
  }
  if (child_count != expected_child_count) {
    parent.valid_mask = 0.0F;
    parent.forbidden = 1.0F;
    parent.observation_quality = 0.0F;
    parent.observation_count = 0.0F;
  }
  if (parent.valid_mask != 1.0F) {
    parent.elevation = 0.0F;
    parent.elevation_variance = maximum_elevation_variance;
    return parent;
  }
  parent.elevation = elevation_sum / static_cast<float>(child_count);
  float population_variance = 0.0F;
  for (std::size_t index = 0U; index < child_count; ++index) {
    const float delta = children[index].elevation - parent.elevation;
    population_variance += delta * delta;
  }
  population_variance /= static_cast<float>(child_count);
  parent.elevation_variance =
      maximum_elevation_variance + population_variance;
  return parent;
}

}  // namespace lunar::observed_map
