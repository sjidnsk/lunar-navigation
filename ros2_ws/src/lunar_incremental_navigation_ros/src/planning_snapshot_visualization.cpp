#include "lunar_incremental_navigation_ros/planning_snapshot_visualization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::incremental_navigation_ros {
namespace {

namespace core = lunar::incremental_navigation;

struct CellBounds final {
  core::GridIndex minimum;
  core::GridIndex maximum;
};

[[nodiscard]] nav_msgs::msg::OccupancyGrid MakeGrid(
    const core::SparseGridGeometry& geometry, const CellBounds bounds) {
  nav_msgs::msg::OccupancyGrid output;
  const core::Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  output.header.frame_id = geometry.frame_id();
  output.info.resolution = static_cast<float>(resolution);
  output.info.width = static_cast<std::uint32_t>(
      bounds.maximum.x - bounds.minimum.x + 1);
  output.info.height = static_cast<std::uint32_t>(
      bounds.maximum.y - bounds.minimum.y + 1);
  output.info.origin.position.x = std::fma(
      static_cast<double>(bounds.minimum.x), resolution, origin.x);
  output.info.origin.position.y = std::fma(
      static_cast<double>(bounds.minimum.y), resolution, origin.y);
  output.info.origin.position.z = origin.z;
  output.info.origin.orientation.w = 1.0;
  output.data.assign(static_cast<std::size_t>(output.info.width) *
                         output.info.height,
                     -1);
  return output;
}

[[nodiscard]] bool IsValidWindow(const PlanningVisualizationWindow& window) {
  return std::isfinite(window.center_map_m.x) &&
         std::isfinite(window.center_map_m.y) && std::isfinite(window.length_m) &&
         window.length_m > 0.0;
}

[[nodiscard]] CellBounds FullBounds(const core::SparseGridGeometry& geometry) {
  return {.minimum = geometry.min_inclusive(),
          .maximum = {.x = geometry.max_exclusive().x - 1,
                      .y = geometry.max_exclusive().y - 1}};
}

[[nodiscard]] CellBounds WindowBounds(
    const core::SparseGridGeometry& geometry,
    const PlanningVisualizationWindow& window) {
  const core::Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  const double half_length = window.length_m / 2.0;
  const auto first_center_index = [resolution](const double lower,
                                               const double origin_coordinate) {
    return static_cast<std::int64_t>(std::ceil(
        (lower - origin_coordinate) / resolution - 0.5));
  };
  const auto last_center_index = [resolution](const double upper,
                                              const double origin_coordinate) {
    return static_cast<std::int64_t>(std::floor(
        (upper - origin_coordinate) / resolution - 0.5));
  };
  const core::GridIndex minimum{
      .x = std::max(geometry.min_inclusive().x,
                    first_center_index(window.center_map_m.x - half_length,
                                       origin.x)),
      .y = std::max(geometry.min_inclusive().y,
                    first_center_index(window.center_map_m.y - half_length,
                                       origin.y))};
  const core::GridIndex maximum{
      .x = std::min(geometry.max_exclusive().x - 1,
                    last_center_index(window.center_map_m.x + half_length,
                                      origin.x)),
      .y = std::min(geometry.max_exclusive().y - 1,
                    last_center_index(window.center_map_m.y + half_length,
                                      origin.y))};
  return {.minimum = minimum, .maximum = maximum};
}

[[nodiscard]] bool HasCells(const CellBounds& bounds) noexcept {
  return bounds.minimum.x <= bounds.maximum.x &&
         bounds.minimum.y <= bounds.maximum.y;
}

[[nodiscard]] std::int8_t EncodeFine(const core::FineCellState state) {
  switch (state) {
    case core::FineCellState::kUnknown:
      return -1;
    case core::FineCellState::kFree:
      return 0;
    case core::FineCellState::kBlocked:
      return 100;
  }
  return -1;
}

[[nodiscard]] std::int8_t EncodeGuidance(
    const core::GuidanceCellState state) {
  switch (state) {
    case core::GuidanceCellState::kUnknown:
      return -1;
    case core::GuidanceCellState::kCandidate:
      return 0;
    case core::GuidanceCellState::kProvenBlocked:
      return 100;
  }
  return -1;
}

[[nodiscard]] std::int8_t EncodeCost(const double cost,
                                     const double display_max) {
  if (!std::isfinite(display_max) || display_max <= 0.0) {
    return 0;
  }
  const double scaled = std::clamp(cost / display_max * 100.0, 0.0, 100.0);
  return static_cast<std::int8_t>(std::lround(scaled));
}

[[nodiscard]] std::int8_t EncodeExecutionRisk(const core::FineCellState state,
                                               const double cost,
                                               const double display_max) {
  if (state == core::FineCellState::kUnknown) return -1;
  if (state == core::FineCellState::kBlocked) return 100;
  if (!std::isfinite(display_max) || display_max <= 0.0) return 0;
  const double scaled = std::clamp(cost / display_max * 99.0, 0.0, 99.0);
  return static_cast<std::int8_t>(std::lround(scaled));
}

[[nodiscard]] geometry_msgs::msg::Point CellCenter(
    const core::SparseGridGeometry& geometry, const core::GridIndex index) {
  const core::Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  geometry_msgs::msg::Point point;
  point.x = std::fma(static_cast<double>(index.x) + 0.5, resolution, origin.x);
  point.y = std::fma(static_cast<double>(index.y) + 0.5, resolution, origin.y);
  return point;
}

[[nodiscard]] std_msgs::msg::ColorRGBA Color(const float red, const float green,
                                              const float blue, const float alpha) {
  std_msgs::msg::ColorRGBA color;
  color.r = red;
  color.g = green;
  color.b = blue;
  color.a = alpha;
  return color;
}

[[nodiscard]] visualization_msgs::msg::Marker MakeCells(
    const core::SparseGridGeometry& geometry, const std::string& name,
    const int id, const double z) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = geometry.frame_id();
  marker.ns = name;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = geometry.resolution_m();
  marker.scale.y = geometry.resolution_m();
  marker.scale.z = z;
  return marker;
}

[[nodiscard]] visualization_msgs::msg::Marker MakeWindowOutline(
    const core::SparseGridGeometry& geometry, const CellBounds bounds) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = geometry.frame_id();
  marker.ns = "local_evidence_window";
  marker.id = 2;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.z = 0.055;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.035;
  marker.color = Color(0.796F, 0.835F, 0.882F, 0.82F);  // #CBD5E1

  const core::Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  const double lower_x = std::fma(static_cast<double>(bounds.minimum.x),
                                  resolution, origin.x);
  const double lower_y = std::fma(static_cast<double>(bounds.minimum.y),
                                  resolution, origin.y);
  const double upper_x = std::fma(
      static_cast<double>(bounds.maximum.x + 1), resolution, origin.x);
  const double upper_y = std::fma(
      static_cast<double>(bounds.maximum.y + 1), resolution, origin.y);
  const auto append_point = [&marker](const double x, const double y) {
    geometry_msgs::msg::Point point;
    point.x = x;
    point.y = y;
    marker.points.push_back(point);
  };
  marker.points.reserve(5U);
  append_point(lower_x, lower_y);
  append_point(upper_x, lower_y);
  append_point(upper_x, upper_y);
  append_point(lower_x, upper_y);
  append_point(lower_x, lower_y);
  return marker;
}

[[nodiscard]] std_msgs::msg::ColorRGBA FineColor(
    const core::FineCellState state, const double cost,
    const double display_max) {
  // UNKNOWN is a light local-evidence haze rather than an opaque mask.  The
  // enclosing local_evidence_window marks its extent, while the global context
  // remains readable below it.
  if (state == core::FineCellState::kUnknown) {
    return Color(0.337F, 0.380F, 0.420F, 0.18F);  // #56616B slate
  }
  if (state == core::FineCellState::kBlocked) {
    return Color(0.773F, 0.188F, 0.188F, 0.98F);  // #C53030 vermilion
  }
  const float ratio = !std::isfinite(display_max) || display_max <= 0.0
      ? 0.0F
      : static_cast<float>(std::clamp(cost / display_max, 0.0, 1.0));
  // Confirmed FREE cells progress from teal (#007C91) to amber (#D69E2E).
  // This keeps local certainty distinct from the blue-grey global candidates.
  return Color(0.839F * ratio, 0.486F + 0.134F * ratio,
               0.569F - 0.389F * ratio, 0.95F);
}

[[nodiscard]] visualization_msgs::msg::Marker MakeCircle(
    const std::string& marker_namespace, const core::Pose2& anchor,
    const double radius, const int id, const std::string& frame) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame;
  marker.ns = marker_namespace;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = anchor.position_m.x;
  marker.pose.position.y = anchor.position_m.y;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.04;
  if (marker_namespace == "hard_radius") {
    marker.color = Color(0.886F, 0.910F, 0.941F, 1.0F);  // #E2E8F0
  } else {
    marker.color = Color(0.502F, 0.353F, 0.835F, 1.0F);  // #805AD5
  }
  constexpr std::size_t kSegments = 32U;
  marker.points.reserve(kSegments + 1U);
  for (std::size_t segment = 0U; segment <= kSegments; ++segment) {
    const double angle = 2.0 * std::numbers::pi *
        static_cast<double>(segment) / static_cast<double>(kSegments);
    geometry_msgs::msg::Point point;
    point.x = radius * std::cos(angle);
    point.y = radius * std::sin(angle);
    marker.points.push_back(point);
  }
  return marker;
}

}  // namespace

FineVisualization ProjectFineVisualization(
    const core::FineTraversabilitySnapshot& snapshot,
    const PlanningVisualizationWindow window, const double cost_display_max) {
  const core::SparseGridGeometry& geometry = snapshot.geometry();
  if (!IsValidWindow(window)) {
    FineVisualization output;
    return output;
  }
  const CellBounds bounds = WindowBounds(geometry, window);
  if (!HasCells(bounds)) {
    FineVisualization output;
    return output;
  }

  FineVisualization output{.state = MakeGrid(geometry, bounds),
                           .cost = MakeGrid(geometry, bounds),
                           .risk = MakeGrid(geometry, bounds)};
  for (std::uint32_t y = 0U; y < output.state.info.height; ++y) {
    for (std::uint32_t x = 0U; x < output.state.info.width; ++x) {
      const core::GridIndex index{
          .x = bounds.minimum.x + static_cast<std::int64_t>(x),
          .y = bounds.minimum.y + static_cast<std::int64_t>(y)};
      const std::size_t offset = static_cast<std::size_t>(y) *
                                     output.state.info.width + x;
      const core::FineCellState state = snapshot.State(index);
      const double cost = snapshot.TraversalCost(index);
      output.state.data[offset] = EncodeFine(state);
      output.cost.data[offset] = EncodeCost(cost, cost_display_max);
      output.risk.data[offset] =
          EncodeExecutionRisk(state, cost, cost_display_max);
    }
  }
  return output;
}

nav_msgs::msg::OccupancyGrid ProjectGuidanceVisualization(
    const core::GlobalGuidanceSnapshot& snapshot) {
  const CellBounds bounds = FullBounds(snapshot.geometry());
  nav_msgs::msg::OccupancyGrid output = MakeGrid(snapshot.geometry(), bounds);
  for (std::uint32_t y = 0U; y < output.info.height; ++y) {
    for (std::uint32_t x = 0U; x < output.info.width; ++x) {
      const core::GridIndex index{
          .x = bounds.minimum.x + static_cast<std::int64_t>(x),
          .y = bounds.minimum.y + static_cast<std::int64_t>(y)};
      output.data[static_cast<std::size_t>(y) * output.info.width + x] =
          EncodeGuidance(snapshot.State(index));
    }
  }
  return output;
}

visualization_msgs::msg::MarkerArray ProjectTraversabilityVisualization(
    const core::FineTraversabilitySnapshot& fine,
    const PlanningVisualizationWindow window, const double cost_display_max,
    const core::GlobalGuidanceSnapshot* const guidance) {
  visualization_msgs::msg::MarkerArray output;
  if (guidance) {
    const auto& geometry = guidance->geometry();
    auto overview = MakeCells(geometry, "traversability_overview", 0, 0.015);
    const CellBounds bounds = FullBounds(geometry);
    for (std::int64_t y = bounds.minimum.y; y <= bounds.maximum.y; ++y) {
      for (std::int64_t x = bounds.minimum.x; x <= bounds.maximum.x; ++x) {
        const core::GridIndex index{.x = x, .y = y};
        overview.points.push_back(CellCenter(geometry, index));
        switch (guidance->State(index)) {
          case core::GuidanceCellState::kUnknown:
            overview.colors.push_back(Color(0.294F, 0.333F, 0.388F, 0.32F));
            break;
          case core::GuidanceCellState::kCandidate:
            // A candidate is only a coarse global hint, never local FREE.
            overview.colors.push_back(Color(0.471F, 0.565F, 0.612F, 0.50F));
            break;
          case core::GuidanceCellState::kProvenBlocked:
            overview.colors.push_back(Color(0.545F, 0.227F, 0.227F, 0.65F));
            break;
        }
      }
    }
    output.markers.push_back(std::move(overview));
  }
  if (!IsValidWindow(window)) return output;
  const auto& geometry = fine.geometry();
  const CellBounds bounds = WindowBounds(geometry, window);
  if (!HasCells(bounds)) return output;
  auto unknown = MakeCells(geometry, "local_unknown_context", 1, 0.035);
  auto detail = MakeCells(geometry, "execution_risk_detail", 2, 0.045);
  for (std::int64_t y = bounds.minimum.y; y <= bounds.maximum.y; ++y) {
    for (std::int64_t x = bounds.minimum.x; x <= bounds.maximum.x; ++x) {
      const core::GridIndex index{.x = x, .y = y};
      const core::FineCellState state = fine.State(index);
      if (state == core::FineCellState::kUnknown) {
        unknown.points.push_back(CellCenter(geometry, index));
        unknown.colors.push_back(FineColor(state, 0.0, cost_display_max));
      } else {
        detail.points.push_back(CellCenter(geometry, index));
        detail.colors.push_back(
            FineColor(state, fine.TraversalCost(index), cost_display_max));
      }
    }
  }
  output.markers.push_back(std::move(unknown));
  output.markers.push_back(std::move(detail));
  output.markers.push_back(MakeWindowOutline(geometry, bounds));
  return output;
}

visualization_msgs::msg::MarkerArray ProjectStartPatchVisualization(
    const core::RequestLocalPlanningView& view,
    const double hard_inflation_radius_m) {
  const core::SparseGridGeometry& geometry = view.geometry();
  visualization_msgs::msg::MarkerArray output;
  visualization_msgs::msg::Marker assumed_unknown;
  assumed_unknown.header.frame_id = geometry.frame_id();
  assumed_unknown.ns = "start_patch_assumed_unknown";
  assumed_unknown.id = 0;
  assumed_unknown.type = visualization_msgs::msg::Marker::CUBE_LIST;
  assumed_unknown.action = visualization_msgs::msg::Marker::ADD;
  assumed_unknown.pose.orientation.w = 1.0;
  assumed_unknown.scale.x = geometry.resolution_m();
  assumed_unknown.scale.y = geometry.resolution_m();
  assumed_unknown.scale.z = 0.03;
  // Purple deliberately reserves the request-local, assumed-free exception.
  assumed_unknown.color = Color(0.502F, 0.353F, 0.835F, 0.85F);  // #805AD5
  for (const core::LocalCellOverride& cell : view.overrides()) {
    if (view.base()->State(cell.index) == core::FineCellState::kUnknown &&
        view.Source(cell.index) == core::LocalCellSource::kStartAssumedFree) {
      assumed_unknown.points.push_back(CellCenter(geometry, cell.index));
    }
  }
  output.markers.push_back(std::move(assumed_unknown));
  output.markers.push_back(MakeCircle(
      "hard_radius", view.patch_anchor(),
      std::max(0.0, hard_inflation_radius_m), 1, geometry.frame_id()));
  output.markers.push_back(MakeCircle("start_patch_radius", view.patch_anchor(),
                                      view.start_patch_radius_m(), 2, geometry.frame_id()));
  return output;
}

}  // namespace lunar::incremental_navigation_ros
