#include "lunar_pure_exploration_ros/task_map_marker_builder.hpp"

#include <array>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using lunar::pure_exploration::CellState;
using lunar::pure_exploration::TaskRaster;
using visualization_msgs::msg::Marker;

Marker CellMarker(const std::string_view frame_id, const char* ns,
                  const float channel) {
  Marker marker;
  marker.header.frame_id = std::string{frame_id};
  marker.ns = ns;
  marker.id = 0;
  marker.type = Marker::CUBE_LIST;
  marker.action = Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.color.r = channel;
  marker.color.g = channel;
  marker.color.b = channel;
  marker.color.a = 1.0F;
  return marker;
}

}  // namespace

visualization_msgs::msg::MarkerArray TaskMapMarkerBuilder::Build(
    const TaskRaster& raster, const std::string_view frame_id) const {
  if (frame_id.empty()) {
    throw std::invalid_argument{"task-map marker frame must not be empty"};
  }
  const double resolution = raster.geometry().resolution;
  std::array<Marker, 3U> markers{
      CellMarker(frame_id, "task_known", 176.0F / 255.0F),
      CellMarker(frame_id, "task_blocked", 0.0F),
      CellMarker(frame_id, "task_unknown", 74.0F / 255.0F)};
  for (auto& marker : markers) {
    marker.scale.x = resolution;
    marker.scale.y = resolution;
    marker.scale.z = 0.02;
    marker.pose.position.z = -0.01;
  }

  for (const auto index : raster.task_cells()) {
    const auto center = raster.CellCenter(index);
    geometry_msgs::msg::Point point;
    point.x = center.x;
    point.y = center.y;
    Marker* destination = nullptr;
    switch (raster.Classify(index)) {
      case CellState::kFree:
        destination = &markers[0U];
        break;
      case CellState::kOccupied:
        destination = &markers[1U];
        break;
      case CellState::kUnknown:
      case CellState::kOutsideMap:
        destination = &markers[2U];
        break;
      case CellState::kOutsideTask:
        continue;
    }
    destination->points.push_back(point);
  }

  visualization_msgs::msg::MarkerArray result;
  result.markers.assign(markers.begin(), markers.end());
  return result;
}

}  // namespace lunar::pure_exploration_ros
