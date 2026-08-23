#include "lunar_pure_exploration_ros/marker_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <geometry_msgs/msg/quaternion.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using lunar::pure_exploration::CandidateView;
using lunar::pure_exploration::FrontierCluster;
using visualization_msgs::msg::Marker;

int MarkerId(const std::size_t index) {
  if (index > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{"marker index exceeds ROS marker id range"};
  }
  return static_cast<int>(index);
}

Marker Deletion(const char* marker_namespace, const std::size_t index) {
  Marker marker;
  marker.ns = marker_namespace;
  marker.id = MarkerId(index);
  marker.action = Marker::DELETE;
  return marker;
}

bool IsSelected(const CandidateView& candidate,
                const std::optional<MarkerSelection>& selected) {
  return selected && candidate.key == selected->candidate_key &&
         candidate.frontier_canonical_key &&
         *candidate.frontier_canonical_key == selected->frontier_canonical_key;
}

}  // namespace

visualization_msgs::msg::MarkerArray MarkerBuilder::Build(
    const std::span<const FrontierCluster> frontiers,
    const std::span<const CandidateView> candidates,
    const std::optional<MarkerSelection>& selected) {
  visualization_msgs::msg::MarkerArray result;
  result.markers.reserve(frontiers.size() + candidates.size() +
                         previous_frontier_count_ + previous_candidate_count_);
  for (std::size_t index = 0U; index < frontiers.size(); ++index) {
    const auto& frontier = frontiers[index];
    Marker marker;
    marker.ns = "frontiers";
    marker.id = MarkerId(index);
    marker.type = Marker::SPHERE;
    marker.action = Marker::ADD;
    marker.pose.position.x = frontier.centroid.x;
    marker.pose.position.y = frontier.centroid.y;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.20;
    marker.scale.y = 0.20;
    marker.scale.z = 0.20;
    marker.color.b = 1.0F;
    marker.color.a = 1.0F;
    result.markers.push_back(std::move(marker));
  }
  for (std::size_t index = 0U; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    Marker marker;
    marker.ns = "candidates";
    marker.id = MarkerId(index);
    marker.type = Marker::ARROW;
    marker.action = Marker::ADD;
    marker.pose.position.x = candidate.pose.x;
    marker.pose.position.y = candidate.pose.y;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.z = std::sin(candidate.pose.yaw / 2.0);
    marker.pose.orientation.w = std::cos(candidate.pose.yaw / 2.0);
    marker.scale.x = 0.35;
    marker.scale.y = 0.08;
    marker.scale.z = 0.08;
    marker.color.a = 1.0F;
    if (IsSelected(candidate, selected)) {
      marker.color.r = 1.0F;
    } else {
      marker.color.g = 1.0F;
    }
    result.markers.push_back(std::move(marker));
  }
  for (std::size_t index = frontiers.size(); index < previous_frontier_count_;
       ++index) {
    result.markers.push_back(Deletion("frontiers", index));
  }
  for (std::size_t index = candidates.size(); index < previous_candidate_count_;
       ++index) {
    result.markers.push_back(Deletion("candidates", index));
  }
  previous_frontier_count_ = frontiers.size();
  previous_candidate_count_ = candidates.size();
  return result;
}

}  // namespace lunar::pure_exploration_ros
