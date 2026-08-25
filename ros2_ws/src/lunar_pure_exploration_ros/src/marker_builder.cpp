#include "lunar_pure_exploration_ros/marker_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using lunar::pure_exploration::CandidateView;
using lunar::pure_exploration::FrontierCluster;
using lunar::pure_exploration::Vec2;
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

geometry_msgs::msg::Point MarkerPoint(const Vec2& point) {
  geometry_msgs::msg::Point result;
  result.x = point.x;
  result.y = point.y;
  return result;
}

}  // namespace

visualization_msgs::msg::MarkerArray MarkerBuilder::Build(
    const std::span<const FrontierCluster> frontiers,
    const std::span<const CandidateView> candidates,
    const std::optional<MarkerSelection>& selected,
    const ApproachMarkers* const approach) {
  const std::size_t approach_intent_count =
      approach ? approach->intent_points.size() : 0U;
  const std::size_t approach_guidance_count =
      approach && !approach->selected_guidance.empty() ? 1U : 0U;
  const std::size_t approach_candidate_count =
      approach ? approach->candidates.size() : 0U;
  visualization_msgs::msg::MarkerArray result;
  result.markers.reserve(frontiers.size() + candidates.size() +
                         approach_intent_count + approach_guidance_count +
                         approach_candidate_count + previous_frontier_count_ +
                         previous_candidate_count_ +
                         previous_approach_intent_count_ +
                         previous_approach_guidance_count_ +
                         previous_approach_candidate_count_);
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
  if (approach) {
    for (std::size_t index = 0U; index < approach->intent_points.size();
         ++index) {
      const auto& intent = approach->intent_points[index];
      Marker marker;
      marker.ns = "approach_intents";
      marker.id = MarkerId(index);
      marker.type = Marker::SPHERE;
      marker.action = Marker::ADD;
      marker.pose.position.x = intent.x;
      marker.pose.position.y = intent.y;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.14;
      marker.scale.y = 0.14;
      marker.scale.z = 0.14;
      marker.color.b = 1.0F;
      marker.color.a = 1.0F;
      result.markers.push_back(std::move(marker));
    }
    if (!approach->selected_guidance.empty()) {
      Marker marker;
      marker.ns = "approach_guidance";
      marker.id = 0;
      marker.type = Marker::LINE_STRIP;
      marker.action = Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.06;
      marker.color.r = 1.0F;
      marker.color.g = 0.65F;
      marker.color.a = 1.0F;
      marker.points.reserve(approach->selected_guidance.size());
      std::ranges::transform(approach->selected_guidance,
                             std::back_inserter(marker.points), MarkerPoint);
      result.markers.push_back(std::move(marker));
    }
    for (std::size_t index = 0U; index < approach->candidates.size(); ++index) {
      const auto& candidate = approach->candidates[index];
      Marker marker;
      marker.ns = "approach_candidates";
      marker.id = MarkerId(index);
      marker.type = Marker::ARROW;
      marker.action = Marker::ADD;
      marker.pose.position.x = candidate.pose.x;
      marker.pose.position.y = candidate.pose.y;
      marker.pose.orientation.z = std::sin(candidate.pose.yaw / 2.0);
      marker.pose.orientation.w = std::cos(candidate.pose.yaw / 2.0);
      marker.scale.x = 0.35;
      marker.scale.y = 0.08;
      marker.scale.z = 0.08;
      marker.color.a = 1.0F;
      if (approach->selected_identity &&
          candidate.identity == *approach->selected_identity) {
        marker.color.r = 1.0F;
      } else {
        marker.color.g = 1.0F;
      }
      result.markers.push_back(std::move(marker));
    }
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
  for (std::size_t index = approach_intent_count;
       index < previous_approach_intent_count_; ++index) {
    result.markers.push_back(Deletion("approach_intents", index));
  }
  for (std::size_t index = approach_guidance_count;
       index < previous_approach_guidance_count_; ++index) {
    result.markers.push_back(Deletion("approach_guidance", index));
  }
  for (std::size_t index = approach_candidate_count;
       index < previous_approach_candidate_count_; ++index) {
    result.markers.push_back(Deletion("approach_candidates", index));
  }
  previous_frontier_count_ = frontiers.size();
  previous_candidate_count_ = candidates.size();
  previous_approach_intent_count_ = approach_intent_count;
  previous_approach_guidance_count_ = approach_guidance_count;
  previous_approach_candidate_count_ = approach_candidate_count;
  return result;
}

}  // namespace lunar::pure_exploration_ros
