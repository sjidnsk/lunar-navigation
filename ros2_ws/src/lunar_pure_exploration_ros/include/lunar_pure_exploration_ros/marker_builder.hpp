#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <lunar_pure_exploration_core/candidate_generator.hpp>
#include <lunar_pure_exploration_core/frontier_detector.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::pure_exploration_ros {

struct MarkerSelection final {
  lunar::pure_exploration::CandidateKey candidate_key;
  std::vector<std::int64_t> frontier_canonical_key;
  lunar::pure_exploration::Pose2 target;
};

// Marker IDs intentionally use the frozen vector index, not display hashes.
// The builder retains only the prior namespace sizes so a replacement batch
// explicitly deletes stale entries while all semantic identity stays frozen.
class MarkerBuilder final {
 public:
  visualization_msgs::msg::MarkerArray Build(
      std::span<const lunar::pure_exploration::FrontierCluster> frontiers,
      std::span<const lunar::pure_exploration::CandidateView> candidates,
      const std::optional<MarkerSelection>& selected);

 private:
  std::size_t previous_frontier_count_{0U};
  std::size_t previous_candidate_count_{0U};
};

}  // namespace lunar::pure_exploration_ros
