#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "lunar_pure_exploration_core/candidate_generator.hpp"

namespace lunar::pure_exploration {

struct CandidateGain {
  std::size_t candidate_index;
  double information_gain_m2;
};

struct PlannedCandidate {
  std::size_t candidate_index;
  double path_length_m;
};

struct RankedCandidate {
  std::size_t candidate_index;
  double information_gain_m2;
  double euclidean_distance_m;
  double path_length_m;
  double heading_change_rad;
  double revisit_penalty;
  double rank_value;
};

struct ScoreWeights {
  double information_gain{0.60};
  double path_cost{0.30};
  double heading_change{0.05};
  double revisit{0.05};
};

class CandidateRanker {
 public:
  CandidateRanker(ScoreWeights weights, double platform_length_m);

  std::vector<RankedCandidate> CoarseRank(
      std::span<const CandidateView> frozen_candidates,
      std::span<const FrontierCluster> frozen_frontiers,
      std::span<const CandidateGain> gains,
      Pose2 frozen_robot_pose) const;

  std::vector<RankedCandidate> FinalRank(
      std::span<const CandidateView> frozen_candidates,
      std::span<const FrontierCluster> frozen_frontiers,
      std::span<const CandidateGain> gains,
      std::span<const PlannedCandidate> planned,
      Pose2 frozen_robot_pose,
      std::span<const Vec2> completed_goal_positions,
      double resolution_m) const;

 private:
  ScoreWeights weights_;
  double platform_length_m_;
};

}  // namespace lunar::pure_exploration
