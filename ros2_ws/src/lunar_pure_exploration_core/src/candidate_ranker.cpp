#include "lunar_pure_exploration_core/candidate_ranker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lunar::pure_exploration {
namespace {

struct FullIdentity {
  const std::vector<std::int64_t>* frontier_key;
  CandidateKey candidate_key;
};

struct FullIdentityLess {
  bool operator()(const FullIdentity& left, const FullIdentity& right) const {
    if (*left.frontier_key != *right.frontier_key) {
      return *left.frontier_key < *right.frontier_key;
    }
    return left.candidate_key < right.candidate_key;
  }
};

struct WorkingRow {
  RankedCandidate ranked;
  double normalized_yaw;
};

bool Finite(Pose2 pose) {
  return std::isfinite(pose.x) && std::isfinite(pose.y) &&
         std::isfinite(pose.yaw);
}

bool Finite(Vec2 point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

double NormalizeYaw(double yaw) {
  if (!std::isfinite(yaw)) {
    throw std::invalid_argument("candidate yaw must be finite");
  }
  if (yaw >= -std::numbers::pi && yaw < std::numbers::pi) {
    return yaw == 0.0 ? 0.0 : yaw;
  }
  if (yaw == std::numbers::pi) {
    return -std::numbers::pi;
  }
  const double full_turn = 2.0 * std::numbers::pi;
  double normalized = std::fmod(yaw + std::numbers::pi, full_turn);
  if (normalized < 0.0) {
    normalized += full_turn;
  }
  normalized -= std::numbers::pi;
  return normalized == 0.0 ? 0.0 : normalized;
}

double HeadingChange(double normalized_target_yaw, double robot_yaw) {
  const double full_turn = 2.0 * std::numbers::pi;
  double heading =
      std::abs(std::remainder(normalized_target_yaw - robot_yaw, full_turn));
  if (!std::isfinite(heading) || heading > std::numbers::pi) {
    throw std::overflow_error("candidate heading change is not finite");
  }
  if (heading == 0.0) {
    heading = 0.0;
  }
  return heading;
}

double EuclideanDistance(Pose2 candidate, Pose2 robot) {
  const double distance =
      std::hypot(candidate.x - robot.x, candidate.y - robot.y);
  if (!std::isfinite(distance)) {
    throw std::overflow_error("candidate Euclidean distance is not finite");
  }
  return distance;
}

double CheckedDouble(long double value, const char* description) {
  if (!std::isfinite(value) ||
      value > static_cast<long double>(std::numeric_limits<double>::max()) ||
      value < -static_cast<long double>(std::numeric_limits<double>::max())) {
    throw std::overflow_error(description);
  }
  const double converted = static_cast<double>(value);
  if (!std::isfinite(converted) || (value != 0.0L && converted == 0.0)) {
    throw std::overflow_error(description);
  }
  return converted == 0.0 ? 0.0 : converted;
}

std::vector<double> ValidateAuthority(
    std::span<const CandidateView> candidates,
    std::span<const FrontierCluster> frontiers,
    std::span<const CandidateGain> gains,
    Pose2 robot_pose,
    std::vector<double>* normalized_yaws,
    std::vector<double>* distances,
    std::vector<double>* headings) {
  if (!Finite(robot_pose)) {
    throw std::invalid_argument("frozen robot pose must be finite");
  }
  if (gains.size() != candidates.size()) {
    throw std::invalid_argument("gain rows must completely cover candidates");
  }

  std::set<FullIdentity, FullIdentityLess> identities;
  normalized_yaws->assign(candidates.size(), 0.0);
  distances->assign(candidates.size(), 0.0);
  headings->assign(candidates.size(), 0.0);
  for (std::size_t index = 0U; index < candidates.size(); ++index) {
    const CandidateView& candidate = candidates[index];
    if (candidate.frontier_index >= frontiers.size()) {
      throw std::invalid_argument("candidate frontier index is out of bounds");
    }
    const FrontierCluster& frontier = frontiers[candidate.frontier_index];
    if (!Finite(candidate.pose) ||
        !std::isfinite(candidate.frontier_distance_m) ||
        candidate.frontier_distance_m < 0.0) {
      throw std::invalid_argument("candidate structure must be finite and nonnegative");
    }
    if (!identities.insert(FullIdentity{&frontier.canonical_key, candidate.key})
             .second) {
      throw std::invalid_argument("candidate full identity must be unique");
    }
    (*normalized_yaws)[index] = NormalizeYaw(candidate.pose.yaw);
    (*distances)[index] = EuclideanDistance(candidate.pose, robot_pose);
    (*headings)[index] =
        HeadingChange((*normalized_yaws)[index], robot_pose.yaw);
  }

  std::vector<double> gain_by_index(candidates.size(), 0.0);
  std::vector<bool> seen(candidates.size(), false);
  for (const CandidateGain& gain : gains) {
    if (gain.candidate_index >= candidates.size()) {
      throw std::invalid_argument("gain candidate index is out of bounds");
    }
    if (seen[gain.candidate_index]) {
      throw std::invalid_argument("gain candidate index must be unique");
    }
    if (!std::isfinite(gain.information_gain_m2) ||
        gain.information_gain_m2 < 0.0) {
      throw std::invalid_argument("candidate gain must be finite and nonnegative");
    }
    seen[gain.candidate_index] = true;
    gain_by_index[gain.candidate_index] = gain.information_gain_m2;
  }
  if (std::ranges::find(seen, false) != seen.end()) {
    throw std::invalid_argument("gain rows must completely cover candidates");
  }
  return gain_by_index;
}

bool IdentityBefore(std::size_t left_index, std::size_t right_index,
                    std::span<const CandidateView> candidates,
                    std::span<const FrontierCluster> frontiers) {
  const CandidateView& left = candidates[left_index];
  const CandidateView& right = candidates[right_index];
  const auto& left_frontier = frontiers[left.frontier_index].canonical_key;
  const auto& right_frontier = frontiers[right.frontier_index].canonical_key;
  if (left_frontier != right_frontier) {
    return left_frontier < right_frontier;
  }
  return left.key < right.key;
}

std::vector<RankedCandidate> ExtractRanked(std::vector<WorkingRow> rows) {
  std::vector<RankedCandidate> result;
  result.reserve(rows.size());
  for (WorkingRow& row : rows) {
    result.push_back(std::move(row.ranked));
  }
  return result;
}

}  // namespace

CandidateRanker::CandidateRanker(ScoreWeights weights,
                                 double platform_length_m)
    : weights_(weights), platform_length_m_(platform_length_m) {
  const std::array<double, 4U> values{
      weights_.information_gain, weights_.path_cost,
      weights_.heading_change, weights_.revisit};
  long double sum = 0.0L;
  for (double weight : values) {
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("candidate score weights must be finite and nonnegative");
    }
    sum += static_cast<long double>(weight);
  }
  if (!(sum > 0.0L) || !std::isfinite(sum)) {
    throw std::invalid_argument("candidate score weight sum must be positive");
  }
  if (!std::isfinite(platform_length_m_) || platform_length_m_ <= 0.0) {
    throw std::invalid_argument("platform length must be finite and positive");
  }
}

std::vector<RankedCandidate> CandidateRanker::CoarseRank(
    std::span<const CandidateView> frozen_candidates,
    std::span<const FrontierCluster> frozen_frontiers,
    std::span<const CandidateGain> gains,
    Pose2 frozen_robot_pose) const {
  std::vector<double> normalized_yaws;
  std::vector<double> distances;
  std::vector<double> headings;
  const std::vector<double> gain_by_index = ValidateAuthority(
      frozen_candidates, frozen_frontiers, gains, frozen_robot_pose,
      &normalized_yaws, &distances, &headings);

  std::vector<WorkingRow> rows;
  rows.reserve(frozen_candidates.size());
  for (std::size_t index = 0U; index < frozen_candidates.size(); ++index) {
    const double gain = gain_by_index[index];
    if (gain == 0.0) {
      continue;
    }
    const double denominator = distances[index] + platform_length_m_;
    if (!std::isfinite(denominator) || denominator <= 0.0) {
      throw std::overflow_error("coarse rank denominator is not finite positive");
    }
    const double utility = gain / denominator;
    if (!std::isfinite(utility)) {
      throw std::overflow_error("coarse rank utility is not finite");
    }
    rows.push_back(WorkingRow{
        RankedCandidate{index, gain, distances[index], 0.0, headings[index],
                        0.0, utility},
        normalized_yaws[index]});
  }

  std::sort(rows.begin(), rows.end(), [&](const WorkingRow& left,
                                          const WorkingRow& right) {
    const RankedCandidate& a = left.ranked;
    const RankedCandidate& b = right.ranked;
    if (a.rank_value != b.rank_value) {
      return a.rank_value > b.rank_value;
    }
    if (a.information_gain_m2 != b.information_gain_m2) {
      return a.information_gain_m2 > b.information_gain_m2;
    }
    if (a.euclidean_distance_m != b.euclidean_distance_m) {
      return a.euclidean_distance_m < b.euclidean_distance_m;
    }
    if (a.heading_change_rad != b.heading_change_rad) {
      return a.heading_change_rad < b.heading_change_rad;
    }
    const Pose2& a_pose = frozen_candidates[a.candidate_index].pose;
    const Pose2& b_pose = frozen_candidates[b.candidate_index].pose;
    if (a_pose.x != b_pose.x) {
      return a_pose.x < b_pose.x;
    }
    if (a_pose.y != b_pose.y) {
      return a_pose.y < b_pose.y;
    }
    if (left.normalized_yaw != right.normalized_yaw) {
      return left.normalized_yaw < right.normalized_yaw;
    }
    return IdentityBefore(a.candidate_index, b.candidate_index,
                          frozen_candidates, frozen_frontiers);
  });
  return ExtractRanked(std::move(rows));
}

std::vector<RankedCandidate> CandidateRanker::FinalRank(
    std::span<const CandidateView> frozen_candidates,
    std::span<const FrontierCluster> frozen_frontiers,
    std::span<const CandidateGain> gains,
    std::span<const PlannedCandidate> planned,
    Pose2 frozen_robot_pose,
    std::span<const Vec2> completed_goal_positions,
    double resolution_m) const {
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0) {
    throw std::invalid_argument("ranking resolution must be finite and positive");
  }
  for (const Vec2 goal : completed_goal_positions) {
    if (!Finite(goal)) {
      throw std::invalid_argument("completed goal position must be finite");
    }
  }

  std::vector<double> normalized_yaws;
  std::vector<double> distances;
  std::vector<double> headings;
  const std::vector<double> gain_by_index = ValidateAuthority(
      frozen_candidates, frozen_frontiers, gains, frozen_robot_pose,
      &normalized_yaws, &distances, &headings);

  std::vector<double> path_by_index(frozen_candidates.size(), 0.0);
  std::vector<bool> reachable(frozen_candidates.size(), false);
  double maximum_gain = 0.0;
  double maximum_path = 0.0;
  for (const PlannedCandidate& row : planned) {
    if (row.candidate_index >= frozen_candidates.size()) {
      throw std::invalid_argument("planned candidate index is out of bounds");
    }
    if (reachable[row.candidate_index]) {
      throw std::invalid_argument("planned candidate index must be unique");
    }
    if (!std::isfinite(row.path_length_m) || row.path_length_m < 0.0) {
      throw std::invalid_argument("planned path length must be finite and nonnegative");
    }
    reachable[row.candidate_index] = true;
    path_by_index[row.candidate_index] = row.path_length_m;
    maximum_gain = std::max(maximum_gain, gain_by_index[row.candidate_index]);
    maximum_path = std::max(maximum_path, row.path_length_m);
  }
  if (planned.empty()) {
    return {};
  }

  const double twice_platform = 2.0 * platform_length_m_;
  const double twice_resolution = 2.0 * resolution_m;
  const double revisit_radius = std::max(twice_platform, twice_resolution);
  if (!std::isfinite(twice_platform) || !std::isfinite(twice_resolution) ||
      !std::isfinite(revisit_radius) || revisit_radius <= 0.0) {
    throw std::overflow_error("revisit radius is not finite positive");
  }

  std::vector<WorkingRow> rows;
  rows.reserve(planned.size());
  for (std::size_t index = 0U; index < frozen_candidates.size(); ++index) {
    if (!reachable[index]) {
      continue;
    }
    const double normalized_gain =
        maximum_gain == 0.0 ? 0.0 : gain_by_index[index] / maximum_gain;
    const double normalized_path =
        maximum_path == 0.0 ? 0.0 : path_by_index[index] / maximum_path;
    const double normalized_heading = headings[index] / std::numbers::pi;
    if (!std::isfinite(normalized_gain) || !std::isfinite(normalized_path) ||
        !std::isfinite(normalized_heading)) {
      throw std::overflow_error("normalized candidate score component is not finite");
    }

    double revisit = 0.0;
    for (const Vec2 goal : completed_goal_positions) {
      const long double dx = static_cast<long double>(frozen_candidates[index].pose.x) -
                             static_cast<long double>(goal.x);
      const long double dy = static_cast<long double>(frozen_candidates[index].pose.y) -
                             static_cast<long double>(goal.y);
      const long double distance = std::hypot(dx, dy);
      if (!std::isfinite(distance)) {
        throw std::overflow_error("completed-goal distance is not finite");
      }
      revisit = revisit == 1.0 ||
                        distance <= static_cast<long double>(revisit_radius)
                    ? 1.0
                    : 0.0;
    }

    const long double gain_component =
        static_cast<long double>(weights_.information_gain) * normalized_gain;
    const long double path_component =
        -static_cast<long double>(weights_.path_cost) * normalized_path;
    const long double heading_component =
        -static_cast<long double>(weights_.heading_change) * normalized_heading;
    const long double revisit_component =
        -static_cast<long double>(weights_.revisit) * revisit;
    (void)CheckedDouble(gain_component, "gain score component is not representable");
    (void)CheckedDouble(path_component, "path score component is not representable");
    (void)CheckedDouble(heading_component,
                        "heading score component is not representable");
    (void)CheckedDouble(revisit_component,
                        "revisit score component is not representable");
    const double score = CheckedDouble(
        gain_component + path_component + heading_component + revisit_component,
        "final candidate score is not representable");
    rows.push_back(WorkingRow{
        RankedCandidate{index, gain_by_index[index], distances[index],
                        path_by_index[index], headings[index], revisit, score},
        normalized_yaws[index]});
  }

  std::sort(rows.begin(), rows.end(), [&](const WorkingRow& left,
                                          const WorkingRow& right) {
    const RankedCandidate& a = left.ranked;
    const RankedCandidate& b = right.ranked;
    if (a.rank_value != b.rank_value) {
      return a.rank_value > b.rank_value;
    }
    if (a.information_gain_m2 != b.information_gain_m2) {
      return a.information_gain_m2 > b.information_gain_m2;
    }
    if (a.path_length_m != b.path_length_m) {
      return a.path_length_m < b.path_length_m;
    }
    if (a.heading_change_rad != b.heading_change_rad) {
      return a.heading_change_rad < b.heading_change_rad;
    }
    const Pose2& a_pose = frozen_candidates[a.candidate_index].pose;
    const Pose2& b_pose = frozen_candidates[b.candidate_index].pose;
    if (a_pose.x != b_pose.x) {
      return a_pose.x < b_pose.x;
    }
    if (a_pose.y != b_pose.y) {
      return a_pose.y < b_pose.y;
    }
    if (left.normalized_yaw != right.normalized_yaw) {
      return left.normalized_yaw < right.normalized_yaw;
    }
    return IdentityBefore(a.candidate_index, b.candidate_index,
                          frozen_candidates, frozen_frontiers);
  });
  return ExtractRanked(std::move(rows));
}

}  // namespace lunar::pure_exploration
