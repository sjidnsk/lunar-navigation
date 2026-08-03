#include "shared/candidate_ranker.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace lunar::planning::shared {

std::vector<CandidateScore> RankCandidates(
    const std::span<const CandidateScore> candidates) {
  std::vector<CandidateScore> ranked;
  ranked.reserve(candidates.size());
  for (const CandidateScore& candidate : candidates) {
    if (!candidate.fully_hard_validated || !std::isfinite(candidate.cost) ||
        candidate.cost < 0.0 || !std::isfinite(candidate.energy) ||
        !std::isfinite(candidate.risk) ||
        !std::isfinite(candidate.smoothness) || candidate.energy < 0.0 ||
        candidate.risk < 0.0 || candidate.smoothness < 0.0) {
      continue;
    }
    ranked.push_back(candidate);
  }
  std::stable_sort(
      ranked.begin(), ranked.end(),
      [](const CandidateScore& lhs, const CandidateScore& rhs) {
        return std::tie(lhs.cost, lhs.stable_index) <
               std::tie(rhs.cost, rhs.stable_index);
      });
  return ranked;
}

}  // namespace lunar::planning::shared
