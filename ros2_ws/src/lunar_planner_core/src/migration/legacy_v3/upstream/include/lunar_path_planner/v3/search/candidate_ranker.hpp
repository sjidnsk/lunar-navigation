#pragma once

#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

struct CandidateScore final {
  std::string stable_candidate_id;
  DurationNanoseconds total_time;
  double energy{};
  double risk{};
  double smoothness{};
  bool fully_hard_validated{};
};

struct TimeEquivalentPool final {
  DurationNanoseconds minimum_validated_time;
  DurationNanoseconds tolerance;
  std::vector<CandidateScore> ordered_candidates;
};

class CandidateRanker final {
 public:
  [[nodiscard]] static Result<TimeEquivalentPool>
  BuildTimeEquivalentPool(
      std::span<const CandidateScore> candidates,
      DurationNanoseconds time_equivalence_tolerance);
};

}  // namespace lunar::planning::v3
