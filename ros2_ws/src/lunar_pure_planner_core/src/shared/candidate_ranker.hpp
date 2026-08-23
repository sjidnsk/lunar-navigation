#pragma once

#include <compare>
#include <cstddef>
#include <span>
#include <vector>

namespace lunar::pure_planning::shared {

struct CandidateScore final {
  double cost{};
  std::size_t stable_index{};
  bool fully_hard_validated{};
  double energy{};
  double risk{};
  double smoothness{};

  bool operator==(const CandidateScore&) const = default;
};

[[nodiscard]] std::vector<CandidateScore> RankCandidates(
    std::span<const CandidateScore> candidates);

}  // namespace lunar::pure_planning::shared
