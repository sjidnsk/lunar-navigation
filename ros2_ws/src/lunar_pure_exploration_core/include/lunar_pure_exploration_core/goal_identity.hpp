#pragma once

#include <compare>
#include <cstdint>
#include <vector>

#include "lunar_pure_exploration_core/candidate_generator.hpp"

namespace lunar::pure_exploration {

enum class GoalKind : std::uint8_t { kBoundaryApproach, kTaskFrontier };

enum class ApproachCandidateKind : std::uint8_t { kTranslation, kRotation };

struct TaskFrontierGoalIdentity {
  std::vector<std::int64_t> frontier_canonical_key;
  CandidateKey candidate_key;
};

struct BoundaryApproachGoalIdentity {
  GridIndex intent_cell;
  CandidateKey candidate_key;
  ApproachCandidateKind candidate_kind;
  auto operator<=>(const BoundaryApproachGoalIdentity&) const = default;
};

}  // namespace lunar::pure_exploration
