#include "lunar_path_planner/v3/search/candidate_ranker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsAsciiAlphanumeric(char value) noexcept {
  const auto byte = static_cast<unsigned char>(value);
  return (byte >= static_cast<unsigned char>('A') &&
          byte <= static_cast<unsigned char>('Z')) ||
         (byte >= static_cast<unsigned char>('a') &&
          byte <= static_cast<unsigned char>('z')) ||
         (byte >= static_cast<unsigned char>('0') &&
          byte <= static_cast<unsigned char>('9'));
}

[[nodiscard]] bool IsIdentifierCharacter(char value) noexcept {
  return IsAsciiAlphanumeric(value) || value == '.' || value == '_' ||
         value == ':' || value == '/' || value == '-';
}

[[nodiscard]] bool IsStableCandidateId(
    std::string_view value) noexcept {
  return !value.empty() && value.size() <= 128U &&
         IsAsciiAlphanumeric(value.front()) &&
         std::ranges::all_of(value.substr(1U),
                             IsIdentifierCharacter);
}

[[nodiscard]] bool IsFiniteNonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

}  // namespace

Result<TimeEquivalentPool> CandidateRanker::BuildTimeEquivalentPool(
    std::span<const CandidateScore> candidates,
    DurationNanoseconds time_equivalence_tolerance) {
  using NanosecondsRep = std::chrono::nanoseconds::rep;

  const NanosecondsRep tolerance_count =
      time_equivalence_tolerance.value.count();
  if (tolerance_count < 0) {
    return Invalid("time_equivalence_tolerance",
                   "time equivalence tolerance must be nonnegative");
  }

  std::vector<CandidateScore> validated_candidates;
  validated_candidates.reserve(candidates.size());
  for (const CandidateScore& candidate : candidates) {
    if (candidate.fully_hard_validated) {
      validated_candidates.push_back(candidate);
    }
  }
  if (validated_candidates.empty()) {
    return Invalid("candidates",
                   "at least one fully hard validated candidate is required");
  }

  std::unordered_set<std::string> stable_ids;
  stable_ids.reserve(validated_candidates.size());
  for (const CandidateScore& candidate : validated_candidates) {
    if (!IsStableCandidateId(candidate.stable_candidate_id)) {
      return Invalid("candidates.stable_candidate_id",
                     "stable candidate ID must be a valid identifier");
    }
    if (!stable_ids.insert(candidate.stable_candidate_id).second) {
      return Invalid("candidates.stable_candidate_id",
                     "fully validated candidate IDs must be unique");
    }
    if (candidate.total_time.value.count() < 0) {
      return Invalid("candidates.total_time",
                     "validated candidate time must be nonnegative");
    }
    if (!IsFiniteNonnegative(candidate.energy)) {
      return Invalid("candidates.energy",
                     "validated candidate energy must be finite and "
                     "nonnegative");
    }
    if (!IsFiniteNonnegative(candidate.risk)) {
      return Invalid("candidates.risk",
                     "validated candidate risk must be finite and "
                     "nonnegative");
    }
    if (!IsFiniteNonnegative(candidate.smoothness)) {
      return Invalid("candidates.smoothness",
                     "validated candidate smoothness must be finite and "
                     "nonnegative");
    }
  }

  const auto minimum_iterator = std::ranges::min_element(
      validated_candidates, {}, [](const CandidateScore& candidate) {
        return candidate.total_time.value.count();
      });
  const NanosecondsRep minimum_count =
      minimum_iterator->total_time.value.count();
  constexpr NanosecondsRep kMaximumNanoseconds =
      std::numeric_limits<NanosecondsRep>::max();
  if (minimum_count > kMaximumNanoseconds - tolerance_count) {
    return Invalid("time_equivalent_band",
                   "minimum validated time plus tolerance overflows "
                   "nanoseconds");
  }
  const NanosecondsRep maximum_equivalent_count =
      minimum_count + tolerance_count;

  std::erase_if(validated_candidates,
                [maximum_equivalent_count](
                    const CandidateScore& candidate) {
                  return candidate.total_time.value.count() >
                         maximum_equivalent_count;
                });

  std::ranges::sort(
      validated_candidates,
      [](const CandidateScore& lhs, const CandidateScore& rhs) {
        return std::tuple{lhs.energy, lhs.risk, lhs.smoothness,
                          lhs.total_time.value.count(),
                          std::string_view{lhs.stable_candidate_id}} <
               std::tuple{rhs.energy, rhs.risk, rhs.smoothness,
                          rhs.total_time.value.count(),
                          std::string_view{rhs.stable_candidate_id}};
      });

  return TimeEquivalentPool{
      .minimum_validated_time =
          DurationNanoseconds{std::chrono::nanoseconds{minimum_count}},
      .tolerance = time_equivalence_tolerance,
      .ordered_candidates = std::move(validated_candidates),
  };
}

}  // namespace lunar::planning::v3
