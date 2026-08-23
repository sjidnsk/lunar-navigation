#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_set>

#include "lunar_pure_exploration_core/types.hpp"

namespace lunar::pure_exploration::detail {

enum class TraceControl { kContinue, kStop };

struct VisibilityWorkBudget {
  std::size_t limit;
  std::size_t used;
};

inline void ConsumeVisibilityWork(VisibilityWorkBudget& budget) {
  if (budget.used == budget.limit) {
    throw std::length_error("visibility work budget exhausted");
  }
  if (budget.used > budget.limit ||
      budget.used == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("visibility work counter overflow");
  }
  ++budget.used;
}

struct TraceCell {
  std::int64_t x;
  std::int64_t y;
  auto operator<=>(const TraceCell&) const = default;
};

struct TraceSummary {
  std::size_t visited_cell_count;
};

struct TraceInstrumentation {
  std::size_t full_product_multiplications{0U};
  std::size_t incremental_product_additions{0U};
  std::size_t visited_insertions{0U};
};

using TraceGroupVisitor = std::function<TraceControl(
    long double first_contact_t, std::span<const TraceCell> canonical_group)>;

namespace traversal_internal {

inline std::int64_t CheckedFloorToInt64(long double value) {
  const long double floored = std::floor(value);
  if (!std::isfinite(floored) ||
      floored < static_cast<long double>(
                    std::numeric_limits<std::int64_t>::min()) ||
      floored > static_cast<long double>(
                    std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("visibility grid coordinate exceeds int64");
  }
  return static_cast<std::int64_t>(floored);
}

inline std::int64_t CheckedAdd(std::int64_t value, std::int64_t delta) {
  if ((delta > 0 && value > std::numeric_limits<std::int64_t>::max() - delta) ||
      (delta < 0 && value < std::numeric_limits<std::int64_t>::min() - delta)) {
    throw std::overflow_error("visibility grid step overflows int64");
  }
  return value + delta;
}

inline bool IsInteger(long double value) {
  return value == std::floor(value);
}

struct TraceCellHash {
  std::size_t operator()(const TraceCell& cell) const noexcept {
    const auto x = static_cast<std::uint64_t>(cell.x);
    const auto y = static_cast<std::uint64_t>(cell.y);
    const std::uint64_t mixed =
        x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6U) + (x >> 2U));
    return static_cast<std::size_t>(mixed);
  }
};

// Every accepted coordinate has an int64 floor.  Scaling an IEEE-754 double
// by 2^1074 therefore needs at most 1138 magnitude bits: 1074 fractional
// bits plus the 64-bit signed coordinate domain.  Products used to compare
// two event ratios need at most twice that width.
constexpr std::size_t kExactCoordinateWords = 36U;
constexpr std::size_t kExactProductWords = 72U;
constexpr std::size_t kDoubleFractionScale = 1074U;
constexpr std::size_t kMaximumTieGroupSize = 4U;
constexpr std::size_t kWordBits = std::numeric_limits<std::uint32_t>::digits;
constexpr std::size_t kCoordinateDifferenceBits =
    kDoubleFractionScale + std::numeric_limits<std::uint64_t>::digits;
constexpr std::size_t kAdvancedDistanceBits =
    kCoordinateDifferenceBits + 1U;
constexpr std::size_t kMaximumEventProductBits =
    kAdvancedDistanceBits + kCoordinateDifferenceBits;

static_assert(sizeof(double) == sizeof(std::uint64_t));
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::radix == 2);
static_assert(std::numeric_limits<double>::digits == 53);
static_assert(std::numeric_limits<double>::max_exponent == 1024);
static_assert(std::numeric_limits<double>::min_exponent == -1021);
static_assert(kExactCoordinateWords * kWordBits >= kAdvancedDistanceBits);
static_assert(kExactProductWords * kWordBits >= kMaximumEventProductBits);

template <std::size_t WordCount>
struct ExactUnsigned {
  std::array<std::uint32_t, WordCount> words{};
  std::size_t used_words{0U};
  std::size_t first_word{WordCount};
};

template <std::size_t WordCount>
inline int Compare(const ExactUnsigned<WordCount>& left,
                   const ExactUnsigned<WordCount>& right) {
  if (left.used_words < right.used_words) {
    return -1;
  }
  if (left.used_words > right.used_words) {
    return 1;
  }
  for (std::size_t offset = 0U; offset < left.used_words; ++offset) {
    const std::size_t index = left.used_words - 1U - offset;
    if (left.words[index] < right.words[index]) {
      return -1;
    }
    if (left.words[index] > right.words[index]) {
      return 1;
    }
  }
  return 0;
}

template <std::size_t WordCount>
inline bool IsZero(const ExactUnsigned<WordCount>& value) {
  return value.used_words == 0U;
}

template <std::size_t WordCount>
inline void AddShifted32(ExactUnsigned<WordCount>& output,
                         std::uint32_t value, std::size_t shift) {
  if (value == 0U) {
    return;
  }
  const std::size_t word = shift / 32U;
  const std::size_t bit = shift % 32U;
  if (word >= WordCount) {
    throw std::overflow_error("exact dyadic shift exceeds fixed capacity");
  }
  const std::uint64_t shifted = static_cast<std::uint64_t>(value) << bit;
  std::uint64_t carry = static_cast<std::uint32_t>(shifted);
  std::size_t index = word;
  while (carry != 0U) {
    if (index >= WordCount) {
      throw std::overflow_error("exact dyadic addition exceeds fixed capacity");
    }
    const std::uint64_t sum = output.words[index] + carry;
    output.words[index] = static_cast<std::uint32_t>(sum);
    carry = sum >> 32U;
    output.used_words = std::max(output.used_words, index + 1U);
    output.first_word = std::min(output.first_word, index);
    ++index;
  }
  const std::uint32_t upper = static_cast<std::uint32_t>(shifted >> 32U);
  if (upper != 0U) {
    index = word + 1U;
    carry = upper;
    while (carry != 0U) {
      if (index >= WordCount) {
        throw std::overflow_error(
            "exact dyadic addition exceeds fixed capacity");
      }
      const std::uint64_t sum = output.words[index] + carry;
      output.words[index] = static_cast<std::uint32_t>(sum);
      carry = sum >> 32U;
      output.used_words = std::max(output.used_words, index + 1U);
      output.first_word = std::min(output.first_word, index);
      ++index;
    }
  }
}

template <std::size_t WordCount>
inline void AddShifted64(ExactUnsigned<WordCount>& output,
                         std::uint64_t value, std::size_t shift) {
  AddShifted32(output, static_cast<std::uint32_t>(value), shift);
  AddShifted32(output, static_cast<std::uint32_t>(value >> 32U),
               shift + 32U);
}

template <std::size_t WordCount>
inline ExactUnsigned<WordCount> CheckedAdd(
    const ExactUnsigned<WordCount>& left,
    const ExactUnsigned<WordCount>& right) {
  ExactUnsigned<WordCount> result;
  std::uint32_t carry = 0U;
  const std::size_t input_words =
      std::max(left.used_words, right.used_words);
  const std::size_t first_word =
      std::min(left.first_word, right.first_word);
  for (std::size_t index = first_word; index < input_words; ++index) {
    const std::uint64_t sum = static_cast<std::uint64_t>(left.words[index]) +
                              right.words[index] + carry;
    result.words[index] = static_cast<std::uint32_t>(sum);
    carry = static_cast<std::uint32_t>(sum >> 32U);
  }
  if (carry != 0U) {
    if (input_words == WordCount) {
      throw std::overflow_error("exact dyadic addition exceeds fixed capacity");
    }
    result.words[input_words] = carry;
    result.used_words = input_words + 1U;
  } else {
    result.used_words = input_words;
  }
  result.first_word = first_word;
  while (result.first_word < result.used_words &&
         result.words[result.first_word] == 0U) {
    ++result.first_word;
  }
  if (result.first_word == result.used_words) {
    result.used_words = 0U;
    result.first_word = WordCount;
  }
  return result;
}

template <std::size_t WordCount>
inline ExactUnsigned<WordCount> CheckedSubtract(
    const ExactUnsigned<WordCount>& high,
    const ExactUnsigned<WordCount>& low) {
  if (Compare(high, low) < 0) {
    throw std::overflow_error("exact dyadic subtraction became negative");
  }
  ExactUnsigned<WordCount> result;
  std::uint32_t borrow = 0U;
  const std::size_t first_word =
      std::min(high.first_word, low.first_word);
  for (std::size_t index = first_word; index < high.used_words; ++index) {
    const std::uint64_t subtrahend =
        static_cast<std::uint64_t>(low.words[index]) + borrow;
    const std::uint64_t minuend = high.words[index];
    result.words[index] = static_cast<std::uint32_t>(minuend - subtrahend);
    borrow = minuend < subtrahend ? 1U : 0U;
  }
  if (borrow != 0U) {
    throw std::overflow_error("exact dyadic subtraction underflow");
  }
  result.used_words = high.used_words;
  while (result.used_words != 0U &&
         result.words[result.used_words - 1U] == 0U) {
    --result.used_words;
  }
  result.first_word = first_word;
  while (result.first_word < result.used_words &&
         result.words[result.first_word] == 0U) {
    ++result.first_word;
  }
  if (result.used_words == 0U) {
    result.first_word = WordCount;
  }
  return result;
}

struct ExactCoordinate {
  bool negative{false};
  ExactUnsigned<kExactCoordinateWords> magnitude;
};

inline ExactCoordinate ExactFromDouble(double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("exact dyadic input must be finite");
  }
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  const std::uint64_t raw_exponent = (bits >> 52U) & 0x7ffU;
  const std::uint64_t fraction = bits & ((std::uint64_t{1U} << 52U) - 1U);
  const std::uint64_t significand =
      raw_exponent == 0U ? fraction
                         : ((std::uint64_t{1U} << 52U) | fraction);
  const std::size_t shift =
      raw_exponent == 0U ? 0U : static_cast<std::size_t>(raw_exponent - 1U);
  ExactCoordinate result;
  AddShifted64(result.magnitude, significand, shift);
  result.negative = ((bits >> 63U) != 0U) && !IsZero(result.magnitude);
  return result;
}

inline ExactCoordinate ExactFromInt64(std::int64_t value) {
  const std::uint64_t magnitude =
      value < 0 ? static_cast<std::uint64_t>(-(value + 1)) + 1U
                : static_cast<std::uint64_t>(value);
  ExactCoordinate result;
  AddShifted64(result.magnitude, magnitude, kDoubleFractionScale);
  result.negative = value < 0;
  return result;
}

inline int Compare(const ExactCoordinate& left, const ExactCoordinate& right) {
  if (left.negative != right.negative) {
    return left.negative ? -1 : 1;
  }
  const int magnitude_order = Compare(left.magnitude, right.magnitude);
  return left.negative ? -magnitude_order : magnitude_order;
}

inline ExactUnsigned<kExactCoordinateWords> PositiveDifference(
    const ExactCoordinate& high, const ExactCoordinate& low) {
  if (Compare(high, low) < 0) {
    throw std::overflow_error("exact dyadic difference became negative");
  }
  if (high.negative == low.negative) {
    return high.negative ? CheckedSubtract(low.magnitude, high.magnitude)
                         : CheckedSubtract(high.magnitude, low.magnitude);
  }
  return CheckedAdd(high.magnitude, low.magnitude);
}

inline ExactUnsigned<kExactCoordinateWords> AbsoluteDifference(
    const ExactCoordinate& left, const ExactCoordinate& right) {
  return Compare(left, right) < 0 ? PositiveDifference(right, left)
                                  : PositiveDifference(left, right);
}

inline ExactUnsigned<kExactCoordinateWords> ExactGridUnit() {
  ExactUnsigned<kExactCoordinateWords> result;
  AddShifted64(result, 1U, kDoubleFractionScale);
  return result;
}

inline ExactUnsigned<kExactProductWords> CheckedMultiply(
    const ExactUnsigned<kExactCoordinateWords>& left,
    const ExactUnsigned<kExactCoordinateWords>& right) {
  ExactUnsigned<kExactProductWords> result;
  if (IsZero(left) || IsZero(right)) {
    return result;
  }
  for (std::size_t left_index = left.first_word;
       left_index < left.used_words;
       ++left_index) {
    if (left.words[left_index] == 0U) {
      continue;
    }
    std::uint64_t carry = 0U;
    for (std::size_t right_index = right.first_word;
         right_index < right.used_words; ++right_index) {
      const std::size_t output_index = left_index + right_index;
      const std::uint64_t product =
          static_cast<std::uint64_t>(left.words[left_index]) *
          right.words[right_index];
      const std::uint64_t sum =
          product + result.words[output_index] + carry;
      result.words[output_index] = static_cast<std::uint32_t>(sum);
      carry = sum >> 32U;
    }
    std::size_t carry_index = left_index + right.used_words;
    while (carry != 0U) {
      if (carry_index >= kExactProductWords) {
        throw std::overflow_error(
            "exact dyadic product exceeds fixed capacity");
      }
      const std::uint64_t sum = result.words[carry_index] + carry;
      result.words[carry_index] = static_cast<std::uint32_t>(sum);
      carry = sum >> 32U;
      ++carry_index;
    }
  }
  result.used_words = std::min(kExactProductWords,
                               left.used_words + right.used_words);
  while (result.used_words != 0U &&
         result.words[result.used_words - 1U] == 0U) {
    --result.used_words;
  }
  if (result.used_words != 0U) {
    result.first_word = left.first_word + right.first_word;
    while (result.first_word < result.used_words &&
           result.words[result.first_word] == 0U) {
      ++result.first_word;
    }
  }
  return result;
}

inline ExactUnsigned<kExactProductWords> GridUnitProductStride(
    const ExactUnsigned<kExactCoordinateWords>& value) {
  ExactUnsigned<kExactProductWords> result;
  for (std::size_t index = 0U; index < value.used_words; ++index) {
    AddShifted32(result, value.words[index],
                 kDoubleFractionScale + index * kWordBits);
  }
  return result;
}

inline int CompareEventRatios(
    const ExactUnsigned<kExactCoordinateWords>& x_distance,
    const ExactUnsigned<kExactCoordinateWords>& x_delta,
    const ExactUnsigned<kExactCoordinateWords>& y_distance,
    const ExactUnsigned<kExactCoordinateWords>& y_delta) {
  if (IsZero(x_delta) || IsZero(y_delta)) {
    throw std::overflow_error("exact event ratio has zero denominator");
  }
  return Compare(CheckedMultiply(x_distance, y_delta),
                 CheckedMultiply(y_distance, x_delta));
}

inline std::array<std::int64_t, 2> ClosedAxisCells(long double coordinate,
                                                   bool on_boundary,
                                                   std::size_t& count) {
  const std::int64_t lower = CheckedFloorToInt64(coordinate);
  if (!on_boundary) {
    count = 1U;
    return {lower, lower};
  }
  count = 2U;
  return {CheckedAdd(lower, -1), lower};
}

}  // namespace traversal_internal

inline TraceSummary TraceClosedSegment(
    Vec2 start_grid, Vec2 end_grid, VisibilityWorkBudget& budget,
    const TraceGroupVisitor& visitor,
    TraceInstrumentation* instrumentation = nullptr) {
  if (!std::isfinite(start_grid.x) || !std::isfinite(start_grid.y) ||
      !std::isfinite(end_grid.x) || !std::isfinite(end_grid.y)) {
    throw std::invalid_argument("visibility segment endpoints must be finite");
  }
  if (!visitor) {
    throw std::invalid_argument("visibility trace visitor is required");
  }

  using traversal_internal::CheckedAdd;
  using traversal_internal::CheckedFloorToInt64;
  using traversal_internal::ClosedAxisCells;
  using traversal_internal::Compare;
  using traversal_internal::ExactFromDouble;
  using traversal_internal::ExactFromInt64;
  using traversal_internal::ExactGridUnit;
  using traversal_internal::ExactUnsigned;
  using traversal_internal::IsInteger;
  using traversal_internal::PositiveDifference;
  using traversal_internal::TraceCellHash;
  constexpr std::size_t kMaximumTieGroupSize =
      traversal_internal::kMaximumTieGroupSize;

  const long double start_x = start_grid.x;
  const long double start_y = start_grid.y;
  const long double delta_x = static_cast<long double>(end_grid.x) - start_x;
  const long double delta_y = static_cast<long double>(end_grid.y) - start_y;
  if (!std::isfinite(delta_x) || !std::isfinite(delta_y)) {
    throw std::overflow_error("visibility segment delta overflow");
  }
  const std::int64_t start_floor_x = CheckedFloorToInt64(start_x);
  const std::int64_t start_floor_y = CheckedFloorToInt64(start_y);
  (void)CheckedFloorToInt64(end_grid.x);
  (void)CheckedFloorToInt64(end_grid.y);

  std::unordered_set<TraceCell, TraceCellHash> visited;
  std::size_t visited_count = 0U;
  bool stopped = false;

  const auto emit_group = [&](long double t,
                              const std::array<TraceCell,
                                               kMaximumTieGroupSize>&
                                  candidates,
                              std::size_t candidate_count) {
    if (candidate_count > kMaximumTieGroupSize) {
      throw std::overflow_error("visibility tie group exceeds fixed capacity");
    }
    std::array<TraceCell, kMaximumTieGroupSize> group{};
    std::size_t group_size = 0U;
    for (std::size_t index = 0U; index < kMaximumTieGroupSize; ++index) {
      if (index >= candidate_count) {
        break;
      }
      if (visited.contains(candidates[index])) {
        continue;
      }
      bool duplicate_in_group = false;
      for (std::size_t group_index = 0U;
           group_index < kMaximumTieGroupSize; ++group_index) {
        if (group_index >= group_size) {
          break;
        }
        duplicate_in_group = duplicate_in_group ||
                             group[group_index] == candidates[index];
      }
      if (!duplicate_in_group) {
        group[group_size] = candidates[index];
        ++group_size;
      }
    }
    std::sort(group.begin(), group.begin() + group_size,
              [](const TraceCell& left, const TraceCell& right) {
                return left.x < right.x ||
                       (left.x == right.x && left.y < right.y);
              });
    if (group_size == 0U) {
      return;
    }
    if (group_size >
        std::numeric_limits<std::size_t>::max() - visited_count) {
      throw std::overflow_error("visibility visited count overflow");
    }
    if (budget.used > budget.limit) {
      throw std::overflow_error("visibility work budget is inconsistent");
    }
    if (group_size > budget.limit - budget.used) {
      throw std::length_error("visibility work budget exhausted");
    }
    for (std::size_t index = 0U; index < kMaximumTieGroupSize; ++index) {
      if (index >= group_size) {
        break;
      }
      const bool inserted = visited.insert(group[index]).second;
      if (!inserted) {
        throw std::logic_error("visibility duplicate escaped group preflight");
      }
      if (instrumentation != nullptr) {
        ++instrumentation->visited_insertions;
      }
      ConsumeVisibilityWork(budget);
    }
    visited_count += group_size;
    if (visitor(t, std::span<const TraceCell>(group.data(), group_size)) ==
        TraceControl::kStop) {
      stopped = true;
    }
  };

  std::size_t start_x_count = 0U;
  std::size_t start_y_count = 0U;
  const auto start_x_cells =
      ClosedAxisCells(start_x, IsInteger(start_x), start_x_count);
  const auto start_y_cells =
      ClosedAxisCells(start_y, IsInteger(start_y), start_y_count);
  std::array<TraceCell, kMaximumTieGroupSize> start_candidates{};
  const std::size_t start_candidate_count = start_x_count * start_y_count;
  for (std::size_t index = 0U; index < kMaximumTieGroupSize; ++index) {
    if (index >= start_candidate_count) {
      break;
    }
    const std::size_t x_index = index / start_y_count;
    const std::size_t y_index = index % start_y_count;
    start_candidates[index] =
        TraceCell{start_x_cells[x_index], start_y_cells[y_index]};
  }
  emit_group(0.0L, start_candidates, start_candidate_count);
  if (stopped || (delta_x == 0.0L && delta_y == 0.0L)) {
    return {visited_count};
  }

  const int step_x = (delta_x > 0.0L) - (delta_x < 0.0L);
  const int step_y = (delta_y > 0.0L) - (delta_y < 0.0L);
  std::int64_t current_x = start_floor_x;
  std::int64_t current_y = start_floor_y;
  if (step_x < 0 && IsInteger(start_x)) {
    current_x = CheckedAdd(current_x, -1);
  }
  if (step_y < 0 && IsInteger(start_y)) {
    current_y = CheckedAdd(current_y, -1);
  }

  std::int64_t next_x = start_floor_x;
  std::int64_t next_y = start_floor_y;
  if (step_x > 0) {
    next_x = CheckedAdd(next_x, 1);
  } else if (step_x < 0) {
    next_x = IsInteger(start_x) ? CheckedAdd(next_x, -1) : next_x;
  }
  if (step_y > 0) {
    next_y = CheckedAdd(next_y, 1);
  } else if (step_y < 0) {
    next_y = IsInteger(start_y) ? CheckedAdd(next_y, -1) : next_y;
  }

  const auto exact_start_x = ExactFromDouble(start_grid.x);
  const auto exact_start_y = ExactFromDouble(start_grid.y);
  const auto exact_end_x = ExactFromDouble(end_grid.x);
  const auto exact_end_y = ExactFromDouble(end_grid.y);
  const auto exact_delta_x =
      traversal_internal::AbsoluteDifference(exact_end_x, exact_start_x);
  const auto exact_delta_y =
      traversal_internal::AbsoluteDifference(exact_end_y, exact_start_y);
  const auto grid_unit = ExactGridUnit();

  ExactUnsigned<traversal_internal::kExactCoordinateWords> next_distance_x{};
  ExactUnsigned<traversal_internal::kExactCoordinateWords> next_distance_y{};
  if (step_x > 0) {
    next_distance_x =
        PositiveDifference(ExactFromInt64(next_x), exact_start_x);
  } else if (step_x < 0) {
    next_distance_x =
        PositiveDifference(exact_start_x, ExactFromInt64(next_x));
  }
  if (step_y > 0) {
    next_distance_y =
        PositiveDifference(ExactFromInt64(next_y), exact_start_y);
  } else if (step_y < 0) {
    next_distance_y =
        PositiveDifference(exact_start_y, ExactFromInt64(next_y));
  }
  bool x_active = step_x != 0 && Compare(next_distance_x, exact_delta_x) <= 0;
  bool y_active = step_y != 0 && Compare(next_distance_y, exact_delta_y) <= 0;
  ExactUnsigned<traversal_internal::kExactProductWords> x_event_product{};
  ExactUnsigned<traversal_internal::kExactProductWords> y_event_product{};
  ExactUnsigned<traversal_internal::kExactProductWords> x_product_stride{};
  ExactUnsigned<traversal_internal::kExactProductWords> y_product_stride{};
  if (x_active && y_active) {
    x_event_product =
        traversal_internal::CheckedMultiply(next_distance_x, exact_delta_y);
    y_event_product =
        traversal_internal::CheckedMultiply(next_distance_y, exact_delta_x);
    x_product_stride =
        traversal_internal::GridUnitProductStride(exact_delta_y);
    y_product_stride =
        traversal_internal::GridUnitProductStride(exact_delta_x);
    if (instrumentation != nullptr) {
      instrumentation->full_product_multiplications += 2U;
    }
  }

  while (!stopped && (x_active || y_active)) {
    bool crosses_x = x_active;
    bool crosses_y = y_active;
    if (x_active && y_active) {
      const int event_order = Compare(x_event_product, y_event_product);
      crosses_x = event_order <= 0;
      crosses_y = event_order >= 0;
    }

    const long double t =
        crosses_x
            ? (static_cast<long double>(next_x) - start_x) / delta_x
            : (static_cast<long double>(next_y) - start_y) / delta_y;
    std::array<TraceCell, kMaximumTieGroupSize> candidates{};
    std::size_t candidate_count = 0U;
    if (crosses_x && crosses_y) {
      const std::int64_t new_x = CheckedAdd(current_x, step_x);
      const std::int64_t new_y = CheckedAdd(current_y, step_y);
      candidates[0U] = {current_x, new_y};
      candidates[1U] = {new_x, current_y};
      candidates[2U] = {new_x, new_y};
      candidate_count = 3U;
      current_x = new_x;
      current_y = new_y;
    } else if (crosses_x) {
      const std::int64_t new_x = CheckedAdd(current_x, step_x);
      if (step_y == 0) {
        candidates[0U] = {new_x, start_y_cells[0U]};
        candidate_count = 1U;
        if (start_y_count == 2U) {
          candidates[1U] = {new_x, start_y_cells[1U]};
          candidate_count = 2U;
        }
      } else {
        candidates[0U] = {new_x, current_y};
        candidate_count = 1U;
      }
      current_x = new_x;
    } else {
      const std::int64_t new_y = CheckedAdd(current_y, step_y);
      if (step_x == 0) {
        candidates[0U] = {start_x_cells[0U], new_y};
        candidate_count = 1U;
        if (start_x_count == 2U) {
          candidates[1U] = {start_x_cells[1U], new_y};
          candidate_count = 2U;
        }
      } else {
        candidates[0U] = {current_x, new_y};
        candidate_count = 1U;
      }
      current_y = new_y;
    }
    emit_group(t, candidates, candidate_count);
    if (stopped) {
      break;
    }

    if (crosses_x) {
      const auto following_distance = CheckedAdd(next_distance_x, grid_unit);
      if (Compare(following_distance, exact_delta_x) <= 0) {
        next_x = CheckedAdd(next_x, step_x);
        next_distance_x = following_distance;
        if (y_active) {
          x_event_product = CheckedAdd(x_event_product, x_product_stride);
          if (instrumentation != nullptr) {
            ++instrumentation->incremental_product_additions;
          }
        }
      } else {
        x_active = false;
      }
    }
    if (crosses_y) {
      const auto following_distance = CheckedAdd(next_distance_y, grid_unit);
      if (Compare(following_distance, exact_delta_y) <= 0) {
        next_y = CheckedAdd(next_y, step_y);
        next_distance_y = following_distance;
        if (x_active) {
          y_event_product = CheckedAdd(y_event_product, y_product_stride);
          if (instrumentation != nullptr) {
            ++instrumentation->incremental_product_additions;
          }
        }
      } else {
        y_active = false;
      }
    }
  }
  return {visited_count};
}

enum class PreTargetContact { kPass, kOccupied, kOutsideTask };

inline PreTargetContact ClassifyPreTargetContact(CellState state,
                                                 GridIndex visited,
                                                 GridIndex target) {
  if (visited == target) {
    return PreTargetContact::kPass;
  }
  if (state == CellState::kOutsideTask) {
    return PreTargetContact::kOutsideTask;
  }
  if (state == CellState::kOccupied) {
    return PreTargetContact::kOccupied;
  }
  return PreTargetContact::kPass;
}

inline void CheckedVisibleIncrement(std::uint32_t& count) {
  if (count == std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("visible unknown count overflow");
  }
  ++count;
}

}  // namespace lunar::pure_exploration::detail
