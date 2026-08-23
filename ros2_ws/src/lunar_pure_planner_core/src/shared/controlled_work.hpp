#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"

namespace lunar::pure_planning::shared {

inline constexpr std::size_t kControlCheckStride = 32U;

[[nodiscard]] inline std::optional<std::string_view> StopReason(
    const SearchControl& control) {
  if (control.canceled()) {
    return "REQUEST_CANCELED";
  }
  if (control.expired()) {
    if (control.canceled()) {
      return "REQUEST_CANCELED";
    }
    return "TIMEOUT";
  }
  return std::nullopt;
}

[[nodiscard]] inline bool ControlCheckDue(const std::size_t work_index) {
  return work_index % kControlCheckStride == 0U;
}

template <typename Value>
[[nodiscard]] std::optional<std::string_view> ControlledFill(
    std::vector<Value>* const values, const std::size_t count,
    const Value& value, const SearchControl& control) {
  if (values == nullptr) {
    return "INVALID_INPUT";
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return stopped;
  }
  values->clear();
  values->reserve(count);
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return stopped;
  }
  while (values->size() < count) {
    if (const auto stopped = StopReason(control); stopped.has_value()) {
      return stopped;
    }
    const std::size_t chunk =
        std::min(kControlCheckStride, count - values->size());
    values->insert(values->end(), chunk, value);
  }
  return StopReason(control);
}

}  // namespace lunar::pure_planning::shared
