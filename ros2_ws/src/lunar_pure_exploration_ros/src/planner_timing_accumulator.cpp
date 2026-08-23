#include "lunar_pure_exploration_ros/planner_timing_accumulator.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>

namespace lunar::pure_exploration_ros {
namespace {

using FieldMap = std::unordered_map<std::string_view, std::string_view>;

constexpr std::array<std::string_view, 10U> kTimingKeys{
    "request_id",         "platform_type",      "environment_mode",
    "planning_outcome",   "reason_code",        "global_elapsed_ms",
    "global_call_count",  "local_elapsed_ms",   "local_call_count",
    "total_elapsed_ms",
};

bool IsKnownKey(const std::string_view key) {
  for (const std::string_view known : kTimingKeys) {
    if (key == known) {
      return true;
    }
  }
  return false;
}

template <typename Unsigned>
bool ParseUnsigned(const std::string_view text, Unsigned& output) {
  if (text.empty()) {
    return false;
  }
  Unsigned parsed{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
  if (error != std::errc{} || end != text.data() + text.size()) {
    return false;
  }
  output = parsed;
  return true;
}

bool ParseElapsed(const std::string_view text, double& output) {
  if (text.empty() || text.front() == '+' || text.front() == '-') {
    return false;
  }
  double parsed{};
  const auto [end, error] = std::from_chars(
      text.data(), text.data() + text.size(), parsed, std::chars_format::general);
  if (error != std::errc{} || end != text.data() + text.size() ||
      !std::isfinite(parsed) || parsed < 0.0) {
    return false;
  }
  output = parsed;
  return true;
}

std::optional<PlannerTiming> ParseTiming(
    const diagnostic_msgs::msg::DiagnosticArray& message) {
  if (message.status.size() != 1U ||
      message.status.front().values.size() != kTimingKeys.size()) {
    return std::nullopt;
  }

  FieldMap fields;
  fields.reserve(kTimingKeys.size());
  for (const auto& entry : message.status.front().values) {
    if (!IsKnownKey(entry.key) ||
        !fields.emplace(entry.key, entry.value).second) {
      return std::nullopt;
    }
  }
  for (const std::string_view key : kTimingKeys) {
    if (!fields.contains(key)) {
      return std::nullopt;
    }
  }

  const std::string_view request_id = fields.at("request_id");
  const std::string_view reason_code = fields.at("reason_code");
  const std::string_view platform_type = fields.at("platform_type");
  if (request_id.empty() || reason_code.empty() ||
      (platform_type != "WHEELED" && platform_type != "LEGGED" &&
       platform_type != "HOPPER")) {
    return std::nullopt;
  }

  std::uint8_t environment_mode{};
  std::uint8_t planning_outcome{};
  if (!ParseUnsigned(fields.at("environment_mode"), environment_mode) ||
      (environment_mode != 1U && environment_mode != 2U) ||
      !ParseUnsigned(fields.at("planning_outcome"), planning_outcome) ||
      planning_outcome > 9U) {
    return std::nullopt;
  }

  PlannerTiming timing{.request_id = std::string{request_id},
                       .global_elapsed_ms = 0.0,
                       .global_call_count = 0U,
                       .local_elapsed_ms = 0.0,
                       .local_call_count = 0U,
                       .total_elapsed_ms = 0.0};
  if (!ParseElapsed(fields.at("global_elapsed_ms"),
                    timing.global_elapsed_ms) ||
      !ParseUnsigned(fields.at("global_call_count"),
                     timing.global_call_count) ||
      !ParseElapsed(fields.at("local_elapsed_ms"), timing.local_elapsed_ms) ||
      !ParseUnsigned(fields.at("local_call_count"), timing.local_call_count) ||
      !ParseElapsed(fields.at("total_elapsed_ms"), timing.total_elapsed_ms)) {
    return std::nullopt;
  }
  return timing;
}

}  // namespace

PlannerTimingAccumulator::PlannerTimingAccumulator(const std::size_t capacity)
    : capacity_(capacity) {
  if (capacity_ == 0U) {
    throw std::invalid_argument{"planner timing capacity must be positive"};
  }
  records_.reserve(capacity_);
}

TimingIngestResult PlannerTimingAccumulator::Ingest(
    const diagnostic_msgs::msg::DiagnosticArray& message) {
  auto parsed = ParseTiming(message);
  if (!parsed.has_value()) {
    return TimingIngestResult::kRejected;
  }

  std::scoped_lock lock{mutex_};
  if (records_.contains(parsed->request_id)) {
    return TimingIngestResult::kDuplicate;
  }
  if (records_.size() == capacity_) {
    records_.erase(arrival_order_.front());
    arrival_order_.pop_front();
  }
  std::string request_id = parsed->request_id;
  arrival_order_.push_back(request_id);
  records_.emplace(std::move(request_id), std::move(*parsed));
  return TimingIngestResult::kAccepted;
}

std::optional<PlannerTiming> PlannerTimingAccumulator::Find(
    const std::string_view request_id) const {
  std::scoped_lock lock{mutex_};
  const auto found = records_.find(std::string{request_id});
  if (found == records_.end()) {
    return std::nullopt;
  }
  return found->second;
}

}  // namespace lunar::pure_exploration_ros
