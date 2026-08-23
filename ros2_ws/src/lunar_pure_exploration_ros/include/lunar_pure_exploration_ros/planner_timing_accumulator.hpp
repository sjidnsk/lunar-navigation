#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>

namespace lunar::pure_exploration_ros {

struct PlannerTiming {
  std::string request_id;
  double global_elapsed_ms;
  std::uint64_t global_call_count;
  double local_elapsed_ms;
  std::uint64_t local_call_count;
  double total_elapsed_ms;
};

enum class TimingIngestResult : std::uint8_t {
  kAccepted,
  kDuplicate,
  kRejected,
};

class PlannerTimingAccumulator {
 public:
  explicit PlannerTimingAccumulator(std::size_t capacity);
  TimingIngestResult Ingest(
      const diagnostic_msgs::msg::DiagnosticArray& message);
  std::optional<PlannerTiming> Find(std::string_view request_id) const;

 private:
  std::size_t capacity_;
  mutable std::mutex mutex_;
  std::deque<std::string> arrival_order_;
  std::unordered_map<std::string, PlannerTiming> records_;
};

}  // namespace lunar::pure_exploration_ros
