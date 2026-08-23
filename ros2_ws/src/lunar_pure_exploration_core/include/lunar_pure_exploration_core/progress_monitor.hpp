#pragma once

#include <chrono>
#include <cstddef>
#include <span>
#include <vector>

#include "lunar_pure_exploration_core/types.hpp"

namespace lunar::pure_exploration {

struct ProgressParameters {
  std::size_t maximum_executable_path_points;
  std::chrono::steady_clock::duration timeout{std::chrono::seconds{30}};
  double minimum_progress_m{0.2};
};

class ProgressMonitor {
 public:
  explicit ProgressMonitor(ProgressParameters parameters);

  void Reset(std::chrono::steady_clock::time_point now,
             std::span<const Vec2> path_polyline, Vec2 position);
  bool Update(std::chrono::steady_clock::time_point now, Vec2 position,
              bool paused);
  double historical_max_arc_length_m() const;

 private:
  static double ProjectArcLength(
      std::span<const Vec2> path_polyline,
      std::span<const long double> cumulative_arc_lengths, Vec2 position);
  bool TimeoutElapsed(std::chrono::steady_clock::time_point now) const;

  ProgressParameters parameters_;
  std::vector<Vec2> path_polyline_;
  std::vector<long double> cumulative_arc_lengths_;
  std::chrono::steady_clock::time_point last_accepted_time_{};
  std::chrono::steady_clock::time_point window_start_{};
  double historical_max_arc_length_m_{0.0};
  double window_baseline_arc_length_m_{0.0};
  bool initialized_{false};
  bool paused_{false};
  bool stuck_latched_{false};
};

}  // namespace lunar::pure_exploration
