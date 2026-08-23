#include "lunar_pure_exploration_core/progress_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lunar::pure_exploration {
namespace {

using Clock = std::chrono::steady_clock;
using DurationRep = Clock::duration::rep;
using UnsignedDurationRep = std::make_unsigned_t<DurationRep>;

static_assert(std::is_integral_v<DurationRep> &&
              std::is_signed_v<DurationRep>);

bool IsFinite(Vec2 point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

}  // namespace

ProgressMonitor::ProgressMonitor(ProgressParameters parameters)
    : parameters_(parameters) {
  if (parameters_.maximum_executable_path_points == 0U ||
      parameters_.timeout <= Clock::duration::zero() ||
      !std::isfinite(parameters_.minimum_progress_m) ||
      parameters_.minimum_progress_m <= 0.0) {
    throw std::invalid_argument("progress parameters must be positive");
  }
}

void ProgressMonitor::Reset(Clock::time_point now,
                            std::span<const Vec2> path_polyline,
                            Vec2 position) {
  if (path_polyline.empty()) {
    throw std::invalid_argument("progress path must not be empty");
  }
  if (path_polyline.size() >
      parameters_.maximum_executable_path_points) {
    throw std::length_error("progress path point limit exceeded");
  }
  for (const Vec2 point : path_polyline) {
    if (!IsFinite(point)) {
      throw std::invalid_argument("progress path points must be finite");
    }
  }
  if (!IsFinite(position)) {
    throw std::invalid_argument("progress position must be finite");
  }
  if (paused_) {
    throw std::logic_error("cannot reset progress while paused");
  }
  if (initialized_ && now < last_accepted_time_) {
    throw std::invalid_argument("progress time must not move backward");
  }

  std::vector<Vec2> owned_path(path_polyline.begin(), path_polyline.end());
  std::vector<long double> cumulative_arc_lengths;
  cumulative_arc_lengths.reserve(owned_path.size());
  cumulative_arc_lengths.push_back(0.0L);
  long double cumulative = 0.0L;
  const long double maximum_double =
      static_cast<long double>(std::numeric_limits<double>::max());
  for (std::size_t index = 1U; index < owned_path.size(); ++index) {
    const long double dx = static_cast<long double>(owned_path[index].x) -
                           static_cast<long double>(owned_path[index - 1U].x);
    const long double dy = static_cast<long double>(owned_path[index].y) -
                           static_cast<long double>(owned_path[index - 1U].y);
    const long double segment_length = std::hypot(dx, dy);
    if (!std::isfinite(segment_length) || segment_length > maximum_double) {
      throw std::overflow_error("progress path segment length overflow");
    }
    cumulative += segment_length;
    if (!std::isfinite(cumulative) || cumulative > maximum_double) {
      throw std::overflow_error("progress path cumulative length overflow");
    }
    cumulative_arc_lengths.push_back(cumulative);
  }

  const double projected_arc =
      ProjectArcLength(owned_path, cumulative_arc_lengths, position);

  path_polyline_ = std::move(owned_path);
  cumulative_arc_lengths_ = std::move(cumulative_arc_lengths);
  historical_max_arc_length_m_ = projected_arc;
  window_baseline_arc_length_m_ = projected_arc;
  last_accepted_time_ = now;
  window_start_ = now;
  initialized_ = true;
  paused_ = false;
  stuck_latched_ = false;
}

bool ProgressMonitor::Update(Clock::time_point now, Vec2 position,
                             bool paused) {
  if (!initialized_) {
    throw std::logic_error("progress monitor has not been reset");
  }
  if (!IsFinite(position)) {
    throw std::invalid_argument("progress position must be finite");
  }
  if (now < last_accepted_time_) {
    throw std::invalid_argument("progress time must not move backward");
  }

  if (paused) {
    paused_ = true;
    last_accepted_time_ = now;
    return stuck_latched_;
  }

  if (paused_) {
    const double projected_arc = ProjectArcLength(
        path_polyline_, cumulative_arc_lengths_, position);
    historical_max_arc_length_m_ = projected_arc;
    window_baseline_arc_length_m_ = projected_arc;
    window_start_ = now;
    last_accepted_time_ = now;
    paused_ = false;
    stuck_latched_ = false;
    return false;
  }

  const double projected_arc =
      ProjectArcLength(path_polyline_, cumulative_arc_lengths_, position);
  const double new_historical_max =
      std::max(historical_max_arc_length_m_, projected_arc);
  if (stuck_latched_) {
    historical_max_arc_length_m_ = new_historical_max;
    last_accepted_time_ = now;
    return true;
  }

  historical_max_arc_length_m_ = new_historical_max;
  if (historical_max_arc_length_m_ - window_baseline_arc_length_m_ >=
      parameters_.minimum_progress_m) {
    window_baseline_arc_length_m_ = historical_max_arc_length_m_;
    window_start_ = now;
  } else if (TimeoutElapsed(now)) {
    stuck_latched_ = true;
  }
  last_accepted_time_ = now;
  return stuck_latched_;
}

double ProgressMonitor::historical_max_arc_length_m() const {
  if (!initialized_) {
    throw std::logic_error("progress monitor has not been reset");
  }
  return historical_max_arc_length_m_;
}

double ProgressMonitor::ProjectArcLength(
    std::span<const Vec2> path_polyline,
    std::span<const long double> cumulative_arc_lengths, Vec2 position) {
  if (path_polyline.size() == 1U) {
    return 0.0;
  }

  long double nearest_distance_squared =
      std::numeric_limits<long double>::infinity();
  long double selected_arc = 0.0L;
  for (std::size_t index = 0U; index + 1U < path_polyline.size(); ++index) {
    const long double ax =
        static_cast<long double>(path_polyline[index].x);
    const long double ay =
        static_cast<long double>(path_polyline[index].y);
    const long double dx =
        static_cast<long double>(path_polyline[index + 1U].x) - ax;
    const long double dy =
        static_cast<long double>(path_polyline[index + 1U].y) - ay;
    const long double length_squared = dx * dx + dy * dy;
    long double fraction = 0.0L;
    if (length_squared > 0.0L) {
      const long double offset_x = static_cast<long double>(position.x) - ax;
      const long double offset_y = static_cast<long double>(position.y) - ay;
      fraction = std::clamp((offset_x * dx + offset_y * dy) /
                                length_squared,
                            0.0L, 1.0L);
    }
    const long double projected_x = ax + fraction * dx;
    const long double projected_y = ay + fraction * dy;
    const long double error_x =
        static_cast<long double>(position.x) - projected_x;
    const long double error_y =
        static_cast<long double>(position.y) - projected_y;
    const long double distance_squared =
        error_x * error_x + error_y * error_y;
    const long double segment_length =
        cumulative_arc_lengths[index + 1U] - cumulative_arc_lengths[index];
    const long double arc =
        cumulative_arc_lengths[index] + fraction * segment_length;
    if (distance_squared < nearest_distance_squared ||
        (distance_squared == nearest_distance_squared && arc > selected_arc)) {
      nearest_distance_squared = distance_squared;
      selected_arc = arc;
    }
  }

  if (!std::isfinite(selected_arc) ||
      selected_arc >
          static_cast<long double>(std::numeric_limits<double>::max())) {
    throw std::overflow_error("projected progress arc length overflow");
  }
  return static_cast<double>(selected_arc);
}

bool ProgressMonitor::TimeoutElapsed(Clock::time_point now) const {
  const auto now_count = now.time_since_epoch().count();
  const auto start_count = window_start_.time_since_epoch().count();
  const UnsignedDurationRep elapsed =
      static_cast<UnsignedDurationRep>(now_count) -
      static_cast<UnsignedDurationRep>(start_count);
  const UnsignedDurationRep timeout =
      static_cast<UnsignedDurationRep>(parameters_.timeout.count());
  return elapsed >= timeout;
}

}  // namespace lunar::pure_exploration
