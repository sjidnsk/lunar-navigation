#include "lunar_pure_exploration_ros/stationary_planning_gate.hpp"

#include <cmath>
#include <stdexcept>

namespace lunar::pure_exploration_ros {

StationaryPlanningGate::StationaryPlanningGate(
    StationaryGateParameters parameters)
    : parameters_(parameters) {
  if (!std::isfinite(parameters_.maximum_linear_speed_mps) ||
      parameters_.maximum_linear_speed_mps < 0.0 ||
      !std::isfinite(parameters_.maximum_angular_speed_radps) ||
      parameters_.maximum_angular_speed_radps < 0.0 ||
      parameters_.confirmation_samples == 0U ||
      parameters_.diagnostic_period <=
          std::chrono::steady_clock::duration::zero()) {
    throw std::invalid_argument("invalid stationary planning gate parameters");
  }
}

void StationaryPlanningGate::Begin(const std::chrono::steady_clock::time_point now,
                                   const SpeedObservation entry_speed) {
  waiting_ = true;
  consecutive_samples_ = 0U;
  entry_speed_ = entry_speed;
  began_at_ = now;
  next_diagnostic_at_ = now + parameters_.diagnostic_period;
}

StationaryGateUpdate StationaryPlanningGate::Observe(
    const std::chrono::steady_clock::time_point now,
    const SpeedObservation observation) {
  StationaryGateUpdate update{.observation = observation};
  if (!waiting_) {
    return update;
  }

  update.elapsed = now - began_at_;
  if (now >= next_diagnostic_at_) {
    update.diagnostic_due = true;
    next_diagnostic_at_ += parameters_.diagnostic_period;
  }

  if (IsStationary(observation)) {
    ++consecutive_samples_;
  } else {
    consecutive_samples_ = 0U;
  }
  update.consecutive_samples = consecutive_samples_;

  if (consecutive_samples_ == parameters_.confirmation_samples) {
    waiting_ = false;
    update.confirmed = true;
  }
  return update;
}

void StationaryPlanningGate::Cancel() noexcept {
  waiting_ = false;
  consecutive_samples_ = 0U;
  entry_speed_ = {};
  began_at_ = {};
  next_diagnostic_at_ = {};
}

bool StationaryPlanningGate::waiting() const noexcept { return waiting_; }

SpeedObservation StationaryPlanningGate::entry_speed() const noexcept {
  return entry_speed_;
}

bool StationaryPlanningGate::IsStationary(
    const SpeedObservation observation) const noexcept {
  return std::isfinite(observation.linear_speed_mps) &&
         std::isfinite(observation.angular_speed_radps) &&
         observation.linear_speed_mps <= parameters_.maximum_linear_speed_mps &&
         observation.angular_speed_radps <=
             parameters_.maximum_angular_speed_radps;
}

}  // namespace lunar::pure_exploration_ros
