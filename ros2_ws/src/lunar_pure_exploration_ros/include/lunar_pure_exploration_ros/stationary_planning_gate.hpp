#pragma once

#include <chrono>
#include <cstdint>

namespace lunar::pure_exploration_ros {

struct SpeedObservation final {
  double linear_speed_mps{};
  double angular_speed_radps{};
};

struct StationaryGateParameters final {
  double maximum_linear_speed_mps{0.01};
  double maximum_angular_speed_radps{0.02};
  std::uint32_t confirmation_samples{3U};
  std::chrono::steady_clock::duration diagnostic_period{
      std::chrono::seconds{5}};
};

struct StationaryGateUpdate final {
  bool confirmed{false};
  bool diagnostic_due{false};
  std::uint32_t consecutive_samples{};
  SpeedObservation observation{};
  std::chrono::steady_clock::duration elapsed{};
};

class StationaryPlanningGate final {
 public:
  explicit StationaryPlanningGate(StationaryGateParameters parameters);

  void Begin(std::chrono::steady_clock::time_point now,
             SpeedObservation entry_speed);
  StationaryGateUpdate Observe(std::chrono::steady_clock::time_point now,
                               SpeedObservation observation);
  void Cancel() noexcept;

  [[nodiscard]] bool waiting() const noexcept;
  [[nodiscard]] SpeedObservation entry_speed() const noexcept;

 private:
  [[nodiscard]] bool IsStationary(SpeedObservation observation) const noexcept;

  StationaryGateParameters parameters_;
  bool waiting_{false};
  std::uint32_t consecutive_samples_{};
  SpeedObservation entry_speed_{};
  std::chrono::steady_clock::time_point began_at_{};
  std::chrono::steady_clock::time_point next_diagnostic_at_{};
};

}  // namespace lunar::pure_exploration_ros
