#pragma once

#include <cstddef>
#include <vector>

#include "lunar_pure_exploration_sim/lunar_scene.hpp"

namespace lunar::pure_exploration_sim {

struct Pose2 {
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
};

struct SensorModel {
  double range_m{10.0};
  double horizontal_fov_rad{1.5707963267948966};
  double radial_step_m{0.1};
  double angular_step_rad{0.009999666686665238};
};

[[nodiscard]] double GlobalKnownEnvelopeMarginM() noexcept;
[[nodiscard]] double LocalContactEnvelopeRadiusM() noexcept;

class ObservationState {
 public:
  ObservationState();

  void Observe(const LunarScene& scene, Pose2 pose, SensorModel sensor);

  [[nodiscard]] bool IsCurrentlyVisible(const LunarScene& scene,
                                        double world_x_m,
                                        double world_y_m) const;
  [[nodiscard]] bool IsKnownGlobalCell(std::size_t x,
                                       std::size_t y) const noexcept;
  [[nodiscard]] std::size_t KnownGlobalCount() const noexcept;
  [[nodiscard]] const std::vector<bool>& KnownGlobalMask() const noexcept;
  [[nodiscard]] const TruthSample* CurrentLocalSample(
      Pose2 map_pose, std::size_t logical_x,
      std::size_t logical_y) const noexcept;

 private:
  struct CachedLocalCell {
    bool visible{false};
    TruthSample sample{};
  };

  void MarkKnown(const LunarScene& scene, double world_x_m,
                 double world_y_m);
  void CacheCurrentLocalCell(const LunarScene& scene, std::size_t logical_x,
                             std::size_t logical_y);
  [[nodiscard]] bool GlobalFreeCellEnvelopeVisible(
      const LunarScene& scene, double world_x_m, double world_y_m) const;
  void SeedInitialKnownStart(const LunarScene& scene, Pose2 pose);
  void CacheCurrentContactEnvelope(const LunarScene& scene);

  std::vector<bool> known_global_;
  std::vector<CachedLocalCell> current_local_;
  Pose2 current_pose_{};
  SensorModel current_sensor_{};
  bool has_observation_{false};
};

}  // namespace lunar::pure_exploration_sim
