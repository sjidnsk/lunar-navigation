#pragma once

#include <cstddef>
#include <vector>

namespace lunar::incremental_navigation_ros {

struct DemoMotionPose final {
  double x_m{};
  double y_m{};
  double yaw_rad{};
};

class DemoMotionFollower final {
 public:
  DemoMotionFollower(double linear_speed_mps, double angular_speed_radps);

  void SetPath(std::vector<DemoMotionPose> path);
  void Advance(double elapsed_s);

  [[nodiscard]] DemoMotionPose pose() const noexcept;

 private:
  double linear_speed_mps_{};
  double angular_speed_radps_{};
  std::vector<DemoMotionPose> path_;
  std::size_t next_pose_{};
  DemoMotionPose pose_;
};

}  // namespace lunar::incremental_navigation_ros
