#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning {

struct ProvisionalGlobalRoute final {
  std::string request_id;
  std::string route_id;
  PlatformType platform_type{PlatformType::kWheeled};
  std::vector<Pose3> poses_map;
};

using ProvisionalRouteObserver =
    std::function<void(const ProvisionalGlobalRoute&)>;

class Planner final {
 public:
  Planner();
  ~Planner();
  Planner(Planner&&) noexcept;
  Planner& operator=(Planner&&) noexcept;
  Planner(const Planner&) = delete;
  Planner& operator=(const Planner&) = delete;

  [[nodiscard]] PlannerOutput Plan(const PlannerInput& input) noexcept;
  [[nodiscard]] PlannerOutput Plan(
      const PlannerInput& input,
      const ProvisionalRouteObserver& provisional_route_observer) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::planning
