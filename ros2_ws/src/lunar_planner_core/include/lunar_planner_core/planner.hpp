#pragma once

#include <memory>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning {

class Planner final {
 public:
  Planner();
  ~Planner();
  Planner(Planner&&) noexcept;
  Planner& operator=(Planner&&) noexcept;
  Planner(const Planner&) = delete;
  Planner& operator=(const Planner&) = delete;

  [[nodiscard]] PlannerOutput Plan(const PlannerInput& input) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::planning
