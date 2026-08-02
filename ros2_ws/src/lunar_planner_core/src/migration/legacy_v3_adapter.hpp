#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning {

enum class LegacyV3FaultMode : std::uint8_t {
  kNone,
  kThrowingRegistry,
};

class LegacyV3Adapter final {
 public:
  LegacyV3Adapter();
  explicit LegacyV3Adapter(LegacyV3FaultMode fault_mode);
  ~LegacyV3Adapter();
  LegacyV3Adapter(LegacyV3Adapter&&) noexcept;
  LegacyV3Adapter& operator=(LegacyV3Adapter&&) noexcept;
  LegacyV3Adapter(const LegacyV3Adapter&) = delete;
  LegacyV3Adapter& operator=(const LegacyV3Adapter&) = delete;

  [[nodiscard]] PlannerOutput Plan(const PlannerInput& input) noexcept;
  [[nodiscard]] std::size_t fallback_invocations() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::planning
