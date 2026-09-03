#pragma once

#include <memory>
#include <optional>

#include "lunar_incremental_navigation_core/global_guidance_snapshot.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"

namespace lunar::incremental_navigation {

struct GlobalElevationPrior final {
  std::shared_ptr<const ElevationSnapshot> elevation;
};

class GlobalGuidanceBuilder final {
 public:
  explicit GlobalGuidanceBuilder(double coarse_resolution_m);

  [[nodiscard]] double coarse_resolution_m() const noexcept;

  [[nodiscard]] std::shared_ptr<const GlobalGuidanceSnapshot> Derive(
      std::shared_ptr<const FineTraversabilitySnapshot> fine,
      std::optional<GlobalElevationPrior> external_prior,
      std::shared_ptr<const GlobalGuidanceSnapshot> previous = nullptr) const;

 private:
  double coarse_resolution_m_{};
};

}  // namespace lunar::incremental_navigation
