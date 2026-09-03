#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/global_guidance_builder.hpp"
#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"
#include "lunar_incremental_navigation_ros/map_adapters.hpp"

namespace lunar::incremental_navigation_ros {

struct ElevationPipelineDerivers final {
  using Fine = std::function<std::shared_ptr<const
      lunar::incremental_navigation::FineTraversabilitySnapshot>(
      std::shared_ptr<const lunar::incremental_navigation::ElevationSnapshot>,
      const lunar::incremental_navigation::PlatformCapability&,
      const lunar::incremental_navigation::TraversabilityProfile&,
      std::shared_ptr<const
          lunar::incremental_navigation::FineTraversabilitySnapshot>,
      std::optional<lunar::incremental_navigation::FineElevationChangeSet>)>;
  using Guidance = std::function<std::shared_ptr<const
      lunar::incremental_navigation::GlobalGuidanceSnapshot>(
      std::shared_ptr<const
          lunar::incremental_navigation::FineTraversabilitySnapshot>,
      std::shared_ptr<const
          lunar::incremental_navigation::GlobalGuidanceSnapshot>)>;

  Fine fine;
  Guidance guidance;
};

class ElevationPipeline final {
 public:
  ElevationPipeline(lunar::incremental_navigation::PlatformCapability capability,
                    lunar::incremental_navigation::TraversabilityProfile profile,
                    double coarse_resolution_m,
                    ElevationPipelineDerivers derivers = {});

  [[nodiscard]] lunar::incremental_navigation::ElevationUpdateResult ApplyLocal(
      const OwnedElevationEvidence& evidence);
  [[nodiscard]] lunar::incremental_navigation::ElevationUpdateResult ApplyLocal(
      const AdapterResult<OwnedElevationEvidence>& adapted);
  [[nodiscard]] bool RunFineDerivation();
  [[nodiscard]] bool RunGuidanceDerivation();

  [[nodiscard]] lunar::incremental_navigation::SnapshotBundle CaptureBundle() const;
  [[nodiscard]] std::size_t PendingFineDirtyTileCount() const;
  [[nodiscard]] std::uint64_t GuidanceDerivationCount() const;
  [[nodiscard]] lunar::incremental_navigation::ElevationMapCounters LocalCounters()
      const;

 private:
  lunar::incremental_navigation::PlatformCapability capability_;
  lunar::incremental_navigation::TraversabilityProfile profile_;
  lunar::incremental_navigation::PersistentElevationMap elevation_;
  lunar::incremental_navigation::FineTraversabilityBuilder fine_builder_;
  lunar::incremental_navigation::GlobalGuidanceBuilder guidance_builder_;
  ElevationPipelineDerivers derivers_;

  mutable std::mutex work_mutex_;
  std::mutex fine_derivation_mutex_;
  std::mutex guidance_derivation_mutex_;
  std::set<lunar::incremental_navigation::TileIndex> pending_fine_dirty_tiles_;
  std::set<lunar::incremental_navigation::GridIndex> pending_fine_dirty_cells_;
  std::uint64_t consumed_fine_revision_{};
  std::uint64_t guidance_derivation_count_{};
  std::shared_ptr<const lunar::incremental_navigation::SnapshotBundle> bundle_;
};

}  // namespace lunar::incremental_navigation_ros
