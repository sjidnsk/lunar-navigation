#include "lunar_incremental_navigation_ros/elevation_pipeline.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace lunar::incremental_navigation_ros {

ElevationPipeline::ElevationPipeline(
    lunar::incremental_navigation::PlatformCapability capability,
    lunar::incremental_navigation::TraversabilityProfile profile,
    const double coarse_resolution_m,
    ElevationPipelineDerivers derivers)
    : capability_(std::move(capability)),
      profile_(std::move(profile)),
      guidance_builder_(coarse_resolution_m),
      derivers_(std::move(derivers)),
      bundle_(std::make_shared<const PipelineSnapshot>()) {}

lunar::incremental_navigation::ElevationUpdateResult ElevationPipeline::ApplyLocal(
    const OwnedElevationEvidence& evidence, const std::int64_t map_stamp_ns) {
  std::scoped_lock lock{work_mutex_};
  auto result = elevation_.Apply(evidence.View());
  if (result.status ==
      lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied) {
    pending_fine_dirty_tiles_.insert(result.dirty_tiles.begin(),
                                     result.dirty_tiles.end());
    const auto raw = elevation_.Snapshot();
    pending_fine_dirty_cells_.insert(raw->changed_cells().begin(),
                                     raw->changed_cells().end());
    raw_map_stamps_[result.raw_elevation_revision] = map_stamp_ns;
  }
  return result;
}

lunar::incremental_navigation::ElevationUpdateResult ElevationPipeline::ApplyLocal(
    const AdapterResult<OwnedElevationEvidence>& adapted,
    const std::int64_t map_stamp_ns) {
  if (adapted.value) {
    return ApplyLocal(*adapted.value, map_stamp_ns);
  }
  std::scoped_lock lock{work_mutex_};
  return elevation_.Apply(lunar::incremental_navigation::ElevationEvidence{});
}

bool ElevationPipeline::RunFineDerivation() {
  std::scoped_lock worker_lock{fine_derivation_mutex_};
  std::shared_ptr<const lunar::incremental_navigation::ElevationSnapshot> raw;
  std::set<lunar::incremental_navigation::TileIndex> dirty_tiles;
  std::set<lunar::incremental_navigation::GridIndex> dirty_cells;
  {
    std::scoped_lock lock{work_mutex_};
    if (pending_fine_dirty_tiles_.empty()) {
      return false;
    }
    raw = elevation_.Snapshot();
    dirty_tiles.swap(pending_fine_dirty_tiles_);
    dirty_cells.swap(pending_fine_dirty_cells_);
  }
  const auto captured =
      std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
  const auto previous = captured->bundle.fine;
  std::vector<lunar::incremental_navigation::GridIndex> merged_cells(
      dirty_cells.begin(), dirty_cells.end());
  const std::optional<lunar::incremental_navigation::FineElevationChangeSet> changes =
      previous
          ? std::optional<lunar::incremental_navigation::FineElevationChangeSet>(
                lunar::incremental_navigation::FineElevationChangeSet{
                    .base_raw_elevation_revision =
                        previous->raw_elevation_revision(),
                    .changed_cells = merged_cells})
          : std::nullopt;
  std::shared_ptr<const lunar::incremental_navigation::FineTraversabilitySnapshot> next;
  try {
    next = derivers_.fine
               ? derivers_.fine(raw, capability_, profile_, previous, changes)
               : fine_builder_.Derive(raw, capability_, profile_, previous,
                                      changes);
  } catch (...) {
    std::scoped_lock lock{work_mutex_};
    pending_fine_dirty_tiles_.insert(dirty_tiles.begin(), dirty_tiles.end());
    pending_fine_dirty_cells_.insert(dirty_cells.begin(), dirty_cells.end());
    throw;
  }
  std::int64_t processed_map_stamp_ns{};
  {
    std::scoped_lock lock{work_mutex_};
    processed_map_stamp_ns = raw_map_stamps_[next->raw_elevation_revision()];
  }
  auto current = std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
  while (true) {
    auto replacement =
        std::make_shared<const PipelineSnapshot>(PipelineSnapshot{
            .bundle = {.fine = next, .guidance = current->bundle.guidance},
            .processed_map_stamp_ns = processed_map_stamp_ns});
    if (std::atomic_compare_exchange_weak_explicit(
            &bundle_, &current, std::move(replacement),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }
  return next != previous;
}

bool ElevationPipeline::RunGuidanceDerivation() {
  std::scoped_lock worker_lock{guidance_derivation_mutex_};
  const auto captured =
      std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
  const auto fine = captured->bundle.fine;
  if (!fine) {
    return false;
  }
  const auto previous = captured->bundle.guidance;
  {
    std::scoped_lock lock{work_mutex_};
    if (consumed_fine_revision_ == fine->fine_traversability_revision()) {
      return false;
    }
    ++guidance_derivation_count_;
  }
  std::shared_ptr<const lunar::incremental_navigation::GlobalGuidanceSnapshot> next;
  next = derivers_.guidance
             ? derivers_.guidance(fine, previous)
             : guidance_builder_.Derive(fine, std::nullopt, previous);
  {
    std::scoped_lock lock{work_mutex_};
    consumed_fine_revision_ = fine->fine_traversability_revision();
  }
  if (next == previous) {
    return false;
  }
  auto current = std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
  while (true) {
    auto replacement =
        std::make_shared<const PipelineSnapshot>(PipelineSnapshot{
            .bundle = {.fine = current->bundle.fine, .guidance = next},
            .processed_map_stamp_ns = current->processed_map_stamp_ns});
    if (std::atomic_compare_exchange_weak_explicit(
            &bundle_, &current, std::move(replacement),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }
  return true;
}

lunar::incremental_navigation::SnapshotBundle ElevationPipeline::CaptureBundle() const {
  return std::atomic_load_explicit(&bundle_, std::memory_order_acquire)->bundle;
}

PolicyMapSnapshotCapture ElevationPipeline::CapturePolicyMapSnapshot() const {
  const auto snapshot = std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
  return {.bundle = snapshot->bundle,
          .processed_map_stamp_ns = snapshot->processed_map_stamp_ns};
}

std::size_t ElevationPipeline::PendingFineDirtyTileCount() const {
  std::scoped_lock lock{work_mutex_};
  return pending_fine_dirty_tiles_.size();
}

std::uint64_t ElevationPipeline::GuidanceDerivationCount() const {
  std::scoped_lock lock{work_mutex_};
  return guidance_derivation_count_;
}

lunar::incremental_navigation::ElevationMapCounters ElevationPipeline::LocalCounters()
    const {
  return elevation_.Counters();
}

}  // namespace lunar::incremental_navigation_ros
