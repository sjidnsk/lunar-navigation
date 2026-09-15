#include "lunar_incremental_navigation_ros/elevation_pipeline.hpp"

#include <algorithm>
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
  if (result.status == lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied) {
    pending_fine_dirty_tiles_.insert(result.dirty_tiles.begin(),
                                     result.dirty_tiles.end());
    const auto raw = elevation_.Snapshot();
    pending_fine_dirty_cells_.insert(raw->changed_cells().begin(),
                                     raw->changed_cells().end());
    latest_accepted_raw_revision_ = result.raw_elevation_revision;
    latest_accepted_map_stamp_ns = map_stamp_ns;
  } else if (result.status == lunar::incremental_navigation::ElevationUpdateResult::Status::kDuplicate) {
    const auto published = std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
    if (published->bundle.fine &&
        published->bundle.fine->raw_elevation_revision() == result.raw_elevation_revision) {
      auto replacement = std::make_shared<PipelineSnapshot>(*published);
      replacement->processed_map_stamp_ns = map_stamp_ns;
      std::shared_ptr<const PipelineSnapshot> immutable = std::move(replacement);
      std::atomic_store_explicit(&bundle_, std::move(immutable), std::memory_order_release);
    } else if (latest_accepted_raw_revision_ == result.raw_elevation_revision) {
      latest_accepted_map_stamp_ns = map_stamp_ns;
    }
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
  std::int64_t captured_map_stamp_ns{};
  {
    std::scoped_lock lock{work_mutex_};
    if (pending_fine_dirty_tiles_.empty()) {
      return false;
    }
    raw = elevation_.Snapshot();
    captured_map_stamp_ns = raw->raw_elevation_revision() == latest_accepted_raw_revision_
        ? latest_accepted_map_stamp_ns : 0;
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
  std::vector<lunar::incremental_navigation::TileIndex> export_dirty(
      dirty_tiles.begin(), dirty_tiles.end());
  export_dirty.insert(export_dirty.end(), next->changed_tiles().begin(),
                      next->changed_tiles().end());
  const auto raw_dirty = export_dirty;
  for (const auto tile : raw_dirty) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        const lunar::incremental_navigation::TileIndex halo{.x = tile.x + dx,
                                                             .y = tile.y + dy};
        if (next->geometry().Contains(halo)) export_dirty.push_back(halo);
      }
    }
  }
  std::sort(export_dirty.begin(), export_dirty.end());
  export_dirty.erase(std::unique(export_dirty.begin(), export_dirty.end()),
                     export_dirty.end());
  {
    std::scoped_lock lock{work_mutex_};
    export_journal_.push_back({.revision = next->fine_traversability_revision(),
                               .dirty_tiles = export_dirty});
    if (export_journal_.size() > kExportJournalCapacity) export_journal_.pop_front();
  }
  auto current = std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
  while (true) {
    auto replacement =
        std::make_shared<const PipelineSnapshot>(PipelineSnapshot{
            .bundle = {.fine = next, .guidance = current->bundle.guidance},
            .processed_map_stamp_ns = captured_map_stamp_ns});
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

PolicyMapSnapshotCapture ElevationPipeline::CapturePolicyMapSnapshot(
    const std::uint64_t since_revision) const {
  const auto snapshot = std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
  PolicyMapSnapshotCapture result{.bundle = snapshot->bundle,
                                  .processed_map_stamp_ns = snapshot->processed_map_stamp_ns};
  if (!snapshot->bundle.fine || since_revision == 0U) return result;
  const std::uint64_t revision = snapshot->bundle.fine->fine_traversability_revision();
  if (since_revision == revision) {
    result.full_snapshot = false;
    result.base_revision = since_revision;
    return result;
  }
  std::scoped_lock lock{work_mutex_};
  if (export_journal_.empty() || since_revision + 1U < export_journal_.front().revision ||
      since_revision >= revision) return result;
  result.full_snapshot = false;
  result.base_revision = since_revision;
  for (const auto& entry : export_journal_) {
    if (entry.revision > since_revision && entry.revision <= revision) {
      result.dirty_tiles.insert(result.dirty_tiles.end(), entry.dirty_tiles.begin(), entry.dirty_tiles.end());
    }
  }
  std::sort(result.dirty_tiles.begin(), result.dirty_tiles.end());
  result.dirty_tiles.erase(std::unique(result.dirty_tiles.begin(), result.dirty_tiles.end()), result.dirty_tiles.end());
  return result;
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
