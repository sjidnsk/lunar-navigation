#include <atomic>
#include <cmath>
#include <cstddef>
#include <future>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/elevation_pipeline.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

using lunar::incremental_navigation::ElevationEvidence;
using lunar::incremental_navigation::GridGeometry;
using lunar::incremental_navigation::PlatformCapability;
using lunar::incremental_navigation::RigidTransform;
using lunar::incremental_navigation::TraversabilityProfile;
using lunar::incremental_navigation::WheeledCapability;

[[nodiscard]] OwnedElevationEvidence Evidence(
    const float value, const double origin_x_m = 0.0,
    const std::size_t width = 7U, const std::size_t height = 7U) {
  OwnedElevationEvidence evidence;
  evidence.geometry = GridGeometry{.frame_id = "map",
                                   .width = width,
                                   .height = height,
                                   .resolution_m = 0.2,
                                   .origin_m = {.x = origin_x_m}};
  evidence.elevation_m.assign(width * height, value);
  evidence.map_from_source = RigidTransform{
      .parent_frame = "map", .child_frame = "map", .rotation = {.w = 1.0}};
  return evidence;
}

[[nodiscard]] OwnedElevationEvidence BlockedTerrainEvidence(
    const double origin_x_m = 0.0) {
  auto evidence = Evidence(0.0F, origin_x_m);
  for (std::size_t index = 0U; index < evidence.elevation_m.size(); ++index) {
    evidence.elevation_m[index] = index % 2U == 0U ? 0.0F : 10.0F;
  }
  return evidence;
}

struct FailureGate final {
  std::atomic<bool> armed{false};
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> released{release.get_future().share()};
};

[[nodiscard]] ElevationPipelineDerivers GatedFailureDerivers(
    FailureGate* fine_gate, FailureGate* guidance_gate) {
  return ElevationPipelineDerivers{
      .fine = [fine_gate](
                  std::shared_ptr<const lunar::incremental_navigation::ElevationSnapshot>
                      raw,
                  const PlatformCapability& capability,
                  const TraversabilityProfile& profile,
                  std::shared_ptr<const
                      lunar::incremental_navigation::FineTraversabilitySnapshot>
                      previous,
                  std::optional<lunar::incremental_navigation::FineElevationChangeSet>
                      changes) {
        if (fine_gate && fine_gate->armed.exchange(false)) {
          fine_gate->entered.set_value();
          fine_gate->released.wait();
          throw std::runtime_error("injected fine derivation failure");
        }
        return lunar::incremental_navigation::FineTraversabilityBuilder{}.Derive(
            std::move(raw), capability, profile, std::move(previous), changes);
      },
      .guidance = [guidance_gate](
                      std::shared_ptr<const lunar::incremental_navigation::
                          FineTraversabilitySnapshot> fine,
                      std::shared_ptr<const lunar::incremental_navigation::
                          GlobalGuidanceSnapshot> previous) {
        if (guidance_gate && guidance_gate->armed.exchange(false)) {
          guidance_gate->entered.set_value();
          guidance_gate->released.wait();
          throw std::runtime_error("injected guidance derivation failure");
        }
        return lunar::incremental_navigation::GlobalGuidanceBuilder{1.0}.Derive(
            std::move(fine), std::nullopt, std::move(previous));
      }};
}

[[nodiscard]] PlatformCapability Capability() {
  return WheeledCapability{
      .footprint_xy_m = {{-0.05, -0.05}, {0.05, -0.05},
                         {0.05, 0.05}, {-0.05, 0.05}},
      .body_extent_m = {.x = 0.1, .y = 0.1, .z = 0.1},
      .wheel_diameter_m = 0.2,
      .wheel_width_m = 0.05,
      .wheelbase_m = 0.1,
      .track_width_m = 0.1,
      .minimum_underbody_clearance_m = 0.1,
      .maximum_local_obstacle_relief_m = 0.2,
      .allow_unsupported_gap = false,
      .minimum_body_z_m = -1.0,
      .maximum_body_z_m = 1.0,
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 1.0,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 1.0,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .maximum_lateral_acceleration_mps2 = 1.0,
      .maximum_curvature_per_m = 1.0,
      .maximum_slope_rad = std::numbers::pi / 4.0,
      .minimum_clearance_m = 0.0,
  };
}

[[nodiscard]] TraversabilityProfile Profile() {
  return TraversabilityProfile{
      .planar_envelope_xy_m = {{-0.05, -0.05}, {0.05, -0.05},
                               {0.05, 0.05}, {-0.05, 0.05}},
      .preferred_clearance_m = 0.0,
      .slope_weight = 1.0,
      .relief_weight = 1.0,
      .clearance_weight = 0.0,
      .start_blind_zone_margin_m = 0.0,
  };
}

TEST(ElevationPipelineTest, CallbacksOnlyWriteRawAndMergeDirtyWork) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  const auto first = pipeline.ApplyLocal(Evidence(0.0F));
  const auto second = pipeline.ApplyLocal(Evidence(0.1F, 0.2));

  EXPECT_EQ(first.status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  EXPECT_EQ(second.status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  EXPECT_EQ(pipeline.CaptureBundle().fine, nullptr);
  EXPECT_EQ(pipeline.PendingFineDirtyTileCount(), 1U);

  EXPECT_TRUE(pipeline.RunFineDerivation());
  const auto bundle = pipeline.CaptureBundle();
  ASSERT_TRUE(bundle.fine);
  EXPECT_EQ(bundle.fine->raw_elevation_revision(), 2U);
  EXPECT_EQ(bundle.fine->fine_traversability_revision(), 1U);
  EXPECT_EQ(bundle.guidance, nullptr);
  EXPECT_EQ(pipeline.PendingFineDirtyTileCount(), 0U);
}

TEST(ElevationPipelineTest, FineSnapshotRetainsItsExactAcceptedMapStamp) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F), 101).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.1F), 202).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);

  const auto captured = pipeline.CapturePolicyMapSnapshot();
  ASSERT_TRUE(captured.bundle.fine);
  EXPECT_EQ(captured.processed_map_stamp_ns, 101);
}

TEST(ElevationPipelineTest,
     MultipleRawRevisionsUseMergedDirtyHaloAndShareUnaffectedFineTile) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F, 60.0)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto first = pipeline.CaptureBundle();
  const auto far_tile = first.fine->FindTile({.x = 1, .y = 0});
  ASSERT_TRUE(far_tile);

  ASSERT_EQ(pipeline.ApplyLocal(BlockedTerrainEvidence(0.0)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(pipeline.ApplyLocal(BlockedTerrainEvidence(20.0)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto second = pipeline.CaptureBundle();

  EXPECT_EQ(second.fine->raw_elevation_revision(), 4U);
  EXPECT_EQ(second.fine->FindTile({.x = 1, .y = 0}), far_tile);
  EXPECT_LT(second.fine->metrics().updated_cells,
            second.fine->geometry().CellCount());
  EXPECT_LT(second.fine->metrics().elevation_cells_examined,
            second.fine->geometry().CellCount());
  EXPECT_EQ(second.fine->State({.x = 3, .y = 3}),
            lunar::incremental_navigation::FineCellState::kBlocked);
  EXPECT_EQ(second.fine->State({.x = 103, .y = 3}),
            lunar::incremental_navigation::FineCellState::kBlocked);
}

TEST(ElevationPipelineTest, FailedFineWorkerRestoresBatchForRetry) {
  FailureGate fine_gate;
  ElevationPipeline pipeline(Capability(), Profile(), 1.0,
                             GatedFailureDerivers(&fine_gate, nullptr));
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto before = pipeline.CaptureBundle();
  ASSERT_EQ(pipeline.ApplyLocal(BlockedTerrainEvidence(0.0)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  fine_gate.armed.store(true);
  auto entered = fine_gate.entered.get_future();
  std::exception_ptr failure;
  std::thread worker([&] {
    try {
      static_cast<void>(pipeline.RunFineDerivation());
    } catch (...) {
      failure = std::current_exception();
    }
  });
  entered.wait();
  const auto concurrent_status =
      pipeline.ApplyLocal(BlockedTerrainEvidence(20.0)).status;
  fine_gate.release.set_value();
  worker.join();

  EXPECT_EQ(concurrent_status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  EXPECT_NE(failure, nullptr);
  EXPECT_EQ(pipeline.CaptureBundle().fine, before.fine);
  EXPECT_GT(pipeline.PendingFineDirtyTileCount(), 0U);

  EXPECT_TRUE(pipeline.RunFineDerivation());
  const auto after = pipeline.CaptureBundle();
  EXPECT_EQ(after.fine->State({.x = 3, .y = 3}),
            lunar::incremental_navigation::FineCellState::kBlocked);
  EXPECT_EQ(after.fine->State({.x = 103, .y = 3}),
            lunar::incremental_navigation::FineCellState::kBlocked);
  EXPECT_EQ(pipeline.PendingFineDirtyTileCount(), 0U);
}

TEST(ElevationPipelineTest, DuplicateDoesNotScheduleOrAdvanceFineRevision) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  const auto evidence = Evidence(0.0F);
  ASSERT_EQ(pipeline.ApplyLocal(evidence).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto first = pipeline.CaptureBundle();

  EXPECT_EQ(pipeline.ApplyLocal(evidence).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kDuplicate);
  EXPECT_EQ(pipeline.PendingFineDirtyTileCount(), 0U);
  EXPECT_FALSE(pipeline.RunFineDerivation());
  const auto duplicate = pipeline.CaptureBundle();
  EXPECT_EQ(duplicate.fine, first.fine);
  EXPECT_EQ(pipeline.LocalCounters().duplicate_updates, 1U);
}

TEST(ElevationPipelineTest, RejectedAdapterInputPreservesPublishedSnapshots) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto before = pipeline.CaptureBundle();
  const AdapterResult<OwnedElevationEvidence> missing_elevation{
      .value = std::nullopt, .reason_code = "INVALID_INPUT"};

  const auto rejected = pipeline.ApplyLocal(missing_elevation);

  EXPECT_EQ(rejected.status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(pipeline.CaptureBundle().fine, before.fine);
  EXPECT_EQ(pipeline.PendingFineDirtyTileCount(), 0U);
  EXPECT_EQ(pipeline.LocalCounters().rejected_updates, 1U);
}

TEST(ElevationPipelineTest,
     FineAndGuidanceWorkersPublishContentChangesIndependentlyWithoutPrior) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  EXPECT_EQ(pipeline.CaptureBundle().guidance, nullptr);

  EXPECT_TRUE(pipeline.RunGuidanceDerivation());
  const auto first = pipeline.CaptureBundle();
  ASSERT_TRUE(first.guidance);
  EXPECT_EQ(first.guidance->source_fine_traversability_revision(), 1U);
  EXPECT_EQ(first.guidance->global_guidance_revision(), 1U);

  ASSERT_EQ(pipeline.ApplyLocal(BlockedTerrainEvidence()).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto fine_only = pipeline.CaptureBundle();
  EXPECT_EQ(fine_only.fine->fine_traversability_revision(), 2U);
  EXPECT_EQ(fine_only.guidance, first.guidance);

  EXPECT_TRUE(pipeline.RunGuidanceDerivation());
  const auto second = pipeline.CaptureBundle();
  EXPECT_EQ(second.guidance->source_fine_traversability_revision(), 2U);
  EXPECT_EQ(second.guidance->global_guidance_revision(), 2U);
}

TEST(ElevationPipelineTest,
     GuidanceUsesConfiguredCoarseResolutionWithoutExternalPrior) {
  ElevationPipeline pipeline(Capability(), Profile(), 0.8);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());

  ASSERT_TRUE(pipeline.RunGuidanceDerivation());
  const auto bundle = pipeline.CaptureBundle();

  ASSERT_TRUE(bundle.guidance);
  EXPECT_DOUBLE_EQ(bundle.guidance->geometry().resolution_m(), 0.8);
  EXPECT_EQ(bundle.guidance->external_elevation_prior(), nullptr);
}

TEST(ElevationPipelineTest,
     ConsumedFineRevisionDoesNotTriggerRepeatedUnchangedGuidanceWork) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  ASSERT_TRUE(pipeline.RunGuidanceDerivation());
  ASSERT_EQ(pipeline.GuidanceDerivationCount(), 1U);

  ASSERT_EQ(pipeline.ApplyLocal(Evidence(1.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  EXPECT_FALSE(pipeline.RunGuidanceDerivation());
  EXPECT_EQ(pipeline.GuidanceDerivationCount(), 2U);

  EXPECT_FALSE(pipeline.RunGuidanceDerivation());
  EXPECT_EQ(pipeline.GuidanceDerivationCount(), 2U);
}

TEST(ElevationPipelineTest, ConcurrentCaptureNeverObservesGuidanceAheadOfFine) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  ASSERT_TRUE(pipeline.RunGuidanceDerivation());
  std::atomic<bool> stop{false};
  std::atomic<bool> inconsistent{false};
  bool worker_failed = false;

  std::thread observer([&] {
    while (!stop.load()) {
      const auto bundle = pipeline.CaptureBundle();
      if (bundle.fine && bundle.guidance &&
          bundle.guidance->source_fine_traversability_revision() >
              bundle.fine->fine_traversability_revision()) {
        inconsistent.store(true);
        return;
      }
    }
  });
  for (std::size_t iteration = 0U; iteration < 64U; ++iteration) {
    const auto evidence = iteration % 2U == 0U ? BlockedTerrainEvidence()
                                                : Evidence(0.0F);
    if (pipeline.ApplyLocal(evidence).status !=
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied ||
        !pipeline.RunFineDerivation()) {
      worker_failed = true;
      break;
    }
    static_cast<void>(pipeline.RunGuidanceDerivation());
  }
  stop.store(true);
  observer.join();

  EXPECT_FALSE(worker_failed);
  EXPECT_FALSE(inconsistent.load());
}

TEST(ElevationPipelineTest, CapturedBundleRemainsImmutableAcrossLaterWorkers) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F)).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  ASSERT_TRUE(pipeline.RunGuidanceDerivation());
  const auto captured = pipeline.CaptureBundle();
  const auto old_raw_revision = captured.fine->raw_elevation_revision();
  const auto old_guidance_revision =
      captured.guidance->global_guidance_revision();

  ASSERT_EQ(pipeline.ApplyLocal(BlockedTerrainEvidence()).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  ASSERT_TRUE(pipeline.RunGuidanceDerivation());
  const auto latest = pipeline.CaptureBundle();

  EXPECT_NE(latest.fine, captured.fine);
  EXPECT_NE(latest.guidance, captured.guidance);
  EXPECT_EQ(captured.fine->raw_elevation_revision(), old_raw_revision);
  EXPECT_EQ(captured.guidance->global_guidance_revision(),
            old_guidance_revision);
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
