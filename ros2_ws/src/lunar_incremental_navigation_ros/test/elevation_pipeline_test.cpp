#include <atomic>
#include <cmath>
#include <chrono>
#include <map>
#include <limits>
#include <cstddef>
#include <future>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/elevation_pipeline.hpp"
#include "lunar_incremental_navigation_ros/policy_map_exporter.hpp"

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
  bool throw_on_release{true};
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
          if (fine_gate->throw_on_release) {
            throw std::runtime_error("injected fine derivation failure");
          }
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

TEST(ElevationPipelineTest, DuplicateRefreshesPublishedFineStampWithoutChangingMap) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  const auto evidence = Evidence(0.0F);
  ASSERT_EQ(pipeline.ApplyLocal(evidence, 101).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  EXPECT_EQ(pipeline.ApplyLocal(evidence, 202).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kDuplicate);
  EXPECT_EQ(pipeline.CapturePolicyMapSnapshot().processed_map_stamp_ns, 202);
}

TEST(ElevationPipelineTest, DuplicatePendingBeforeDerivationAdvancesOnlyMatchingFine) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F), 101).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.1F), 202).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.1F), 303).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kDuplicate);
  EXPECT_EQ(pipeline.CapturePolicyMapSnapshot().processed_map_stamp_ns, 101);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto published = pipeline.CapturePolicyMapSnapshot();
  EXPECT_EQ(published.bundle.fine->raw_elevation_revision(), 2U);
  EXPECT_EQ(published.processed_map_stamp_ns, 303);
  EXPECT_FALSE(pipeline.RunFineDerivation());
}

class DuplicateDuringDerivation : public testing::TestWithParam<bool> {};

TEST_P(DuplicateDuringDerivation, PublishesExactRawStampPairWithoutNewerRawLeak) {
  FailureGate gate;
  gate.throw_on_release = false;
  ElevationPipeline pipeline(Capability(), Profile(), 1.0,
                             GatedFailureDerivers(&gate, nullptr));
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F), 101).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto original = pipeline.CapturePolicyMapSnapshot();
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.1F), 202).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  gate.armed.store(true);
  auto entered = gate.entered.get_future();
  auto worker = std::async(std::launch::async, [&] { return pipeline.RunFineDerivation(); });
  const auto entered_status = entered.wait_for(std::chrono::seconds(5));
  if (entered_status != std::future_status::ready) {
    gate.release.set_value();
    worker.wait();
    FAIL() << "fine derivation did not reach injected gate";
  }
  EXPECT_EQ(pipeline.ApplyLocal(Evidence(0.1F), 303).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kDuplicate);
  if (GetParam()) {
    EXPECT_EQ(pipeline.ApplyLocal(Evidence(0.2F), 404).status,
              lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
    EXPECT_EQ(pipeline.ApplyLocal(Evidence(0.2F), 505).status,
              lunar::incremental_navigation::ElevationUpdateResult::Status::kDuplicate);
  }
  const AdapterResult<OwnedElevationEvidence> rejected{
      .value = std::nullopt, .reason_code = "INVALID_INPUT"};
  EXPECT_EQ(pipeline.ApplyLocal(rejected, 999).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(pipeline.CapturePolicyMapSnapshot().processed_map_stamp_ns, 101);
  gate.release.set_value();
  ASSERT_TRUE(worker.get());
  const auto published = pipeline.CapturePolicyMapSnapshot();
  EXPECT_EQ(published.bundle.fine->raw_elevation_revision(), 2U);
  EXPECT_EQ(published.processed_map_stamp_ns, GetParam() ? 202 : 303);
  // The previously captured immutable pair cannot change after publication.
  EXPECT_EQ(original.bundle.fine->raw_elevation_revision(), 1U);
  EXPECT_EQ(original.processed_map_stamp_ns, 101);
  if (GetParam()) {
    ASSERT_TRUE(pipeline.RunFineDerivation());
    const auto latest = pipeline.CapturePolicyMapSnapshot();
    EXPECT_EQ(latest.bundle.fine->raw_elevation_revision(), 3U);
    EXPECT_EQ(latest.processed_map_stamp_ns, 505);
  }
  const auto before_rejection = pipeline.CapturePolicyMapSnapshot();
  EXPECT_EQ(pipeline.ApplyLocal(rejected, 1000).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(pipeline.ApplyLocal(OwnedElevationEvidence{}, 1001).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(pipeline.CapturePolicyMapSnapshot().processed_map_stamp_ns,
            before_rejection.processed_map_stamp_ns);
}

INSTANTIATE_TEST_SUITE_P(PendingDuplicate, DuplicateDuringDerivation,
                        testing::Bool());

using PolicyResponse = lunar_planning_msgs::srv::GetPolicyMap::Response;
using TileKey = std::pair<std::int64_t, std::int64_t>;
using WireTiles = std::map<TileKey, lunar_planning_msgs::msg::PolicyMapTile>;

PolicyResponse ExportCaptured(const ElevationPipeline& pipeline,
                              const PolicyMapExporter& exporter,
                              const std::uint64_t since = 0U) {
  const auto capture = pipeline.CapturePolicyMapSnapshot(since);
  return exporter.Export({.fine = capture.bundle.fine,
                          .epoch = "direct-export-regression",
                          .processed_stamp_ns = capture.processed_map_stamp_ns,
                          .base_revision = capture.base_revision,
                          .full_snapshot = capture.full_snapshot,
                          .dirty_tiles = capture.dirty_tiles}, since);
}

void ApplyWireTiles(const PolicyResponse& response, WireTiles& tiles) {
  if (response.full_snapshot) tiles.clear();
  for (const auto& tile : response.tiles) tiles[{tile.tile_x, tile.tile_y}] = tile;
}

void ExpectFloatArraysEqual(const std::vector<float>& actual,
                           const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!(actual[i] == expected[i] ||
          (std::isnan(actual[i]) && std::isnan(expected[i])))) {
      FAIL() << "float array differs at " << i << ": " << actual[i]
             << " vs " << expected[i];
    }
  }
}

void ExpectAllExportedFieldsEqual(const WireTiles& reconstructed,
                                 const PolicyResponse& full) {
  ASSERT_TRUE(full.ready);
  ASSERT_TRUE(full.full_snapshot);
  ASSERT_EQ(reconstructed.size(), full.tiles.size());
  for (const auto& tile : full.tiles) {
    SCOPED_TRACE(testing::Message() << "tile " << tile.tile_x << "," << tile.tile_y);
    const auto found = reconstructed.find({tile.tile_x, tile.tile_y});
    ASSERT_NE(found, reconstructed.end());
    EXPECT_EQ(found->second.states, tile.states);
    EXPECT_EQ(found->second.intrinsic_states, tile.intrinsic_states);
    EXPECT_EQ(found->second.observed, tile.observed);
    ExpectFloatArraysEqual(found->second.costs, tile.costs);
    ExpectFloatArraysEqual(found->second.elevation_m, tile.elevation_m);
  }
}

TEST(PolicyMapPipelineExport, HeightOnlyDeltaMatchesFullWithUnchangedNavigationFields) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  PolicyMapExporter exporter(Capability(), Profile());
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F), 10).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto base = ExportCaptured(pipeline, exporter);
  ASSERT_TRUE(base.ready);
  ASSERT_FALSE(base.tiles.empty());
  WireTiles reconstructed;
  ApplyWireTiles(base, reconstructed);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.1F), 20).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  EXPECT_TRUE(pipeline.CaptureBundle().fine->changed_tiles().empty());
  const auto delta = ExportCaptured(pipeline, exporter, base.fine_revision);
  const auto full = ExportCaptured(pipeline, exporter);
  ASSERT_FALSE(delta.full_snapshot);
  ASSERT_FALSE(delta.tiles.empty());
  EXPECT_EQ(delta.base_revision, base.fine_revision);
  ASSERT_EQ(base.tiles.size(), full.tiles.size());
  EXPECT_EQ(base.tiles[0].states, full.tiles[0].states);
  EXPECT_EQ(base.tiles[0].costs, full.tiles[0].costs);
  EXPECT_FLOAT_EQ(base.tiles[0].elevation_m[0], 0.F);
  EXPECT_FLOAT_EQ(full.tiles[0].elevation_m[0], .1F);
  ApplyWireTiles(delta, reconstructed);
  ExpectAllExportedFieldsEqual(reconstructed, full);
}

TEST(PolicyMapPipelineExport, ObservedOnlyDeltaMatchesFullWithUnchangedNavigationFields) {
  auto capability = Capability();
  auto profile = Profile();
  profile.planar_envelope_xy_m = {{-.3, -.3}, {.3, -.3}, {.3, .3}, {-.3, .3}};
  std::get<WheeledCapability>(capability).footprint_xy_m = profile.planar_envelope_xy_m;
  ElevationPipeline pipeline(capability, profile, 1.0);
  PolicyMapExporter exporter(capability, profile);
  auto evidence = Evidence(0.0F);
  evidence.terrain_measurements.assign(49, {.center_known = true,
      .neighborhood_complete = true, .slope_rad = 0, .relief_m = 2, .positive_rise_m = 2});
  ASSERT_EQ(pipeline.ApplyLocal(evidence, 10).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto base = ExportCaptured(pipeline, exporter);
  WireTiles reconstructed;
  ApplyWireTiles(base, reconstructed);
  evidence.terrain_measurements[24] = {.center_known = true, .neighborhood_complete = true};
  ASSERT_EQ(pipeline.ApplyLocal(evidence, 20).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  EXPECT_TRUE(pipeline.CaptureBundle().fine->changed_tiles().empty());
  const auto delta = ExportCaptured(pipeline, exporter, base.fine_revision);
  const auto full = ExportCaptured(pipeline, exporter);
  ASSERT_FALSE(delta.full_snapshot);
  ASSERT_EQ(base.tiles.size(), 1U);
  ASSERT_EQ(full.tiles.size(), 1U);
  EXPECT_EQ(base.tiles[0].states, full.tiles[0].states);
  EXPECT_EQ(base.tiles[0].costs, full.tiles[0].costs);
  const auto center = 3 * lunar::incremental_navigation::kGridTileWidthCells + 3;
  EXPECT_EQ(base.tiles[0].observed[center], kPolicyMapBlocked);
  EXPECT_EQ(full.tiles[0].observed[center], kPolicyMapFree);
  ApplyWireTiles(delta, reconstructed);
  ExpectAllExportedFieldsEqual(reconstructed, full);
}

TEST(PolicyMapPipelineExport, MeasuredUnknownTileIsExportedWithoutAllocatedNavigationTile) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  PolicyMapExporter exporter(Capability(), Profile());
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.25F, 0.0, 1, 1), 10).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  EXPECT_TRUE(pipeline.CaptureBundle().fine->tile_indices().empty());
  const auto full = ExportCaptured(pipeline, exporter);
  ASSERT_TRUE(full.ready);
  ASSERT_EQ(full.tiles.size(), 1U);
  EXPECT_EQ(full.tiles[0].states[0], kPolicyMapUnknown);
  EXPECT_EQ(full.tiles[0].observed[0], kPolicyMapUnknown);
  EXPECT_FLOAT_EQ(full.tiles[0].elevation_m[0], .25F);
  WireTiles reconstructed;
  ApplyWireTiles(full, reconstructed);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(.5F, 0.0, 1, 1), 20).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(.75F, 120.0, 1, 1), 30).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto delta = ExportCaptured(pipeline, exporter, full.fine_revision);
  ASSERT_FALSE(delta.full_snapshot);
  EXPECT_EQ(delta.tiles.size(), 2U);
  ApplyWireTiles(delta, reconstructed);
  ExpectAllExportedFieldsEqual(reconstructed, ExportCaptured(pipeline, exporter));
}

TEST(PolicyMapPipelineExport, CoalescedRawUpdatesAndSkippedFineRevisionsMatchFull) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  PolicyMapExporter exporter(Capability(), Profile());
  for (const double x : {0., 120., 240.}) {
    ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F, x), 10).status,
              lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  }
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto base = ExportCaptured(pipeline, exporter);
  ASSERT_EQ(base.tiles.size(), 3U);
  WireTiles reconstructed;
  ApplyWireTiles(base, reconstructed);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.1F, 0.), 20).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.2F, 240.), 30).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  EXPECT_TRUE(pipeline.CaptureBundle().fine->changed_tiles().empty());
  const auto coalesced = ExportCaptured(pipeline, exporter, base.fine_revision);
  ASSERT_FALSE(coalesced.full_snapshot);
  ApplyWireTiles(coalesced, reconstructed);
  ExpectAllExportedFieldsEqual(reconstructed, ExportCaptured(pipeline, exporter));
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.3F, 120.), 40).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto skipped = ExportCaptured(pipeline, exporter, base.fine_revision);
  ASSERT_FALSE(skipped.full_snapshot);
  EXPECT_EQ(skipped.base_revision, base.fine_revision);
  EXPECT_EQ(skipped.fine_revision, base.fine_revision + 2U);
  ApplyWireTiles(base, reconstructed);
  ApplyWireTiles(skipped, reconstructed);
  const auto full = ExportCaptured(pipeline, exporter);
  ExpectAllExportedFieldsEqual(reconstructed, full);
  // A second client request does not consume or shorten the first client's history.
  const auto second_client = ExportCaptured(pipeline, exporter, coalesced.fine_revision);
  ASSERT_FALSE(second_client.full_snapshot);
  const auto repeated = ExportCaptured(pipeline, exporter, base.fine_revision);
  ApplyWireTiles(base, reconstructed);
  ApplyWireTiles(repeated, reconstructed);
  ExpectAllExportedFieldsEqual(reconstructed, full);
}

TEST(PolicyMapPipelineExport, JournalRetains128TransitionsThenFallsBackToFull) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  PolicyMapExporter exporter(Capability(), Profile());
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F), 1).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto base = ExportCaptured(pipeline, exporter);
  for (int revision = 2; revision <= 129; ++revision) {
    ASSERT_EQ(pipeline.ApplyLocal(Evidence(static_cast<float>(revision) * .001F), revision).status,
              lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
    ASSERT_TRUE(pipeline.RunFineDerivation());
  }
  const auto retained = ExportCaptured(pipeline, exporter, base.fine_revision);
  ASSERT_FALSE(retained.full_snapshot);
  EXPECT_EQ(retained.fine_revision, 129U);
  WireTiles reconstructed;
  ApplyWireTiles(base, reconstructed);
  ApplyWireTiles(retained, reconstructed);
  ExpectAllExportedFieldsEqual(reconstructed, ExportCaptured(pipeline, exporter));
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(.130F), 130).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto gap = ExportCaptured(pipeline, exporter, base.fine_revision);
  ASSERT_TRUE(gap.full_snapshot);
  ApplyWireTiles(gap, reconstructed);
  ExpectAllExportedFieldsEqual(reconstructed, ExportCaptured(pipeline, exporter));
  const auto recent = ExportCaptured(pipeline, exporter, retained.fine_revision);
  EXPECT_FALSE(recent.full_snapshot);
  EXPECT_EQ(recent.base_revision, retained.fine_revision);
}

TEST(ElevationPipelineTest, JournalDeltaCoversCoalescedAcceptedRawTiles) {
  ElevationPipeline pipeline(Capability(), Profile(), 1.0);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.0F, 0.0), 10).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto base = pipeline.CapturePolicyMapSnapshot();
  ASSERT_TRUE(base.bundle.fine);
  const auto revision = base.bundle.fine->fine_traversability_revision();
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.1F, 0.0), 20).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(pipeline.ApplyLocal(Evidence(0.2F, 60.0), 30).status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(pipeline.RunFineDerivation());
  const auto delta = pipeline.CapturePolicyMapSnapshot(revision);
  EXPECT_FALSE(delta.full_snapshot);
  EXPECT_EQ(delta.base_revision, revision);
  EXPECT_GE(delta.dirty_tiles.size(), 2U);
  EXPECT_EQ(delta.processed_map_stamp_ns, 30);
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
