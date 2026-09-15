#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/policy_map_exporter.hpp"

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"

namespace lunar::incremental_navigation_ros {

TEST(PolicyMapExporter, ExportsOnlyMeasuredIntrinsicClassificationAsObserved) {
  EXPECT_EQ(EffectiveObservedState(
                lunar::incremental_navigation::IntrinsicCellState::kFree,
                0.0F),
            kPolicyMapFree);
  EXPECT_EQ(EffectiveObservedState(
                lunar::incremental_navigation::IntrinsicCellState::kBlocked,
                0.0F),
            kPolicyMapBlocked);
  EXPECT_EQ(EffectiveObservedState(
                lunar::incremental_navigation::IntrinsicCellState::kFree,
                std::numeric_limits<float>::quiet_NaN()),
            kPolicyMapUnknown);
  EXPECT_EQ(EffectiveObservedState(
                lunar::incremental_navigation::IntrinsicCellState::kUnknown,
                0.0F),
            kPolicyMapUnknown);
}

TEST(PolicyMapExporter, UsesFullSnapshotForBootstrapAndMapRevisionChanges) {
  using namespace lunar::incremental_navigation;
  const WheeledCapability wheel{.footprint_xy_m = {{-.05, -.05}, {.05, -.05}, {.05, .05}, {-.05, .05}},
      .body_extent_m = {.x=.1,.y=.1,.z=.1}, .wheel_diameter_m=.2, .wheel_width_m=.05,
      .wheelbase_m=.1, .track_width_m=.1, .minimum_underbody_clearance_m=.1,
      .maximum_local_obstacle_relief_m=.2, .maximum_forward_speed_mps=.2,
      .maximum_reverse_speed_mps=.2, .maximum_spin_rate_radps=1., .maximum_acceleration_mps2=.5,
      .maximum_braking_deceleration_mps2=.5, .maximum_yaw_acceleration_radps2=.5,
      .maximum_lateral_acceleration_mps2=.5, .maximum_curvature_per_m=1.,
      .maximum_slope_rad=.5, .minimum_clearance_m=.0};
  const TraversabilityProfile profile{.planar_envelope_xy_m = wheel.footprint_xy_m,
      .slope_weight=1., .relief_weight=1.};
  PersistentElevationMap elevation;
  const GridGeometry geometry{.frame_id="map", .width=7, .height=7, .resolution_m=.2};
  const std::vector<float> heights(49, 0.F);
  ASSERT_EQ(elevation.Apply(ElevationEvidence{.geometry=geometry, .elevation_m=heights,
      .map_from_source={.parent_frame="map", .child_frame="map"}}).status,
      ElevationUpdateResult::Status::kApplied);
  const auto fine = FineTraversabilityBuilder{}.Derive(elevation.Snapshot(), wheel, profile);
  PolicyMapExporter exporter(wheel, profile);
  const PolicyMapExportInput input{.fine=fine, .epoch="e", .processed_stamp_ns=10};
  const auto bootstrap = exporter.Export(input, 0);
  EXPECT_TRUE(bootstrap.full_snapshot);
  EXPECT_FALSE(bootstrap.tiles.empty());
  const auto refresh = exporter.Export(input, fine->fine_traversability_revision());
  EXPECT_FALSE(refresh.full_snapshot);
  EXPECT_TRUE(refresh.tiles.empty());
}

}  // namespace lunar::incremental_navigation_ros
