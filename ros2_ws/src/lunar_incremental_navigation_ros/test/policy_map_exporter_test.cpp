#include <limits>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/policy_map_exporter.hpp"

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

}  // namespace lunar::incremental_navigation_ros
