#include <string>

#include <gtest/gtest.h>
#include <rmw/qos_profiles.h>

#include "lunar_incremental_navigation_ros/traversability_qos.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

TEST(TraversabilityQosTest, BuildsReliableTransientLocalInputProfile) {
  const auto qos = MakeTraversabilityInputQos("reliable", "transient_local");

  ASSERT_TRUE(qos.has_value());
  const auto profile = qos->get_rmw_qos_profile();
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  EXPECT_EQ(profile.depth, 1U);
}

TEST(TraversabilityQosTest, Task3LocalMapsUseReliableTransientLocalDepthOne) {
  const auto profile = MakeTask3LocalMapInputQos().get_rmw_qos_profile();

  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.depth, 1U);
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
}

TEST(TraversabilityQosTest, BuildsBestEffortVolatileInputProfile) {
  const auto qos = MakeTraversabilityInputQos("best_effort", "volatile");

  ASSERT_TRUE(qos.has_value());
  const auto profile = qos->get_rmw_qos_profile();
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
  EXPECT_EQ(profile.depth, 1U);
}

TEST(TraversabilityQosTest, RejectsUnsupportedInputProfile) {
  EXPECT_FALSE(MakeTraversabilityInputQos("reliable", "volatilee").has_value());
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
