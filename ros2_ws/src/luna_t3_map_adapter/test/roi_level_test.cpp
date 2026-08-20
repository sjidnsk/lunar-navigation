#include "luna_t3_map_adapter/roi_level.hpp"

#include <gtest/gtest.h>

#include <string>

namespace luna::task3 {
namespace {

using lunar::planning::GlobalMapConfig;

TEST(RoiLevelTest, SelectsFinestAdmissibleLevelForTaskRoi) {
  const GlobalMapConfig config{};

  const auto at_100_m = SelectGlobalLevel({0.0, 0.0, 100.0, 100.0}, config);
  ASSERT_TRUE(at_100_m.ok());
  EXPECT_EQ(at_100_m.value->level, 1U);
  EXPECT_EQ(at_100_m.value->width, 250U);
  EXPECT_EQ(at_100_m.value->height, 250U);
  EXPECT_DOUBLE_EQ(at_100_m.value->resolution_m, 0.4);

  const auto at_300_m = SelectGlobalLevel({0.0, 0.0, 300.0, 300.0}, config);
  ASSERT_TRUE(at_300_m.ok());
  EXPECT_EQ(at_300_m.value->level, 3U);
  EXPECT_EQ(at_300_m.value->width, 188U);
  EXPECT_EQ(at_300_m.value->height, 188U);
  EXPECT_DOUBLE_EQ(at_300_m.value->resolution_m, 1.6);

  const auto at_500_m = SelectGlobalLevel({0.0, 0.0, 500.0, 500.0}, config);
  ASSERT_TRUE(at_500_m.ok());
  EXPECT_EQ(at_500_m.value->level, 4U);
  EXPECT_EQ(at_500_m.value->width, 157U);
  EXPECT_EQ(at_500_m.value->height, 157U);
  EXPECT_DOUBLE_EQ(at_500_m.value->resolution_m, 3.2);
}

TEST(RoiLevelTest, RejectsInvalidAndTooLargeTaskRoi) {
  const GlobalMapConfig config{};

  const auto invalid = SelectGlobalLevel({0.0, 0.0, 0.0, 100.0}, config);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.reason_code, "TASK_ROI_INVALID");

  const auto too_large =
      SelectGlobalLevel({0.0, 0.0, 1024.1, 10.0}, config);
  ASSERT_FALSE(too_large.ok());
  EXPECT_EQ(too_large.reason_code, "GLOBAL_MAP_SCALE_UNSUPPORTED");
}

}  // namespace
}  // namespace luna::task3
