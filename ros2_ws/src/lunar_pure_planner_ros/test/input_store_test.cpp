#include <memory>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/input_store.hpp"

namespace lunar::pure_planner_ros {
namespace {

nav_msgs::msg::Odometry::ConstSharedPtr StampedOdometry(const std::int32_t sec) {
  auto message = std::make_shared<nav_msgs::msg::Odometry>();
  message->header.stamp.sec = sec;
  return message;
}

geometry_msgs::msg::TransformStamped DirectMapFromOdom(const double x) {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.translation.x = x;
  transform.transform.rotation.w = 1.0;
  return transform;
}

nav_msgs::msg::OccupancyGrid::ConstSharedPtr GlobalMap(
    const std::int32_t sec) {
  auto message = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  message->header.stamp.sec = sec;
  return message;
}

grid_map_msgs::msg::GridMap::ConstSharedPtr LocalMap(const std::int32_t sec) {
  auto message = std::make_shared<grid_map_msgs::msg::GridMap>();
  message->header.stamp.sec = sec;
  return message;
}

TEST(InputStore, LastArrivalWinsRegardlessOfStamp) {
  InputStore store;
  store.UpdateOdometry(StampedOdometry(100));
  store.UpdateOdometry(StampedOdometry(0));

  EXPECT_EQ(store.Capture().odometry->header.stamp.sec, 0);
}

TEST(InputStore, LocalSequenceAdvancesForEachArrivalRegardlessOfStamp) {
  InputStore store;
  store.UpdateLocal(LocalMap(100));
  const auto first = store.Capture();
  store.UpdateLocal(LocalMap(0));
  const auto second = store.Capture();

  EXPECT_EQ(first.local_sequence, 1U);
  EXPECT_EQ(second.local_sequence, first.local_sequence + 1U);
  ASSERT_TRUE(second.local_map);
  EXPECT_EQ(second.local_map->header.stamp.sec, 0);
}

TEST(InputStore, CapturesOnlyMatchingTokenAndKeepsDuplicateTokenSequence) {
  InputStore store{true};
  store.UpdateLocal(LocalMap(10));
  store.UpdateOdometry(StampedOdometry(9));
  EXPECT_FALSE(store.CaptureSynchronized().has_value());

  store.UpdateOdometry(StampedOdometry(10));
  const auto first = store.CaptureSynchronized();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->local_sequence, 1U);
  EXPECT_EQ(first->local_arrival_sequence, 1U);
  ASSERT_TRUE(first->local_map);
  EXPECT_EQ(first->local_map->header.stamp.sec, 10);

  store.UpdateLocal(LocalMap(10));
  const auto duplicate = store.CaptureSynchronized();
  ASSERT_TRUE(duplicate.has_value());
  EXPECT_EQ(duplicate->local_sequence, first->local_sequence);
  EXPECT_GT(duplicate->local_arrival_sequence, first->local_arrival_sequence);
}

TEST(InputStore, TokenIdempotenceKeepsDuplicateGlobalAndTfSequences) {
  InputStore store{true};
  store.UpdateGlobal(GlobalMap(10));
  tf2_msgs::msg::TFMessage first_tf;
  auto first_transform = DirectMapFromOdom(1.0);
  first_transform.header.stamp.sec = 10;
  first_tf.transforms.push_back(first_transform);
  store.UpdateTf(first_tf);
  const auto first = store.Capture();

  store.UpdateGlobal(GlobalMap(10));
  tf2_msgs::msg::TFMessage duplicate_tf;
  auto duplicate_transform = DirectMapFromOdom(2.0);
  duplicate_transform.header.stamp.sec = 10;
  duplicate_tf.transforms.push_back(duplicate_transform);
  store.UpdateTf(duplicate_tf);
  const auto duplicate = store.Capture();

  EXPECT_EQ(duplicate.global_sequence, first.global_sequence);
  EXPECT_EQ(duplicate.tf_sequence, first.tf_sequence);
  ASSERT_TRUE(duplicate.map_from_odom.has_value());
  EXPECT_DOUBLE_EQ(duplicate.map_from_odom->transform.translation.x, 2.0);

  store.UpdateGlobal(GlobalMap(11));
  tf2_msgs::msg::TFMessage next_tf;
  auto next_transform = DirectMapFromOdom(3.0);
  next_transform.header.stamp.sec = 11;
  next_tf.transforms.push_back(next_transform);
  store.UpdateTf(next_tf);
  const auto next = store.Capture();

  EXPECT_EQ(next.global_sequence, first.global_sequence + 1U);
  EXPECT_EQ(next.tf_sequence, first.tf_sequence + 1U);
}

TEST(InputStore, DefaultModeCountsDuplicateGlobalAndTfArrivals) {
  InputStore store;
  store.UpdateGlobal(GlobalMap(10));
  tf2_msgs::msg::TFMessage transform;
  auto map_from_odom = DirectMapFromOdom(1.0);
  map_from_odom.header.stamp.sec = 10;
  transform.transforms.push_back(map_from_odom);
  store.UpdateTf(transform);
  const auto first = store.Capture();

  store.UpdateGlobal(GlobalMap(10));
  store.UpdateTf(transform);
  const auto duplicate = store.Capture();

  EXPECT_EQ(duplicate.global_sequence, first.global_sequence + 1U);
  EXPECT_EQ(duplicate.tf_sequence, first.tf_sequence + 1U);
}

TEST(InputStore, KeepsOnlyDirectMapFromOdomFromTfMessages) {
  InputStore store;
  tf2_msgs::msg::TFMessage message;
  message.transforms.push_back(DirectMapFromOdom(1.0));
  auto odom_from_base = DirectMapFromOdom(99.0);
  odom_from_base.header.frame_id = "odom";
  odom_from_base.child_frame_id = "base_link";
  message.transforms.push_back(odom_from_base);

  store.UpdateTf(message);
  const auto snapshot = store.Capture();

  ASSERT_TRUE(snapshot.map_from_odom.has_value());
  EXPECT_EQ(snapshot.map_from_odom->transform.translation.x, 1.0);
}

TEST(InputStore, NewerArrivalReplacesEarlierDirectTransformWithoutTimeComparison) {
  InputStore store;
  tf2_msgs::msg::TFMessage first;
  auto early = DirectMapFromOdom(1.0);
  early.header.stamp.sec = 100;
  first.transforms.push_back(early);
  store.UpdateTf(first);
  tf2_msgs::msg::TFMessage second;
  auto late = DirectMapFromOdom(2.0);
  late.header.stamp.sec = 0;
  second.transforms.push_back(late);
  store.UpdateTf(second);

  const auto snapshot = store.Capture();
  ASSERT_TRUE(snapshot.map_from_odom.has_value());
  EXPECT_EQ(snapshot.map_from_odom->header.stamp.sec, 0);
  EXPECT_EQ(snapshot.map_from_odom->transform.translation.x, 2.0);
}

}  // namespace
}  // namespace lunar::pure_planner_ros
