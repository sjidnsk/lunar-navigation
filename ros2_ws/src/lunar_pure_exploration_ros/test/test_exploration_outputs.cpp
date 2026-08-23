#include "lunar_pure_exploration_ros/marker_builder.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace lunar::pure_exploration_ros {
namespace {

using lunar::pure_exploration::CandidateKey;
using lunar::pure_exploration::CandidateView;
using lunar::pure_exploration::FrontierCluster;
using lunar::pure_exploration::Pose2;

FrontierCluster Frontier(std::uint64_t display_id,
                         std::vector<std::int64_t> canonical_key,
                         double x) {
  return {.id = display_id,
          .cells = {{1, 2}},
          .interface_edges = {},
          .canonical_key = std::move(canonical_key),
          .centroid = {x, 3.0},
          .length_m = 1.0};
}

CandidateView Candidate(std::uint64_t display_id, std::size_t frontier_index,
                        std::vector<std::int64_t> frontier_key, double x) {
  auto key = std::make_shared<const std::vector<std::int64_t>>(
      std::move(frontier_key));
  return {.id = display_id,
          .frontier_id = 7U,
          .frontier_index = frontier_index,
          .key = {.x_mm = static_cast<std::int64_t>(x * 1000.0),
                  .y_mm = 0,
                  .yaw_tenth_deg = 0},
          .pose = {.x = x, .y = 0.0, .yaw = 0.0},
          .frontier_distance_m = 1.0,
          .frontier_canonical_key = std::move(key)};
}

TEST(ExplorationMarkerBuilderTest,
     UsesFrozenVectorIndicesAndFullIdentityToMarkOnlyCommittedCandidate) {
  const std::vector<FrontierCluster> frontiers{
      Frontier(7U, {1, 2, 3}, 1.0), Frontier(7U, {4, 5, 6}, 2.0)};
  const std::vector<CandidateView> candidates{
      Candidate(99U, 0U, {1, 2, 3}, 4.0),
      Candidate(99U, 1U, {4, 5, 6}, 5.0)};
  const MarkerSelection selected{.candidate_key = candidates[1].key,
                                 .frontier_canonical_key = {4, 5, 6},
                                 .target = Pose2{5.0, 0.0, 0.0}};

  MarkerBuilder builder;
  const auto markers = builder.Build(frontiers, candidates, selected);

  ASSERT_EQ(markers.markers.size(), 4U);
  EXPECT_EQ(markers.markers[0].ns, "frontiers");
  EXPECT_EQ(markers.markers[0].id, 0);
  EXPECT_EQ(markers.markers[1].ns, "frontiers");
  EXPECT_EQ(markers.markers[1].id, 1);
  EXPECT_EQ(markers.markers[2].ns, "candidates");
  EXPECT_EQ(markers.markers[2].id, 0);
  EXPECT_EQ(markers.markers[2].color.g, 1.0F);
  EXPECT_EQ(markers.markers[3].ns, "candidates");
  EXPECT_EQ(markers.markers[3].id, 1);
  EXPECT_EQ(markers.markers[3].color.r, 1.0F);
}

TEST(ExplorationMarkerBuilderTest, DeletesMarkersAbsentFromReplacementBatch) {
  MarkerBuilder builder;
  const std::vector<FrontierCluster> frontiers{
      Frontier(7U, {1, 2, 3}, 1.0), Frontier(8U, {4, 5, 6}, 2.0)};
  const std::vector<CandidateView> candidates{
      Candidate(9U, 0U, {1, 2, 3}, 4.0),
      Candidate(10U, 1U, {4, 5, 6}, 5.0)};
  static_cast<void>(builder.Build(frontiers, candidates, std::nullopt));

  const auto markers = builder.Build(
      std::vector<FrontierCluster>{frontiers.front()},
      std::vector<CandidateView>{candidates.front()}, std::nullopt);

  ASSERT_EQ(markers.markers.size(), 4U);
  EXPECT_EQ(markers.markers[2].ns, "frontiers");
  EXPECT_EQ(markers.markers[2].id, 1);
  EXPECT_EQ(markers.markers[2].action,
            visualization_msgs::msg::Marker::DELETE);
  EXPECT_EQ(markers.markers[3].ns, "candidates");
  EXPECT_EQ(markers.markers[3].id, 1);
  EXPECT_EQ(markers.markers[3].action,
            visualization_msgs::msg::Marker::DELETE);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
