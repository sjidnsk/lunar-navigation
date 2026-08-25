#include "lunar_pure_exploration_ros/marker_builder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace lunar::pure_exploration_ros {
namespace {

using lunar::pure_exploration::CandidateKey;
using lunar::pure_exploration::CandidateView;
using lunar::pure_exploration::ApproachCandidateKind;
using lunar::pure_exploration::BoundaryApproachGoalIdentity;
using lunar::pure_exploration::FrontierCluster;
using lunar::pure_exploration::Pose2;
using lunar::pure_exploration::Vec2;

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

BoundaryApproachGoalIdentity ApproachIdentity(
    const std::int32_t intent_x, const std::int64_t candidate_x,
    const ApproachCandidateKind kind = ApproachCandidateKind::kTranslation) {
  return {.intent_cell = {.x = intent_x, .y = 4},
          .candidate_key = {.x_mm = candidate_x,
                            .y_mm = 2500,
                            .yaw_tenth_deg = 0},
          .candidate_kind = kind};
}

ApproachMarkerCandidate ApproachCandidateMarker(
    BoundaryApproachGoalIdentity identity, const double x,
    const double yaw = 0.0) {
  return {.identity = std::move(identity),
          .pose = {.x = x, .y = 2.5, .yaw = yaw}};
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

TEST(ExplorationMarkerBuilderTest,
     BuildsDistinctApproachNamespacesAndSelectsByFullIdentity) {
  const auto first_identity = ApproachIdentity(8, 2500);
  const auto selected_identity = ApproachIdentity(9, 2500);
  const ApproachMarkers approach{
      .intent_points = {Vec2{4.25, 2.25}, Vec2{4.75, 2.25}},
      .selected_guidance = {Vec2{2.5, 2.5}, Vec2{4.75, 2.25}},
      .candidates = {ApproachCandidateMarker(first_identity, 2.5),
                     ApproachCandidateMarker(selected_identity, 2.5)},
      .selected_identity = selected_identity};

  MarkerBuilder builder;
  const auto markers = builder.Build({}, {}, std::nullopt, &approach);

  ASSERT_EQ(markers.markers.size(), 5U);
  EXPECT_EQ(markers.markers[0].ns, "approach_intents");
  EXPECT_EQ(markers.markers[0].id, 0);
  EXPECT_EQ(markers.markers[1].ns, "approach_intents");
  EXPECT_EQ(markers.markers[1].id, 1);
  EXPECT_EQ(markers.markers[2].ns, "approach_guidance");
  EXPECT_EQ(markers.markers[2].id, 0);
  EXPECT_EQ(markers.markers[2].type,
            visualization_msgs::msg::Marker::LINE_STRIP);
  EXPECT_EQ(markers.markers[3].ns, "approach_candidates");
  EXPECT_EQ(markers.markers[3].color.g, 1.0F);
  EXPECT_EQ(markers.markers[4].ns, "approach_candidates");
  EXPECT_EQ(markers.markers[4].color.r, 1.0F);
}

TEST(ExplorationMarkerBuilderTest,
     DeletesShortenedApproachDataAndClearsBothPhaseDirections) {
  MarkerBuilder builder;
  const ApproachMarkers initial{
      .intent_points = {Vec2{4.25, 2.25}, Vec2{4.75, 2.25},
                        Vec2{5.25, 2.25}},
      .selected_guidance = {Vec2{2.5, 2.5}, Vec2{4.75, 2.25}},
      .candidates = {
          ApproachCandidateMarker(ApproachIdentity(8, 2500), 2.5),
          ApproachCandidateMarker(ApproachIdentity(9, 3000), 3.0)},
      .selected_identity = ApproachIdentity(9, 3000)};
  static_cast<void>(builder.Build({}, {}, std::nullopt, &initial));

  const ApproachMarkers shortened{
      .intent_points = {Vec2{4.25, 2.25}},
      .selected_guidance = {},
      .candidates = {
          ApproachCandidateMarker(ApproachIdentity(8, 2500), 2.5)},
      .selected_identity = std::nullopt};
  const auto shortened_markers =
      builder.Build({}, {}, std::nullopt, &shortened);

  const auto has_delete = [](const auto& markers, const std::string& ns,
                             const int id) {
    return std::ranges::any_of(markers.markers, [&](const auto& marker) {
      return marker.ns == ns && marker.id == id &&
             marker.action == visualization_msgs::msg::Marker::DELETE;
    });
  };
  EXPECT_TRUE(has_delete(shortened_markers, "approach_intents", 1));
  EXPECT_TRUE(has_delete(shortened_markers, "approach_intents", 2));
  EXPECT_TRUE(has_delete(shortened_markers, "approach_guidance", 0));
  EXPECT_TRUE(has_delete(shortened_markers, "approach_candidates", 1));

  const std::vector<FrontierCluster> frontiers{
      Frontier(7U, {1, 2, 3}, 1.0)};
  const std::vector<CandidateView> candidates{
      Candidate(9U, 0U, {1, 2, 3}, 4.0)};
  const auto explore_markers =
      builder.Build(frontiers, candidates, std::nullopt, nullptr);
  EXPECT_TRUE(has_delete(explore_markers, "approach_intents", 0));
  EXPECT_TRUE(has_delete(explore_markers, "approach_candidates", 0));
  EXPECT_TRUE(std::ranges::any_of(explore_markers.markers,
                                 [](const auto& marker) {
                                   return marker.ns == "frontiers" &&
                                          marker.action ==
                                              visualization_msgs::msg::Marker::ADD;
                                 }));

  const auto approach_markers =
      builder.Build({}, {}, std::nullopt, &shortened);
  EXPECT_TRUE(has_delete(approach_markers, "frontiers", 0));
  EXPECT_TRUE(has_delete(approach_markers, "candidates", 0));
  EXPECT_TRUE(std::ranges::any_of(approach_markers.markers,
                                 [](const auto& marker) {
                                   return marker.ns == "approach_candidates" &&
                                          marker.action ==
                                              visualization_msgs::msg::Marker::ADD;
                                 }));
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
