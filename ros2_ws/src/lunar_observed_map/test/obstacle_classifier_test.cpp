#include "lunar_observed_map/obstacle_classifier.hpp"

#include <array>
#include <optional>

#include <gtest/gtest.h>

namespace lunar::observed_map {
namespace {

CellEvidence Evidence(
    const double elevation, const double variance = 0.0,
    const bool forbidden = false) {
  CellEvidence value;
  value.valid = true;
  value.elevation_m = elevation;
  value.elevation_m2 = variance;
  value.observation_count = variance == 0.0 ? 1U : 2U;
  value.forbidden = forbidden;
  return value;
}

CellNeighborhood FlatNeighborhood(const double elevation = 0.0) {
  CellNeighborhood neighborhood;
  neighborhood.fill(Evidence(elevation));
  return neighborhood;
}

TEST(ObstacleClassifier, ClassifiesUnknownFreeAndOccupiedIndependently) {
  ObstacleClassifier classifier;
  CellNeighborhood unknown;
  EXPECT_EQ(
      classifier.Classify(unknown).occupancy, Occupancy::kUnknown);

  CellNeighborhood free = FlatNeighborhood();
  EXPECT_EQ(classifier.Classify(free).occupancy, Occupancy::kFree);

  CellNeighborhood occupied = FlatNeighborhood();
  occupied[kNeighborhoodCenter] = Evidence(0.21);
  const CellClassification result = classifier.Classify(occupied);
  EXPECT_EQ(result.occupancy, Occupancy::kOccupied);
  EXPECT_TRUE(result.obstacle);
}

TEST(ObstacleClassifier, ThresholdIsFreeAtPointTwoAndOccupiedAbovePointTwo) {
  ObstacleClassifier classifier;
  CellNeighborhood at_threshold = FlatNeighborhood();
  at_threshold[kNeighborhoodCenter] = Evidence(0.20);
  EXPECT_EQ(
      classifier.Classify(at_threshold).occupancy, Occupancy::kFree);

  CellNeighborhood over_threshold = FlatNeighborhood();
  over_threshold[kNeighborhoodCenter] = Evidence(0.200001);
  EXPECT_EQ(
      classifier.Classify(over_threshold).occupancy,
      Occupancy::kOccupied);
}

TEST(ObstacleClassifier, DetectsNegativeStepAsObstacleWithConservativeHeight) {
  ObstacleClassifier classifier;
  CellNeighborhood trench = FlatNeighborhood();
  trench[kNeighborhoodCenter] = Evidence(-0.21);

  const CellClassification result = classifier.Classify(trench);

  EXPECT_EQ(result.occupancy, Occupancy::kOccupied);
  EXPECT_TRUE(result.obstacle);
  EXPECT_GE(result.obstacle_height_m, kL0ResolutionM);
}

TEST(ObstacleClassifier, UsesObservedNeighborsOnlyForSupport) {
  ObstacleClassifier classifier;
  CellNeighborhood sparse;
  sparse[kNeighborhoodCenter] = Evidence(1.0);
  sparse[0] = Evidence(1.0, 0.5);
  sparse[1] = std::nullopt;

  const CellClassification result = classifier.Classify(sparse);

  EXPECT_EQ(result.occupancy, Occupancy::kFree);
  EXPECT_DOUBLE_EQ(result.support_elevation_m, 1.0);
  EXPECT_DOUBLE_EQ(result.obstacle_variance, 0.0);
}

TEST(ObstacleClassifier, ForbiddenRemainsIndependentOfUnknownAndObstacle) {
  ObstacleClassifier classifier;
  CellNeighborhood unknown_forbidden;
  CellEvidence marker;
  marker.forbidden = true;
  unknown_forbidden[kNeighborhoodCenter] = marker;
  const CellClassification unknown = classifier.Classify(unknown_forbidden);
  EXPECT_EQ(unknown.occupancy, Occupancy::kUnknown);
  EXPECT_TRUE(unknown.forbidden);
  EXPECT_FALSE(unknown.obstacle);

  CellNeighborhood occupied_forbidden = FlatNeighborhood();
  occupied_forbidden[kNeighborhoodCenter] = Evidence(1.0, 0.0, true);
  const CellClassification occupied =
      classifier.Classify(occupied_forbidden);
  EXPECT_EQ(occupied.occupancy, Occupancy::kOccupied);
  EXPECT_TRUE(occupied.forbidden);
  EXPECT_TRUE(occupied.obstacle);
}

}  // namespace
}  // namespace lunar::observed_map
