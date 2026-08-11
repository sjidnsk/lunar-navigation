#include "lunar_unreal_tcp_bridge/coordinate_transform.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace lunar::unreal_tcp {
namespace {

constexpr double kTolerance = 1.0e-10;
constexpr char kCalibrationHash[] = "sha256:coordinate-golden";

CoordinateTransformConfig Config() {
  CoordinateTransformConfig config;
  config.basis_map_from_unreal =
      (Eigen::Vector3d{1.0, -1.0, 1.0}).asDiagonal();
  config.unreal_world_origin_in_map_m = {10.0, -4.0, 2.0};
  config.length_unit_to_m = 0.01;
  config.calibration_hash = kCalibrationHash;
  return config;
}

void ExpectVectorNear(
    const Eigen::Vector3d& actual, const Eigen::Vector3d& expected) {
  EXPECT_NEAR(actual.x(), expected.x(), kTolerance);
  EXPECT_NEAR(actual.y(), expected.y(), kTolerance);
  EXPECT_NEAR(actual.z(), expected.z(), kTolerance);
}

void ExpectRotationNear(
    const Eigen::Matrix3d& actual, const Eigen::Matrix3d& expected) {
  EXPECT_LT((actual - expected).norm(), kTolerance);
}

TEST(CoordinateTransform, MapsOriginUnitAxesAndNegativeLandscapePosition) {
  const CoordinateTransform transform{Config(), kCalibrationHash};
  ExpectVectorNear(
      transform.UnrealPositionToMap({0.0, 0.0, 0.0}),
      {10.0, -4.0, 2.0});
  ExpectVectorNear(
      transform.UnrealPositionToMap({100.0, 0.0, 0.0}),
      {11.0, -4.0, 2.0});
  ExpectVectorNear(
      transform.UnrealPositionToMap({0.0, 100.0, 0.0}),
      {10.0, -5.0, 2.0});
  ExpectVectorNear(
      transform.UnrealPositionToMap({0.0, 0.0, 100.0}),
      {10.0, -4.0, 3.0});

  auto landscape = Config();
  landscape.unreal_world_origin_in_map_m.setZero();
  const CoordinateTransform landscape_transform{landscape, kCalibrationHash};
  ExpectVectorNear(
      landscape_transform.UnrealPositionToMap(
          {-205160.0, -511559.0, 50.0}),
      {-2051.6, 5115.59, 0.5});
}

TEST(CoordinateTransform, ChangesQuaternionBasisForYawAndRoll) {
  const CoordinateTransform transform{Config(), kCalibrationHash};
  const Eigen::Quaterniond unreal_yaw{
      Eigen::AngleAxisd{M_PI / 2.0, Eigen::Vector3d::UnitZ()}};
  const Eigen::Quaterniond map_yaw =
      transform.UnrealOrientationToMap(unreal_yaw);
  ExpectRotationNear(
      map_yaw.toRotationMatrix(),
      Eigen::AngleAxisd{-M_PI / 2.0, Eigen::Vector3d::UnitZ()}
          .toRotationMatrix());

  const Eigen::Quaterniond unreal_roll{
      Eigen::AngleAxisd{M_PI / 3.0, Eigen::Vector3d::UnitX()}};
  ExpectRotationNear(
      transform.UnrealOrientationToMap(unreal_roll).toRotationMatrix(),
      Eigen::AngleAxisd{-M_PI / 3.0, Eigen::Vector3d::UnitX()}
          .toRotationMatrix());
}

TEST(CoordinateTransform, AppliesPseudovectorRuleToAngularVelocity) {
  const CoordinateTransform transform{Config(), kCalibrationHash};
  ExpectVectorNear(
      transform.UnrealLinearVelocityToMap({100.0, 200.0, 300.0}),
      {1.0, -2.0, 3.0});
  ExpectVectorNear(
      transform.UnrealAngularVelocityToMap({1.0, 2.0, 3.0}),
      {-1.0, 2.0, -3.0});
}

TEST(CoordinateTransform, RoundTripsPoseVelocityAndRigidCalibrations) {
  auto config = Config();
  config.base_link_from_unreal_root(2, 3) = 0.5;
  config.base_footprint_from_base_link(2, 3) = 0.4;
  const CoordinateTransform transform{config, kCalibrationHash};
  const Eigen::Vector3d unreal_position{-340.0, 120.0, 80.0};
  const Eigen::Quaterniond unreal_orientation{
      Eigen::AngleAxisd{0.37, Eigen::Vector3d::UnitZ()}};

  ExpectVectorNear(
      transform.MapPositionToUnreal(
          transform.UnrealPositionToMap(unreal_position)),
      unreal_position);
  ExpectVectorNear(
      transform.MapLinearVelocityToUnreal(
          transform.UnrealLinearVelocityToMap({7.0, -8.0, 9.0})),
      {7.0, -8.0, 9.0});
  ExpectVectorNear(
      transform.MapAngularVelocityToUnreal(
          transform.UnrealAngularVelocityToMap({0.1, -0.2, 0.3})),
      {0.1, -0.2, 0.3});
  ExpectRotationNear(
      transform.MapOrientationToUnreal(
          transform.UnrealOrientationToMap(unreal_orientation))
          .toRotationMatrix(),
      unreal_orientation.toRotationMatrix());

  const Eigen::Isometry3d map_from_footprint =
      transform.MapFromBaseFootprint(unreal_position, unreal_orientation);
  const Eigen::Isometry3d unreal_from_root =
      transform.UnrealFromRootForMapBaseFootprint(map_from_footprint);
  ExpectVectorNear(unreal_from_root.translation(), unreal_position);
  ExpectRotationNear(
      unreal_from_root.rotation(), unreal_orientation.toRotationMatrix());
}

TEST(CoordinateTransform, RejectsInvalidBasisCalibrationAndHash) {
  auto singular = Config();
  singular.basis_map_from_unreal.row(2).setZero();
  EXPECT_THROW(
      CoordinateTransform(singular, kCalibrationHash), std::invalid_argument);

  auto non_rigid = Config();
  non_rigid.base_link_from_unreal_root(0, 0) = 2.0;
  EXPECT_THROW(
      CoordinateTransform(non_rigid, kCalibrationHash),
      std::invalid_argument);

  EXPECT_THROW(
      CoordinateTransform(Config(), "sha256:different"),
      std::invalid_argument);
}

}  // namespace
}  // namespace lunar::unreal_tcp
