#include "lunar_unreal_tcp_bridge/coordinate_transform.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lunar::unreal_tcp {
namespace {

constexpr double kRigidTolerance = 1.0e-9;

[[nodiscard]] bool Finite(const Eigen::MatrixXd& matrix) {
  return matrix.array().isFinite().all();
}

[[nodiscard]] bool IsOrthonormal(const Eigen::Matrix3d& matrix) {
  return Finite(matrix) &&
      (matrix.transpose() * matrix - Eigen::Matrix3d::Identity()).norm() <=
          kRigidTolerance &&
      std::abs(std::abs(matrix.determinant()) - 1.0) <= kRigidTolerance;
}

[[nodiscard]] Eigen::Isometry3d RigidTransform(
    const Eigen::Matrix4d& matrix, const char* reason_code) {
  if (!Finite(matrix) ||
      (matrix.row(3) - Eigen::RowVector4d{0.0, 0.0, 0.0, 1.0}).norm() >
          kRigidTolerance ||
      !IsOrthonormal(matrix.block<3, 3>(0, 0)) ||
      matrix.block<3, 3>(0, 0).determinant() < 0.0) {
    throw std::invalid_argument(reason_code);
  }
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.matrix() = matrix;
  return transform;
}

[[nodiscard]] Eigen::Quaterniond UnitQuaternion(
    const Eigen::Quaterniond& value, const char* reason_code) {
  if (!value.coeffs().array().isFinite().all() ||
      value.squaredNorm() <= kRigidTolerance) {
    throw std::invalid_argument(reason_code);
  }
  return value.normalized();
}

}  // namespace

CoordinateTransform::CoordinateTransform(
    CoordinateTransformConfig config,
    const std::string& declared_calibration_hash)
    : config_(std::move(config)) {
  if (!IsOrthonormal(config_.basis_map_from_unreal)) {
    throw std::invalid_argument("COORDINATE_BASIS_INVALID");
  }
  if (!config_.unreal_world_origin_in_map_m.array().isFinite().all() ||
      !std::isfinite(config_.length_unit_to_m) ||
      config_.length_unit_to_m <= 0.0) {
    throw std::invalid_argument("COORDINATE_SCALE_OR_ORIGIN_INVALID");
  }
  if (config_.calibration_hash.empty() ||
      config_.calibration_hash != declared_calibration_hash) {
    throw std::invalid_argument("CALIBRATION_HASH_MISMATCH");
  }
  inverse_basis_ = config_.basis_map_from_unreal.inverse();
  basis_determinant_ = config_.basis_map_from_unreal.determinant();
  base_link_from_unreal_root_ = RigidTransform(
      config_.base_link_from_unreal_root,
      "BASE_LINK_FROM_UNREAL_ROOT_NOT_RIGID");
  base_footprint_from_base_link_ = RigidTransform(
      config_.base_footprint_from_base_link,
      "BASE_FOOTPRINT_FROM_BASE_LINK_NOT_RIGID");
}

Eigen::Vector3d CoordinateTransform::UnrealPositionToMap(
    const Eigen::Vector3d& position_native) const {
  if (!position_native.array().isFinite().all()) {
    throw std::invalid_argument("UNREAL_POSITION_NONFINITE");
  }
  return config_.unreal_world_origin_in_map_m +
      config_.basis_map_from_unreal * position_native *
          config_.length_unit_to_m;
}

Eigen::Vector3d CoordinateTransform::MapPositionToUnreal(
    const Eigen::Vector3d& position_m) const {
  if (!position_m.array().isFinite().all()) {
    throw std::invalid_argument("MAP_POSITION_NONFINITE");
  }
  return inverse_basis_ *
      (position_m - config_.unreal_world_origin_in_map_m) /
      config_.length_unit_to_m;
}

Eigen::Vector3d CoordinateTransform::UnrealDirectionToMap(
    const Eigen::Vector3d& direction) const {
  if (!direction.array().isFinite().all()) {
    throw std::invalid_argument("UNREAL_DIRECTION_NONFINITE");
  }
  return config_.basis_map_from_unreal * direction;
}

Eigen::Vector3d CoordinateTransform::MapDirectionToUnreal(
    const Eigen::Vector3d& direction) const {
  if (!direction.array().isFinite().all()) {
    throw std::invalid_argument("MAP_DIRECTION_NONFINITE");
  }
  return inverse_basis_ * direction;
}

Eigen::Vector3d CoordinateTransform::UnrealLinearVelocityToMap(
    const Eigen::Vector3d& velocity_native_per_s) const {
  return UnrealDirectionToMap(velocity_native_per_s) *
      config_.length_unit_to_m;
}

Eigen::Vector3d CoordinateTransform::MapLinearVelocityToUnreal(
    const Eigen::Vector3d& velocity_mps) const {
  return MapDirectionToUnreal(velocity_mps) / config_.length_unit_to_m;
}

Eigen::Vector3d CoordinateTransform::UnrealAngularVelocityToMap(
    const Eigen::Vector3d& angular_velocity_radps) const {
  return basis_determinant_ *
      UnrealDirectionToMap(angular_velocity_radps);
}

Eigen::Vector3d CoordinateTransform::MapAngularVelocityToUnreal(
    const Eigen::Vector3d& angular_velocity_radps) const {
  return basis_determinant_ *
      MapDirectionToUnreal(angular_velocity_radps);
}

Eigen::Quaterniond CoordinateTransform::UnrealOrientationToMap(
    const Eigen::Quaterniond& orientation) const {
  const Eigen::Matrix3d rotation =
      config_.basis_map_from_unreal *
      UnitQuaternion(orientation, "UNREAL_ORIENTATION_INVALID")
          .toRotationMatrix() *
      inverse_basis_;
  return UnitQuaternion(
      Eigen::Quaterniond{rotation}, "MAP_ORIENTATION_INVALID");
}

Eigen::Quaterniond CoordinateTransform::MapOrientationToUnreal(
    const Eigen::Quaterniond& orientation) const {
  const Eigen::Matrix3d rotation =
      inverse_basis_ *
      UnitQuaternion(orientation, "MAP_ORIENTATION_INVALID")
          .toRotationMatrix() *
      config_.basis_map_from_unreal;
  return UnitQuaternion(
      Eigen::Quaterniond{rotation}, "UNREAL_ORIENTATION_INVALID");
}

Eigen::Isometry3d CoordinateTransform::MapFromUnrealRoot(
    const Eigen::Vector3d& position_native,
    const Eigen::Quaterniond& orientation) const {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = UnrealPositionToMap(position_native);
  result.linear() = UnrealOrientationToMap(orientation).toRotationMatrix();
  return result;
}

Eigen::Isometry3d CoordinateTransform::MapFromBaseLink(
    const Eigen::Vector3d& root_position_native,
    const Eigen::Quaterniond& root_orientation) const {
  return MapFromUnrealRoot(root_position_native, root_orientation) *
      base_link_from_unreal_root_.inverse();
}

Eigen::Isometry3d CoordinateTransform::MapFromBaseFootprint(
    const Eigen::Vector3d& root_position_native,
    const Eigen::Quaterniond& root_orientation) const {
  return MapFromBaseLink(root_position_native, root_orientation) *
      base_footprint_from_base_link_.inverse();
}

Eigen::Isometry3d CoordinateTransform::UnrealFromRootForMapBaseFootprint(
    const Eigen::Isometry3d& map_from_base_footprint) const {
  if (!Finite(map_from_base_footprint.matrix()) ||
      !IsOrthonormal(map_from_base_footprint.rotation()) ||
      map_from_base_footprint.rotation().determinant() < 0.0) {
    throw std::invalid_argument("MAP_FROM_BASE_FOOTPRINT_NOT_RIGID");
  }
  const Eigen::Isometry3d map_from_unreal_root =
      map_from_base_footprint * base_footprint_from_base_link_ *
      base_link_from_unreal_root_;
  Eigen::Isometry3d unreal_from_root = Eigen::Isometry3d::Identity();
  unreal_from_root.translation() =
      MapPositionToUnreal(map_from_unreal_root.translation());
  unreal_from_root.linear() = MapOrientationToUnreal(
      Eigen::Quaterniond{map_from_unreal_root.rotation()})
                                  .toRotationMatrix();
  return unreal_from_root;
}

const Eigen::Isometry3d& CoordinateTransform::base_link_from_unreal_root()
    const {
  return base_link_from_unreal_root_;
}

const Eigen::Isometry3d&
CoordinateTransform::base_footprint_from_base_link() const {
  return base_footprint_from_base_link_;
}

double CoordinateTransform::length_unit_to_m() const noexcept {
  return config_.length_unit_to_m;
}

const std::string& CoordinateTransform::calibration_hash() const noexcept {
  return config_.calibration_hash;
}

}  // namespace lunar::unreal_tcp
