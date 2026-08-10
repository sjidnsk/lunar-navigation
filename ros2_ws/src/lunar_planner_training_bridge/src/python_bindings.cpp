#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <urdf/model.h>
#include <urdf_model/link.h>

#include "lunar_planner_training_bridge/request.hpp"
#include "lunar_planner_training_bridge/visibility.hpp"

namespace py = pybind11;
namespace planning = lunar::planning;
namespace training = lunar::planning::training;

namespace {

void AddUrdfMeshFilename(
    const urdf::GeometrySharedPtr &geometry,
    std::set<std::string, std::less<>> &filenames) {
  if (geometry == nullptr || geometry->type != urdf::Geometry::MESH) {
    return;
  }
  const auto mesh = std::static_pointer_cast<urdf::Mesh>(geometry);
  if (mesh->filename.empty() || !std::isfinite(mesh->scale.x) ||
      !std::isfinite(mesh->scale.y) || !std::isfinite(mesh->scale.z) ||
      mesh->scale.x <= 0.0 || mesh->scale.y <= 0.0 || mesh->scale.z <= 0.0) {
    throw py::value_error("URDF mesh filename or scale is invalid");
  }
  filenames.insert(mesh->filename);
}

[[nodiscard]] py::tuple ValidateUrdfGeometry(
    const std::string &document,
    const std::string &base_frame_id) {
  urdf::Model model;
  if (!model.initString(document) || model.getLink(base_frame_id) == nullptr) {
    throw py::value_error("URDF model or base frame is invalid");
  }
  std::vector<urdf::LinkSharedPtr> links;
  model.getLinks(links);
  std::set<std::string, std::less<>> filenames;
  for (const auto &link : links) {
    for (const auto &visual : link->visual_array) {
      if (visual != nullptr) {
        AddUrdfMeshFilename(visual->geometry, filenames);
      }
    }
    for (const auto &collision : link->collision_array) {
      if (collision != nullptr) {
        AddUrdfMeshFilename(collision->geometry, filenames);
      }
    }
  }
  if (filenames.empty()) {
    throw py::value_error("URDF requires at least one mesh");
  }
  py::tuple result(filenames.size());
  std::size_t index = 0U;
  for (const auto &filename : filenames) {
    result[index++] = py::str(filename);
  }
  return result;
}

[[nodiscard]] planning::GridLayer GridLayerFromArray(const py::array &values) {
  if ((values.flags() & py::array::c_style) == 0) {
    throw py::value_error("grid layer array must be C-contiguous");
  }
  if (values.ndim() != 1) {
    throw py::value_error("grid layer array must be one-dimensional");
  }
  const py::ssize_t size = values.size();
  if (values.dtype().is(py::dtype::of<float>())) {
    const auto *begin = static_cast<const float *>(values.data());
    return planning::GridLayer{
        .values = std::vector<float>(begin, begin + size),
    };
  }
  if (values.dtype().is(py::dtype::of<std::uint8_t>())) {
    const auto *begin = static_cast<const std::uint8_t *>(values.data());
    return planning::GridLayer{
        .values = std::vector<std::uint8_t>(begin, begin + size),
    };
  }
  if (values.dtype().is(py::dtype::of<std::uint32_t>())) {
    const auto *begin = static_cast<const std::uint32_t *>(values.data());
    return planning::GridLayer{
        .values = std::vector<std::uint32_t>(begin, begin + size),
    };
  }
  throw py::type_error("grid layer dtype must be float32, uint8, or uint32");
}

void RequireExactArray(const py::array &values, const py::dtype &dtype,
                       const py::ssize_t dimensions,
                       const char *const name,
                       const char *const dtype_name) {
  if ((values.flags() & py::array::c_style) == 0) {
    throw py::value_error(std::string{name} + " must be C-contiguous");
  }
  if (values.ndim() != dimensions) {
    throw py::value_error(std::string{name} + " has invalid dimensions");
  }
  if (!values.dtype().is(dtype)) {
    throw py::type_error(std::string{name} + " dtype must be " + dtype_name);
  }
}

void RequireSameGridShape(const py::array &reference,
                          const py::array &values,
                          const char *const name) {
  if (reference.shape(0) != values.shape(0) ||
      reference.shape(1) != values.shape(1)) {
    throw py::value_error(std::string{name} + " shape mismatch");
  }
}

void RequireFiniteFloatArray(const py::array &values,
                             const char *const name) {
  const auto *data = static_cast<const float *>(values.data());
  if (!std::all_of(data, data + values.size(),
                   [](const float value) { return std::isfinite(value); })) {
    throw py::value_error(std::string{name} + " must be finite");
  }
}

void RequireFiniteDoubleArray(const py::array &values,
                              const char *const name) {
  const auto *data = static_cast<const double *>(values.data());
  if (!std::all_of(data, data + values.size(),
                   [](const double value) { return std::isfinite(value); })) {
    throw py::value_error(std::string{name} + " must be finite");
  }
}

template <typename Value>
[[nodiscard]] py::array_t<Value> CopyArray(const std::vector<Value> &values) {
  py::array_t<Value> result(py::array::ShapeContainer{
      static_cast<py::ssize_t>(values.size()),
  });
  std::copy(values.begin(), values.end(), result.mutable_data());
  return result;
}

template <typename Value>
[[nodiscard]] py::array_t<Value> CopyArray2d(
    const std::vector<Value> &values,
    const std::size_t height,
    const std::size_t width) {
  py::array_t<Value> result(py::array::ShapeContainer{
      static_cast<py::ssize_t>(height),
      static_cast<py::ssize_t>(width),
  });
  std::copy(values.begin(), values.end(), result.mutable_data());
  return result;
}

template <typename Value>
[[nodiscard]] py::array_t<Value> ReadonlyArray(py::array_t<Value> result) {
  result.attr("setflags")(false);
  return result;
}

[[nodiscard]] py::object GridLayerValues(const planning::GridLayer &layer) {
  return std::visit(
      [](const auto &values) -> py::object { return CopyArray(values); },
      layer.values);
}

[[nodiscard]] std::string PlatformTypeName(
    const planning::PlatformType platform_type) {
  switch (platform_type) {
    case planning::PlatformType::kWheeled:
      return "WHEELED";
    case planning::PlatformType::kLegged:
      return "LEGGED";
    case planning::PlatformType::kHopper:
      return "HOPPER";
  }
  throw py::value_error("unknown platform type");
}

[[nodiscard]] planning::PlatformType PlatformTypeFromPython(
    const py::object &value) {
  if (py::isinstance<py::str>(value)) {
    const std::string name = value.cast<std::string>();
    if (name == "WHEELED") {
      return planning::PlatformType::kWheeled;
    }
    if (name == "LEGGED") {
      return planning::PlatformType::kLegged;
    }
    if (name == "HOPPER") {
      return planning::PlatformType::kHopper;
    }
    throw py::value_error("platform type must be WHEELED, LEGGED, or HOPPER");
  }
  return value.cast<planning::PlatformType>();
}

void BindGeometry(py::module_ &module) {
  py::class_<planning::TimePoint>(module, "TimePoint")
      .def(py::init<>())
      .def_readwrite("nanoseconds_since_epoch",
                     &planning::TimePoint::nanoseconds_since_epoch);
  py::class_<planning::Vec2>(module, "Vec2")
      .def(py::init<>())
      .def_readwrite("x", &planning::Vec2::x)
      .def_readwrite("y", &planning::Vec2::y);
  py::class_<planning::Vec3>(module, "Vec3")
      .def(py::init<>())
      .def_readwrite("x", &planning::Vec3::x)
      .def_readwrite("y", &planning::Vec3::y)
      .def_readwrite("z", &planning::Vec3::z);
  py::class_<planning::Quaternion>(module, "Quaternion")
      .def(py::init<>())
      .def_readwrite("w", &planning::Quaternion::w)
      .def_readwrite("x", &planning::Quaternion::x)
      .def_readwrite("y", &planning::Quaternion::y)
      .def_readwrite("z", &planning::Quaternion::z);
  py::class_<planning::Pose3>(module, "Pose3")
      .def(py::init<>())
      .def_readwrite("position_m", &planning::Pose3::position_m)
      .def_readwrite("orientation", &planning::Pose3::orientation);
  py::class_<planning::Twist3>(module, "Twist3")
      .def(py::init<>())
      .def_readwrite("linear_mps", &planning::Twist3::linear_mps)
      .def_readwrite("angular_radps", &planning::Twist3::angular_radps);
  py::class_<planning::Interval>(module, "Interval")
      .def(py::init<>())
      .def_readwrite("lower", &planning::Interval::lower)
      .def_readwrite("upper", &planning::Interval::upper);
  py::class_<planning::Polygon2>(module, "Polygon2")
      .def(py::init<>())
      .def_readwrite("vertices", &planning::Polygon2::vertices);
}

void BindWorld(py::module_ &module) {
  py::class_<planning::GridLayer>(module, "GridLayer")
      .def(py::init<>())
      .def(py::init(&GridLayerFromArray), py::arg("values"))
      .def_property("values", &GridLayerValues,
                    [](planning::GridLayer &self, const py::array &values) {
                      self.values = GridLayerFromArray(values).values;
                    })
      .def_property_readonly("size", &planning::GridLayer::size);
  py::class_<planning::GridMap>(module, "GridMap")
      .def(py::init<>())
      .def_readwrite("frame_id", &planning::GridMap::frame_id)
      .def_readwrite("stamp", &planning::GridMap::stamp)
      .def_readwrite("width", &planning::GridMap::width)
      .def_readwrite("height", &planning::GridMap::height)
      .def_readwrite("resolution_m", &planning::GridMap::resolution_m)
      .def_readwrite("origin_m", &planning::GridMap::origin_m)
      .def_readwrite("layers", &planning::GridMap::layers)
      .def_property_readonly("cell_count", &planning::GridMap::CellCount);
  py::class_<planning::RigidTransform>(module, "RigidTransform")
      .def(py::init<>())
      .def_readwrite("parent_frame", &planning::RigidTransform::parent_frame)
      .def_readwrite("child_frame", &planning::RigidTransform::child_frame)
      .def_readwrite("stamp", &planning::RigidTransform::stamp)
      .def_readwrite("translation_m", &planning::RigidTransform::translation_m)
      .def_readwrite("rotation", &planning::RigidTransform::rotation);
  py::class_<planning::WorldSnapshot>(module, "WorldSnapshot")
      .def(py::init<>())
      .def_readwrite("global_map", &planning::WorldSnapshot::global_map)
      .def_readwrite("local_map", &planning::WorldSnapshot::local_map)
      .def_readwrite("map_from_odom", &planning::WorldSnapshot::map_from_odom);
}

void BindGoalsAndState(py::module_ &module) {
  py::class_<planning::PointGoal>(module, "PointGoal")
      .def(py::init<>())
      .def_readwrite("position_m", &planning::PointGoal::position_m)
      .def_readwrite("tolerance_m", &planning::PointGoal::tolerance_m);
  py::class_<planning::PlanarRegionGoal>(module, "PlanarRegionGoal")
      .def(py::init<>())
      .def_readwrite("boundary_m", &planning::PlanarRegionGoal::boundary_m)
      .def_readwrite("normal_tolerance_m",
                     &planning::PlanarRegionGoal::normal_tolerance_m);
  py::class_<planning::GoalRegion>(module, "GoalRegion")
      .def(py::init<>())
      .def_readwrite("goal_id", &planning::GoalRegion::goal_id)
      .def_readwrite("target", &planning::GoalRegion::target)
      .def_readwrite("yaw_rad", &planning::GoalRegion::yaw_rad)
      .def_readwrite("yaw_tolerance_rad",
                     &planning::GoalRegion::yaw_tolerance_rad);

  py::class_<planning::WheeledState>(module, "WheeledState")
      .def(py::init<>())
      .def_readwrite("pose", &planning::WheeledState::pose)
      .def_readwrite("velocity", &planning::WheeledState::velocity);
  py::class_<planning::LeggedState>(module, "LeggedState")
      .def(py::init<>())
      .def_readwrite("body_pose", &planning::LeggedState::body_pose)
      .def_readwrite("body_velocity", &planning::LeggedState::body_velocity);
  py::class_<planning::HopperState>(module, "HopperState")
      .def(py::init<>())
      .def_readwrite("pose", &planning::HopperState::pose)
      .def_readwrite("velocity", &planning::HopperState::velocity);
}

void BindCapabilities(py::module_ &module) {
  py::enum_<planning::PlatformType>(module, "PlatformType")
      .value("WHEELED", planning::PlatformType::kWheeled)
      .value("LEGGED", planning::PlatformType::kLegged)
      .value("HOPPER", planning::PlatformType::kHopper);
  py::enum_<planning::WheelPrimitiveKind>(module, "WheelPrimitiveKind")
      .value("FORWARD", planning::WheelPrimitiveKind::kForward)
      .value("REVERSE", planning::WheelPrimitiveKind::kReverse)
      .value("FORWARD_ARC", planning::WheelPrimitiveKind::kForwardArc)
      .value("REVERSE_ARC", planning::WheelPrimitiveKind::kReverseArc)
      .value("SPIN_CLOCKWISE", planning::WheelPrimitiveKind::kSpinClockwise)
      .value("SPIN_COUNTERCLOCKWISE",
             planning::WheelPrimitiveKind::kSpinCounterclockwise)
      .value("STOP_AND_SWITCH", planning::WheelPrimitiveKind::kStopAndSwitch);
  py::class_<planning::WheelMotionPrimitive>(module, "WheelMotionPrimitive")
      .def(py::init<>())
      .def_readwrite("primitive_id",
                     &planning::WheelMotionPrimitive::primitive_id)
      .def_readwrite("kind", &planning::WheelMotionPrimitive::kind)
      .def_readwrite("relative_end_pose",
                     &planning::WheelMotionPrimitive::relative_end_pose);
  py::class_<planning::WheeledCapability>(module, "WheeledCapability")
      .def(py::init<>())
      .def_readwrite("footprint_xy_m",
                     &planning::WheeledCapability::footprint_xy_m)
      .def_readwrite("body_extent_m",
                     &planning::WheeledCapability::body_extent_m)
      .def_readwrite("wheel_diameter_m",
                     &planning::WheeledCapability::wheel_diameter_m)
      .def_readwrite("wheel_width_m",
                     &planning::WheeledCapability::wheel_width_m)
      .def_readwrite("wheelbase_m",
                     &planning::WheeledCapability::wheelbase_m)
      .def_readwrite("track_width_m",
                     &planning::WheeledCapability::track_width_m)
      .def_readwrite(
          "minimum_underbody_clearance_m",
          &planning::WheeledCapability::minimum_underbody_clearance_m)
      .def_readwrite(
          "maximum_local_obstacle_relief_m",
          &planning::WheeledCapability::maximum_local_obstacle_relief_m)
      .def_readwrite("allow_unsupported_gap",
                     &planning::WheeledCapability::allow_unsupported_gap)
      .def_readwrite("minimum_body_z_m",
                     &planning::WheeledCapability::minimum_body_z_m)
      .def_readwrite("maximum_body_z_m",
                     &planning::WheeledCapability::maximum_body_z_m)
      .def_readwrite("maximum_forward_speed_mps",
                     &planning::WheeledCapability::maximum_forward_speed_mps)
      .def_readwrite("maximum_reverse_speed_mps",
                     &planning::WheeledCapability::maximum_reverse_speed_mps)
      .def_readwrite("maximum_spin_rate_radps",
                     &planning::WheeledCapability::maximum_spin_rate_radps)
      .def_readwrite("maximum_acceleration_mps2",
                     &planning::WheeledCapability::maximum_acceleration_mps2)
      .def_readwrite(
          "maximum_braking_deceleration_mps2",
          &planning::WheeledCapability::maximum_braking_deceleration_mps2)
      .def_readwrite(
          "maximum_yaw_acceleration_radps2",
          &planning::WheeledCapability::maximum_yaw_acceleration_radps2)
      .def_readwrite(
          "maximum_lateral_acceleration_mps2",
          &planning::WheeledCapability::maximum_lateral_acceleration_mps2)
      .def_readwrite("maximum_curvature_per_m",
                     &planning::WheeledCapability::maximum_curvature_per_m)
      .def_readwrite("maximum_slope_rad",
                     &planning::WheeledCapability::maximum_slope_rad)
      .def_readwrite("minimum_clearance_m",
                     &planning::WheeledCapability::minimum_clearance_m)
      .def_readwrite("motion_primitives",
                     &planning::WheeledCapability::motion_primitives);

  py::enum_<planning::LeggedPrimitiveKind>(module, "LeggedPrimitiveKind")
      .value("FORWARD", planning::LeggedPrimitiveKind::kForward)
      .value("BACKWARD", planning::LeggedPrimitiveKind::kBackward)
      .value("LATERAL_LEFT", planning::LeggedPrimitiveKind::kLateralLeft)
      .value("LATERAL_RIGHT", planning::LeggedPrimitiveKind::kLateralRight)
      .value("SPIN", planning::LeggedPrimitiveKind::kSpin)
      .value("COUPLED", planning::LeggedPrimitiveKind::kCoupled);
  py::class_<planning::LeggedBodyPrimitive>(module, "LeggedBodyPrimitive")
      .def(py::init<>())
      .def_readwrite("primitive_id",
                     &planning::LeggedBodyPrimitive::primitive_id)
      .def_readwrite("kind", &planning::LeggedBodyPrimitive::kind)
      .def_readwrite("body_frame_displacement_m",
                     &planning::LeggedBodyPrimitive::body_frame_displacement_m)
      .def_readwrite("yaw_change_rad",
                     &planning::LeggedBodyPrimitive::yaw_change_rad);
  py::class_<planning::LeggedCapability>(module, "LeggedCapability")
      .def(py::init<>())
      .def_readwrite("body_extent_m",
                     &planning::LeggedCapability::body_extent_m)
      .def_readwrite("platform_mass_kg",
                     &planning::LeggedCapability::platform_mass_kg)
      .def_readwrite("maximum_payload_kg",
                     &planning::LeggedCapability::maximum_payload_kg)
      .def_readwrite("maximum_slope_rad",
                     &planning::LeggedCapability::maximum_slope_rad)
      .def_readwrite("maximum_step_height_m",
                     &planning::LeggedCapability::maximum_step_height_m)
      .def_readwrite("maximum_gap_width_m",
                     &planning::LeggedCapability::maximum_gap_width_m)
      .def_readwrite("minimum_body_clearance_m",
                     &planning::LeggedCapability::minimum_body_clearance_m)
      .def_readwrite("step_vertical_rate_mps",
                     &planning::LeggedCapability::step_vertical_rate_mps)
      .def_readwrite("body_height_m",
                     &planning::LeggedCapability::body_height_m)
      .def_readwrite("forward_speed_mps",
                     &planning::LeggedCapability::forward_speed_mps)
      .def_readwrite("lateral_speed_mps",
                     &planning::LeggedCapability::lateral_speed_mps)
      .def_readwrite("yaw_rate_radps",
                     &planning::LeggedCapability::yaw_rate_radps)
      .def_readwrite(
          "maximum_linear_acceleration_mps2",
          &planning::LeggedCapability::maximum_linear_acceleration_mps2)
      .def_readwrite(
          "maximum_yaw_acceleration_radps2",
          &planning::LeggedCapability::maximum_yaw_acceleration_radps2)
      .def_readwrite("motion_primitives",
                     &planning::LeggedCapability::motion_primitives);

  py::class_<planning::HopperCapability>(module, "HopperCapability")
      .def(py::init<>())
      .def_readwrite("specific_impulse_s",
                     &planning::HopperCapability::specific_impulse_s)
      .def_readwrite("reference_total_mass_kg",
                     &planning::HopperCapability::reference_total_mass_kg)
      .def_readwrite(
          "reference_propellant_mass_kg",
          &planning::HopperCapability::reference_propellant_mass_kg)
      .def_readwrite("landing_support_radius_m",
                     &planning::HopperCapability::landing_support_radius_m)
      .def_readwrite("flight_collision_radius_m",
                     &planning::HopperCapability::flight_collision_radius_m)
      .def_readwrite(
          "maximum_landing_plane_residual_m",
          &planning::HopperCapability::maximum_landing_plane_residual_m)
      .def_readwrite("landing_lateral_margin_m",
                     &planning::HopperCapability::landing_lateral_margin_m)
      .def_readwrite("flight_map_margin_m",
                     &planning::HopperCapability::flight_map_margin_m)
      .def_readwrite(
          "reachability_delta_v_margin_ratio",
          &planning::HopperCapability::reachability_delta_v_margin_ratio)
      .def_readwrite("standard_gravity_mps2",
                     &planning::HopperCapability::standard_gravity_mps2)
      .def_readwrite("maximum_landing_slope_rad",
                     &planning::HopperCapability::maximum_landing_slope_rad);
}

void BindConfig(py::module_ &module) {
  py::class_<planning::AraStarConfig>(module, "AraStarConfig")
      .def(py::init<>())
      .def_readwrite("initial_epsilon",
                     &planning::AraStarConfig::initial_epsilon)
      .def_readwrite("epsilon_decrement",
                     &planning::AraStarConfig::epsilon_decrement)
      .def_readwrite("target_epsilon", &planning::AraStarConfig::target_epsilon);
  py::class_<planning::CorridorConfig>(module, "CorridorConfig")
      .def(py::init<>())
      .def_readwrite("maximum_regions",
                     &planning::CorridorConfig::maximum_regions)
      .def_readwrite("maximum_inflation_iterations",
                     &planning::CorridorConfig::maximum_inflation_iterations)
      .def_readwrite("maximum_halfplanes_per_region",
                     &planning::CorridorConfig::maximum_halfplanes_per_region)
      .def_readwrite("maximum_split_depth",
                     &planning::CorridorConfig::maximum_split_depth)
      .def_readwrite("minimum_overlap_m",
                     &planning::CorridorConfig::minimum_overlap_m)
      .def_readwrite("sampling_spacing_m",
                     &planning::CorridorConfig::sampling_spacing_m);
  py::class_<planning::OptimizationConfig>(module, "OptimizationConfig")
      .def(py::init<>())
      .def_readwrite(
          "maximum_smoothing_control_points",
          &planning::OptimizationConfig::maximum_smoothing_control_points)
      .def_readwrite("maximum_smoothing_samples",
                     &planning::OptimizationConfig::maximum_smoothing_samples)
      .def_readwrite("maximum_iterations",
                     &planning::OptimizationConfig::maximum_iterations)
      .def_readwrite(
          "maximum_trust_region_reductions",
          &planning::OptimizationConfig::maximum_trust_region_reductions)
      .def_readwrite("initial_trust_region_m",
                     &planning::OptimizationConfig::initial_trust_region_m)
      .def_readwrite("minimum_trust_region_m",
                     &planning::OptimizationConfig::minimum_trust_region_m)
      .def_readwrite("constraint_tolerance",
                     &planning::OptimizationConfig::constraint_tolerance)
      .def_readwrite("require_smoothed_execution",
                     &planning::OptimizationConfig::require_smoothed_execution);
  py::class_<planning::MapSafetyConfig>(module, "MapSafetyConfig")
      .def(py::init<>())
      .def_readwrite("project_maximum_slope_rad",
                     &planning::MapSafetyConfig::project_maximum_slope_rad)
      .def_readwrite("maximum_elevation_variance_m2",
                     &planning::MapSafetyConfig::maximum_elevation_variance_m2)
      .def_readwrite("maximum_obstacle_variance_m2",
                     &planning::MapSafetyConfig::maximum_obstacle_variance_m2)
      .def_readwrite("maximum_observation_age_s",
                     &planning::MapSafetyConfig::maximum_observation_age_s)
      .def_readwrite("minimum_observation_quality",
                     &planning::MapSafetyConfig::minimum_observation_quality)
      .def_readwrite("minimum_observation_count",
                     &planning::MapSafetyConfig::minimum_observation_count);
  py::class_<planning::WheelPlannerConfig>(module, "WheelPlannerConfig")
      .def(py::init<>())
      .def_readwrite("xy_resolution_m",
                     &planning::WheelPlannerConfig::xy_resolution_m)
      .def_readwrite("yaw_bin_count",
                     &planning::WheelPlannerConfig::yaw_bin_count);
  py::class_<planning::LeggedPlannerConfig>(module, "LeggedPlannerConfig")
      .def(py::init<>())
      .def_readwrite("xy_resolution_m",
                     &planning::LeggedPlannerConfig::xy_resolution_m)
      .def_readwrite("yaw_bin_count",
                     &planning::LeggedPlannerConfig::yaw_bin_count);
  py::class_<planning::HopperPlannerConfig>(module, "HopperPlannerConfig")
      .def(py::init<>());
  py::class_<planning::GlobalMapConfig>(module, "GlobalMapConfig")
      .def(py::init<>())
      .def_readwrite("base_resolution_m",
                     &planning::GlobalMapConfig::base_resolution_m)
      .def_readwrite("maximum_level",
                     &planning::GlobalMapConfig::maximum_level)
      .def_readwrite("maximum_cells",
                     &planning::GlobalMapConfig::maximum_cells)
      .def_readwrite("maximum_axis_cells",
                     &planning::GlobalMapConfig::maximum_axis_cells)
      .def_readwrite("target_axis_cells",
                     &planning::GlobalMapConfig::target_axis_cells);
  py::class_<planning::GlobalSearchConfig>(module, "GlobalSearchConfig")
      .def(py::init<>())
      .def_readwrite("maximum_preview_points",
                     &planning::GlobalSearchConfig::maximum_preview_points)
      .def_readwrite("slope_weight",
                     &planning::GlobalSearchConfig::slope_weight)
      .def_readwrite("roughness_weight",
                     &planning::GlobalSearchConfig::roughness_weight)
      .def_readwrite("clearance_weight",
                     &planning::GlobalSearchConfig::clearance_weight);
  py::class_<planning::LocalFrontierConfig>(module, "LocalFrontierConfig")
      .def(py::init<>())
      .def_readwrite("wheel_horizon_m",
                     &planning::LocalFrontierConfig::wheel_horizon_m)
      .def_readwrite("legged_horizon_m",
                     &planning::LocalFrontierConfig::legged_horizon_m)
      .def_readwrite(
          "additional_corridor_margin_m",
          &planning::LocalFrontierConfig::additional_corridor_margin_m);
  py::class_<planning::PlannerConfig>(module, "PlannerConfig")
      .def(py::init<>())
      .def_readwrite("maximum_input_skew",
                     &planning::PlannerConfig::maximum_input_skew)
      .def_readwrite("search", &planning::PlannerConfig::search)
      .def_readwrite("corridor", &planning::PlannerConfig::corridor)
      .def_readwrite("optimization", &planning::PlannerConfig::optimization)
      .def_readwrite("map_safety", &planning::PlannerConfig::map_safety)
      .def_readwrite("wheel", &planning::PlannerConfig::wheel)
      .def_readwrite("legged", &planning::PlannerConfig::legged)
      .def_readwrite("hopper", &planning::PlannerConfig::hopper)
      .def_readwrite("global_map", &planning::PlannerConfig::global_map)
      .def_readwrite("global_search", &planning::PlannerConfig::global_search)
      .def_readwrite("local_frontier", &planning::PlannerConfig::local_frontier)
      .def_readwrite("stable_candidate_order",
                     &planning::PlannerConfig::stable_candidate_order);
}

void BindExecution(py::module_ &module) {
  py::enum_<planning::GroundExecutionState>(module, "GroundExecutionState")
      .value("IDLE", planning::GroundExecutionState::kIdle)
      .value("EXECUTING", planning::GroundExecutionState::kExecuting)
      .value("HOLDING", planning::GroundExecutionState::kHolding)
      .value("FAULT", planning::GroundExecutionState::kFault);
  py::class_<planning::GroundExecutionContext>(module, "GroundExecutionContext")
      .def(py::init<>())
      .def_readwrite("state", &planning::GroundExecutionContext::state)
      .def_readwrite("active_plan_id",
                     &planning::GroundExecutionContext::active_plan_id)
      .def_readwrite("active_segment_id",
                     &planning::GroundExecutionContext::active_segment_id);
  py::enum_<planning::HopperExecutionState>(module, "HopperExecutionState")
      .value("GROUND_HOLD", planning::HopperExecutionState::kGroundHold)
      .value("JUMP_READY", planning::HopperExecutionState::kJumpReady)
      .value("JUMP_COMMITTED", planning::HopperExecutionState::kJumpCommitted)
      .value("IN_FLIGHT", planning::HopperExecutionState::kInFlight)
      .value("LANDED_HOLD", planning::HopperExecutionState::kLandedHold)
      .value("EMERGENCY_DELEGATED",
             planning::HopperExecutionState::kEmergencyDelegated);
  py::class_<planning::HopperExecutionContext>(module, "HopperExecutionContext")
      .def(py::init<>())
      .def_readwrite("state", &planning::HopperExecutionContext::state)
      .def_readwrite("active_plan_id",
                     &planning::HopperExecutionContext::active_plan_id)
      .def_readwrite("active_segment_id",
                     &planning::HopperExecutionContext::active_segment_id);
}

void BindOutput(py::module_ &module) {
  py::enum_<planning::PlanningOutcome>(module, "PlanningOutcome")
      .value("NEW_REFERENCE_AVAILABLE",
             planning::PlanningOutcome::kNewReferenceAvailable)
      .value("SAFE_FRONTIER_REFERENCE_AVAILABLE",
             planning::PlanningOutcome::kSafeFrontierReferenceAvailable)
      .value("NO_KNOWN_SAFE_ROUTE",
             planning::PlanningOutcome::kNoKnownSafeRoute)
      .value("GOAL_INFEASIBLE", planning::PlanningOutcome::kGoalInfeasible)
      .value("INVALID_REQUEST", planning::PlanningOutcome::kInvalidRequest)
      .value("STALE_INPUT", planning::PlanningOutcome::kStaleInput)
      .value("NUMERICAL_FAILURE", planning::PlanningOutcome::kNumericalFailure)
      .value("RESOURCE_EXHAUSTED",
             planning::PlanningOutcome::kResourceExhausted)
      .value("ACTIVE_REFERENCE_INVALIDATED",
             planning::PlanningOutcome::kActiveReferenceInvalidated)
      .value("CANCELED", planning::PlanningOutcome::kCanceled);
  py::enum_<planning::ExecutionDirective>(module, "ExecutionDirective")
      .value("ACTIVATE_NEW_REFERENCE",
             planning::ExecutionDirective::kActivateNewReference)
      .value("CONTINUE_ACTIVE_REFERENCE",
             planning::ExecutionDirective::kContinueActiveReference)
      .value("HOLD_POSITION", planning::ExecutionDirective::kHoldPosition)
      .value("CONTINUE_COMMITTED_HOP",
             planning::ExecutionDirective::kContinueCommittedHop)
      .value("NO_SAFE_REFERENCE",
             planning::ExecutionDirective::kNoSafeReference);
  py::enum_<planning::TrajectorySemantics>(module, "TrajectorySemantics")
      .value("WHEELED_BASE", planning::TrajectorySemantics::kWheeledBase)
      .value("LEGGED_BODY_REFERENCE",
             planning::TrajectorySemantics::kLeggedBodyReference);
  py::class_<planning::TrajectoryPoint>(module, "TrajectoryPoint")
      .def(py::init<>())
      .def_readwrite("time_from_start",
                     &planning::TrajectoryPoint::time_from_start)
      .def_readwrite("pose", &planning::TrajectoryPoint::pose)
      .def_readwrite("velocity", &planning::TrajectoryPoint::velocity);
  py::class_<planning::TrajectoryReference>(module, "TrajectoryReference")
      .def(py::init<>())
      .def_readwrite("semantics", &planning::TrajectoryReference::semantics)
      .def_readwrite("points", &planning::TrajectoryReference::points);
  py::class_<planning::HopSegment>(module, "HopSegment")
      .def(py::init<>())
      .def_readwrite("segment_id", &planning::HopSegment::segment_id)
      .def_readwrite("launch_pose", &planning::HopSegment::launch_pose)
      .def_readwrite("landing_region_boundary_m",
                     &planning::HopSegment::landing_region_boundary_m)
      .def_readwrite("flight_time", &planning::HopSegment::flight_time)
      .def_readwrite("launch_velocity_mps",
                     &planning::HopSegment::launch_velocity_mps)
      .def_readwrite("flight_tube_radius_m",
                     &planning::HopSegment::flight_tube_radius_m)
      .def_readwrite("nominal_landing_point_m",
                     &planning::HopSegment::nominal_landing_point_m)
      .def_readwrite("required_delta_v_mps",
                     &planning::HopSegment::required_delta_v_mps)
      .def_readwrite("available_delta_v_mps",
                     &planning::HopSegment::available_delta_v_mps)
      .def_readwrite("capability_version",
                     &planning::HopSegment::capability_version)
      .def_readwrite("global_map_generation",
                     &planning::HopSegment::global_map_generation)
      .def_readwrite("local_map_generation",
                     &planning::HopSegment::local_map_generation);
  py::class_<planning::HopReference>(module, "HopReference")
      .def(py::init<>())
      .def_readwrite("segments", &planning::HopReference::segments);
  py::class_<planning::MotionReference>(module, "MotionReference")
      .def(py::init<>())
      .def_readwrite("plan_id", &planning::MotionReference::plan_id)
      .def_property(
          "platform_type",
          [](const planning::MotionReference &self) {
            return PlatformTypeName(self.platform_type);
          },
          [](planning::MotionReference &self, const py::object &value) {
            self.platform_type = PlatformTypeFromPython(value);
          })
      .def_readwrite("input_time", &planning::MotionReference::input_time)
      .def_readwrite("data", &planning::MotionReference::data);
  py::class_<planning::PlannerDiagnostics>(module, "PlannerDiagnostics")
      .def(py::init<>())
      .def_readwrite("planner_name",
                     &planning::PlannerDiagnostics::planner_name)
      .def_readwrite("elapsed", &planning::PlannerDiagnostics::elapsed)
      .def_readwrite("expanded_states",
                     &planning::PlannerDiagnostics::expanded_states)
      .def_readwrite("best_cost", &planning::PlannerDiagnostics::best_cost)
      .def_readwrite("warning_codes",
                     &planning::PlannerDiagnostics::warning_codes);
  py::class_<planning::PlannerOutput>(module, "PlannerOutput")
      .def(py::init<>())
      .def_readwrite("outcome", &planning::PlannerOutput::outcome)
      .def_readwrite("directive", &planning::PlannerOutput::directive)
      .def_readwrite("reason_code", &planning::PlannerOutput::reason_code)
      .def_readwrite("reference", &planning::PlannerOutput::reference)
      .def_readwrite("diagnostics", &planning::PlannerOutput::diagnostics);
}

[[nodiscard]] planning::HopperLandingEvidenceGrid HopperLandingGridFromArrays(
    const py::array &certified,
    const py::array &aim_positions_m,
    const py::array &boundary_m,
    const py::array &area_m2,
    const std::string &algorithm_id) {
  RequireExactArray(certified, py::dtype::of<bool>(), 2,
                    "certified landing mask", "bool");
  RequireExactArray(aim_positions_m, py::dtype::of<double>(), 3,
                    "landing aim positions", "float64");
  RequireExactArray(boundary_m, py::dtype::of<double>(), 4,
                    "landing boundaries", "float64");
  RequireExactArray(area_m2, py::dtype::of<double>(), 2,
                    "landing areas", "float64");
  if (certified.shape(0) <= 0 || certified.shape(1) <= 0 ||
      aim_positions_m.shape(0) != certified.shape(0) ||
      aim_positions_m.shape(1) != certified.shape(1) ||
      aim_positions_m.shape(2) != 3 ||
      boundary_m.shape(0) != certified.shape(0) ||
      boundary_m.shape(1) != certified.shape(1) ||
      boundary_m.shape(2) != 4 || boundary_m.shape(3) != 3 ||
      area_m2.shape(0) != certified.shape(0) ||
      area_m2.shape(1) != certified.shape(1)) {
    throw py::value_error("hopper landing evidence shape mismatch");
  }
  if (algorithm_id.empty()) {
    throw py::value_error("hopper landing evidence algorithm is missing");
  }
  RequireFiniteDoubleArray(aim_positions_m, "landing aim positions");
  RequireFiniteDoubleArray(boundary_m, "landing boundaries");
  RequireFiniteDoubleArray(area_m2, "landing areas");
  const auto *certified_data = static_cast<const bool *>(certified.data());
  const auto *aim_data = static_cast<const double *>(aim_positions_m.data());
  const auto *boundary_data = static_cast<const double *>(boundary_m.data());
  const auto *area_data = static_cast<const double *>(area_m2.data());
  planning::HopperLandingEvidenceGrid result{
      .width = static_cast<std::size_t>(certified.shape(1)),
      .height = static_cast<std::size_t>(certified.shape(0)),
      .algorithm_id = algorithm_id,
  };
  result.landings.reserve(static_cast<std::size_t>(certified.size()));
  for (py::ssize_t index = 0; index < certified.size(); ++index) {
    planning::HopperLandingEvidence evidence{
        .certified = static_cast<std::uint8_t>(certified_data[index]),
        .aim_position_on_surface_m = planning::Vec3{
            .x = aim_data[index * 3],
            .y = aim_data[index * 3 + 1],
            .z = aim_data[index * 3 + 2],
        },
        .area_m2 = area_data[index],
    };
    for (std::size_t vertex = 0U; vertex < evidence.boundary_m.size();
         ++vertex) {
      const std::size_t offset =
          static_cast<std::size_t>(index) * 12U + vertex * 3U;
      evidence.boundary_m[vertex] = planning::Vec3{
          .x = boundary_data[offset],
          .y = boundary_data[offset + 1U],
          .z = boundary_data[offset + 2U],
      };
    }
    result.landings.push_back(std::move(evidence));
  }
  return result;
}

void BindProjection(py::module_ &module) {
  py::class_<planning::TraversabilityProjection>(
      module, "TraversabilityProjection")
      .def_property_readonly(
          "platform_type",
          [](const planning::TraversabilityProjection &self) {
            return PlatformTypeName(self.platform_type);
          })
      .def_readonly("width", &planning::TraversabilityProjection::width)
      .def_readonly("height", &planning::TraversabilityProjection::height)
      .def_property_readonly(
          "known",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(self.known, self.height, self.width);
          })
      .def_property_readonly(
          "intrinsic_feasible",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(
                self.intrinsic_feasible, self.height, self.width);
          })
      .def_property_readonly(
          "hard_feasible",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(
                self.hard_feasible, self.height, self.width);
          })
      .def_property_readonly(
          "clearance_m",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(self.clearance_m, self.height, self.width);
          })
      .def_property_readonly(
          "slope_rad",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(self.slope_rad, self.height, self.width);
          })
      .def_property_readonly(
          "roughness_m",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(self.roughness_m, self.height, self.width);
          })
      .def_property_readonly(
          "traversal_cost",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(
                self.traversal_cost, self.height, self.width);
          })
      .def_property_readonly(
          "connected_component",
          [](const planning::TraversabilityProjection &self) {
            return CopyArray2d(
                self.connected_component, self.height, self.width);
          });
  py::class_<planning::ReachabilityProjection>(
      module, "ReachabilityProjection")
      .def_property_readonly(
          "platform_type",
          [](const planning::ReachabilityProjection &self) {
            return PlatformTypeName(self.platform_type);
          })
      .def_readonly("width", &planning::ReachabilityProjection::width)
      .def_readonly("height", &planning::ReachabilityProjection::height)
      .def_property_readonly(
          "reachable",
          [](const planning::ReachabilityProjection &self) {
            return CopyArray2d(self.reachable, self.height, self.width);
          })
      .def_readonly("algorithm_id",
                    &planning::ReachabilityProjection::algorithm_id)
      .def_readonly("maximum_edge_distance_m",
                    &planning::ReachabilityProjection::maximum_edge_distance_m)
      .def_readonly(
          "candidate_edges_evaluated",
          &planning::ReachabilityProjection::candidate_edges_evaluated)
      .def_readonly("certified_edges",
                    &planning::ReachabilityProjection::certified_edges)
      .def_readonly("rejected_edges",
                    &planning::ReachabilityProjection::rejected_edges)
      .def_readonly(
          "maximum_certified_edge_distance_m",
          &planning::ReachabilityProjection::maximum_certified_edge_distance_m);
  py::class_<planning::HopperLandingEvidenceGrid>(
      module, "HopperLandingEvidenceGrid")
      .def(py::init(&HopperLandingGridFromArrays),
           py::arg("certified"), py::arg("aim_positions_m"),
           py::arg("boundary_m"), py::arg("area_m2"),
           py::arg("algorithm_id"))
      .def_readonly("width", &planning::HopperLandingEvidenceGrid::width)
      .def_readonly("height", &planning::HopperLandingEvidenceGrid::height)
      .def_readonly("algorithm_id",
                    &planning::HopperLandingEvidenceGrid::algorithm_id);
  py::class_<planning::HopperLandingEvidenceProjection>(
      module, "HopperLandingEvidenceProjection")
      .def_property_readonly(
          "certified",
          [](const planning::HopperLandingEvidenceProjection &self) {
            py::array_t<bool> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.landings.size())});
            std::transform(
                self.landings.begin(), self.landings.end(),
                result.mutable_data(), [](const auto &landing) {
                  return landing.certified != 0U;
                });
            return result;
          })
      .def_property_readonly(
          "aim_positions_m",
          [](const planning::HopperLandingEvidenceProjection &self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.landings.size()),
                static_cast<py::ssize_t>(3)});
            double *output = result.mutable_data();
            for (std::size_t index = 0U; index < self.landings.size();
                 ++index) {
              const auto value = self.landings[index].aim_position_on_surface_m;
              output[index * 3U] = value.x;
              output[index * 3U + 1U] = value.y;
              output[index * 3U + 2U] = value.z;
            }
            return result;
          })
      .def_property_readonly(
          "boundary_m",
          [](const planning::HopperLandingEvidenceProjection &self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.landings.size()),
                static_cast<py::ssize_t>(4),
                static_cast<py::ssize_t>(3)});
            double *output = result.mutable_data();
            for (std::size_t index = 0U; index < self.landings.size();
                 ++index) {
              for (std::size_t vertex = 0U; vertex < 4U; ++vertex) {
                const auto value = self.landings[index].boundary_m[vertex];
                const std::size_t offset = index * 12U + vertex * 3U;
                output[offset] = value.x;
                output[offset + 1U] = value.y;
                output[offset + 2U] = value.z;
              }
            }
            return result;
          })
      .def_property_readonly(
          "area_m2",
          [](const planning::HopperLandingEvidenceProjection &self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.landings.size())});
            std::transform(
                self.landings.begin(), self.landings.end(),
                result.mutable_data(),
                [](const auto &landing) { return landing.area_m2; });
            return result;
          })
      .def_readonly("algorithm_id",
                    &planning::HopperLandingEvidenceProjection::algorithm_id)
      .def_readonly(
          "candidates_evaluated",
          &planning::HopperLandingEvidenceProjection::candidates_evaluated)
      .def_readonly("certified_count",
                    &planning::HopperLandingEvidenceProjection::certified_count);
}

void BindPrimitiveReachability(py::module_& module) {
  auto snapshot_class = py::class_<planning::PrimitiveReachabilitySnapshot>(
      module, "PrimitiveReachabilitySnapshot")
      .def_property_readonly(
          "platform_type",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            return PlatformTypeName(self.platform_type);
          })
      .def_readonly("width", &planning::PrimitiveReachabilitySnapshot::width)
      .def_readonly("height", &planning::PrimitiveReachabilitySnapshot::height)
      .def_readonly(
          "algorithm_id",
          &planning::PrimitiveReachabilitySnapshot::algorithm_id)
      .def_readonly(
          "state_schema",
          &planning::PrimitiveReachabilitySnapshot::state_schema)
      .def_readonly(
          "primitive_set_sha256",
          &planning::PrimitiveReachabilitySnapshot::primitive_set_sha256)
      .def_readonly(
          "world_evidence_sha256",
          &planning::PrimitiveReachabilitySnapshot::world_evidence_sha256)
      .def_readonly(
          "graph_sha256",
          &planning::PrimitiveReachabilitySnapshot::graph_sha256)
      .def_readonly(
          "revision", &planning::PrimitiveReachabilitySnapshot::revision)
      .def_readonly(
          "invalidated_edge_count",
          &planning::PrimitiveReachabilitySnapshot::invalidated_edge_count)
      .def_readonly(
          "revalidated_edge_count",
          &planning::PrimitiveReachabilitySnapshot::revalidated_edge_count)
      .def_property_readonly(
          "state_ids",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<std::uint64_t> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size())});
            std::transform(
                self.states.begin(), self.states.end(), result.mutable_data(),
                [](const auto& state) { return state.state_id; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "positions_m",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size()),
                static_cast<py::ssize_t>(3)});
            double* output = result.mutable_data();
            for (std::size_t index = 0U; index < self.states.size(); ++index) {
              output[index * 3U] = self.states[index].position_m.x;
              output[index * 3U + 1U] = self.states[index].position_m.y;
              output[index * 3U + 2U] = self.states[index].position_m.z;
            }
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "yaw_rad",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size())});
            std::transform(
                self.states.begin(), self.states.end(), result.mutable_data(),
                [](const auto& state) { return state.yaw_rad; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "cells",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<std::int32_t> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size()),
                static_cast<py::ssize_t>(2)});
            std::int32_t* output = result.mutable_data();
            for (std::size_t index = 0U; index < self.states.size(); ++index) {
              output[index * 2U] = self.states[index].cell_y;
              output[index * 2U + 1U] = self.states[index].cell_x;
            }
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "yaw_bin",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<std::int32_t> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size())});
            std::transform(
                self.states.begin(), self.states.end(), result.mutable_data(),
                [](const auto& state) { return state.yaw_bin; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "motion_mode",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<std::int32_t> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size())});
            std::transform(
                self.states.begin(), self.states.end(), result.mutable_data(),
                [](const auto& state) { return state.motion_mode; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "body_z_m",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size()),
                static_cast<py::ssize_t>(2)});
            double* output = result.mutable_data();
            for (std::size_t index = 0U; index < self.states.size(); ++index) {
              output[index * 2U] = self.states[index].body_z_m.lower;
              output[index * 2U + 1U] = self.states[index].body_z_m.upper;
            }
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "path_cost",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.states.size())});
            std::transform(
                self.states.begin(), self.states.end(), result.mutable_data(),
                [](const auto& state) { return state.path_cost; });
            return ReadonlyArray(std::move(result));
          });
  const auto state_flag = [](
                              const planning::PrimitiveReachabilitySnapshot& self,
                              const auto predicate) {
    py::array_t<bool> result(py::array::ShapeContainer{
        static_cast<py::ssize_t>(self.states.size())});
    std::transform(
        self.states.begin(), self.states.end(), result.mutable_data(),
        predicate);
    return ReadonlyArray(std::move(result));
  };
  snapshot_class
      .def_property_readonly(
          "forward_reachable",
          [state_flag](const planning::PrimitiveReachabilitySnapshot& self) {
            return state_flag(self, [](const auto& state) {
              return state.forward_reachable != 0U;
            });
          })
      .def_property_readonly(
          "returnable",
          [state_flag](const planning::PrimitiveReachabilitySnapshot& self) {
            return state_flag(self, [](const auto& state) {
              return state.returnable != 0U;
            });
          })
      .def_property_readonly(
          "observation_state",
          [state_flag](const planning::PrimitiveReachabilitySnapshot& self) {
            return state_flag(self, [](const auto& state) {
              return state.observation_state != 0U;
            });
          })
      .def_property_readonly(
          "direct_successor",
          [state_flag](const planning::PrimitiveReachabilitySnapshot& self) {
            return state_flag(self, [](const auto& state) {
              return state.direct_successor != 0U;
            });
          })
      .def_property_readonly(
          "recoverable",
          [state_flag](const planning::PrimitiveReachabilitySnapshot& self) {
            return state_flag(self, [](const auto& state) {
              return state.forward_reachable != 0U && state.returnable != 0U;
            });
          })
      .def_property_readonly(
          "edge_source_ids",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<std::uint64_t> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.edges.size())});
            std::transform(
                self.edges.begin(), self.edges.end(), result.mutable_data(),
                [](const auto& edge) { return edge.source_state_id; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "edge_target_ids",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<std::uint64_t> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.edges.size())});
            std::transform(
                self.edges.begin(), self.edges.end(), result.mutable_data(),
                [](const auto& edge) { return edge.target_state_id; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "edge_primitive_indices",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<std::uint32_t> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.edges.size())});
            std::transform(
                self.edges.begin(), self.edges.end(), result.mutable_data(),
                [](const auto& edge) { return edge.primitive_index; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "edge_primitive_ids",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            std::vector<std::string> result;
            result.reserve(self.edges.size());
            std::transform(
                self.edges.begin(), self.edges.end(),
                std::back_inserter(result),
                [](const auto& edge) { return edge.primitive_id; });
            return result;
          })
      .def_property_readonly(
          "edge_cost",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            py::array_t<double> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(self.edges.size())});
            std::transform(
                self.edges.begin(), self.edges.end(), result.mutable_data(),
                [](const auto& edge) { return edge.cost; });
            return ReadonlyArray(std::move(result));
          })
      .def_property_readonly(
          "reachable",
          [](const planning::PrimitiveReachabilitySnapshot& self) {
            return ReadonlyArray(
                CopyArray2d(self.reachable, self.height, self.width));
          });

  py::class_<training::TrainingPrimitiveReachabilityEngine>(
      module, "PrimitiveReachabilityEngine")
      .def(py::init<>())
      .def(
          "update",
          [](training::TrainingPrimitiveReachabilityEngine& self,
             const training::TrainingPlanRequest& request,
             const std::optional<double> maximum_action_distance_m) {
            planning::PrimitiveReachabilityResult result;
            {
              py::gil_scoped_release release;
              result = self.Update(request, maximum_action_distance_m);
            }
            if (!result.ok()) {
              throw std::runtime_error(result.reason_code);
            }
            return std::move(*result.snapshot);
          },
          py::arg("request"),
          py::arg("maximum_action_distance_m") = std::nullopt)
      .def(
          "reset", &training::TrainingPrimitiveReachabilityEngine::Reset,
          py::call_guard<py::gil_scoped_release>());
}

void BindVisibility(py::module_ &module) {
  py::class_<training::VisibilityKernel>(module, "VisibilityKernel")
      .def(py::init<double, double>(), py::arg("resolution_m"),
           py::arg("range_m"))
      .def_property_readonly("resolution_m",
                             &training::VisibilityKernel::resolution_m)
      .def_property_readonly("range_m",
                             &training::VisibilityKernel::range_m)
      .def(
          "estimate_candidate_gains",
          [](const training::VisibilityKernel &self,
             const py::array &observed, const py::array &obstacle_ratio,
             const py::array &roi_ratio,
             const py::array &priority_weight,
             const py::array &candidate_cells) {
            RequireExactArray(observed, py::dtype::of<bool>(), 2,
                              "observed", "bool");
            RequireExactArray(obstacle_ratio, py::dtype::of<float>(), 2,
                              "obstacle ratio", "float32");
            RequireExactArray(roi_ratio, py::dtype::of<float>(), 2,
                              "ROI ratio", "float32");
            RequireExactArray(priority_weight, py::dtype::of<float>(), 2,
                              "priority weight", "float32");
            RequireExactArray(candidate_cells,
                              py::dtype::of<std::int32_t>(), 2,
                              "candidate cells", "int32");
            if (candidate_cells.shape(1) != 2) {
              throw py::value_error("candidate cells must have shape [N,2]");
            }
            RequireSameGridShape(observed, obstacle_ratio, "obstacle ratio");
            RequireSameGridShape(observed, roi_ratio, "ROI ratio");
            RequireSameGridShape(
                observed, priority_weight, "priority weight");
            RequireFiniteFloatArray(obstacle_ratio, "obstacle ratio");
            RequireFiniteFloatArray(roi_ratio, "ROI ratio");
            RequireFiniteFloatArray(priority_weight, "priority weight");

            const training::GridShape shape{
                .height = static_cast<std::size_t>(observed.shape(0)),
                .width = static_cast<std::size_t>(observed.shape(1)),
            };
            const auto *observed_data =
                static_cast<const bool *>(observed.data());
            std::vector<std::uint8_t> observed_copy(
                static_cast<std::size_t>(observed.size()), 0U);
            std::transform(
                observed_data, observed_data + observed.size(),
                observed_copy.begin(),
                [](const bool value) { return value ? 1U : 0U; });
            const auto *candidate_data =
                static_cast<const std::int32_t *>(candidate_cells.data());
            std::vector<training::GridCell> candidates;
            candidates.reserve(
                static_cast<std::size_t>(candidate_cells.shape(0)));
            for (py::ssize_t index = 0; index < candidate_cells.shape(0);
                 ++index) {
              candidates.push_back(training::GridCell{
                  .row = candidate_data[index * 2],
                  .column = candidate_data[index * 2 + 1],
              });
              if (candidates.back().row < 0 ||
                  candidates.back().column < 0 ||
                  static_cast<std::size_t>(candidates.back().row) >=
                      shape.height ||
                  static_cast<std::size_t>(candidates.back().column) >=
                      shape.width) {
                throw py::value_error("candidate cell is outside the grid");
              }
            }

            std::vector<training::CandidateGain> gains;
            {
              py::gil_scoped_release release;
              gains = self.EstimateCandidateGains(
                  shape, observed_copy,
                  std::span<const float>{
                      static_cast<const float *>(obstacle_ratio.data()),
                      static_cast<std::size_t>(obstacle_ratio.size())},
                  std::span<const float>{
                      static_cast<const float *>(roi_ratio.data()),
                      static_cast<std::size_t>(roi_ratio.size())},
                  std::span<const float>{
                      static_cast<const float *>(priority_weight.data()),
                      static_cast<std::size_t>(priority_weight.size())},
                  candidates);
            }
            py::array_t<float> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(gains.size()),
                static_cast<py::ssize_t>(2)});
            auto *output = result.mutable_data();
            for (std::size_t index = 0U; index < gains.size(); ++index) {
              output[index * 2U] = gains[index].roi;
              output[index * 2U + 1U] = gains[index].priority;
            }
            return result;
          },
          py::arg("observed"), py::arg("obstacle_ratio"),
          py::arg("roi_ratio"), py::arg("priority_weight"),
          py::arg("candidate_cells"))
      .def(
          "reveal_from_pose",
          [](const training::VisibilityKernel &self,
             const py::array &truth_obstacle_ratio,
             const std::int32_t pose_row,
             const std::int32_t pose_column) {
            RequireExactArray(truth_obstacle_ratio,
                              py::dtype::of<float>(), 2,
                              "truth obstacle ratio", "float32");
            RequireFiniteFloatArray(
                truth_obstacle_ratio, "truth obstacle ratio");
            const training::GridShape shape{
                .height = static_cast<std::size_t>(
                    truth_obstacle_ratio.shape(0)),
                .width = static_cast<std::size_t>(
                    truth_obstacle_ratio.shape(1)),
            };
            if (pose_row < 0 || pose_column < 0 ||
                static_cast<std::size_t>(pose_row) >= shape.height ||
                static_cast<std::size_t>(pose_column) >= shape.width) {
              throw py::value_error("visibility pose is outside the grid");
            }
            std::vector<std::uint8_t> visible;
            {
              py::gil_scoped_release release;
              visible = self.RevealFromPose(
                  shape,
                  training::GridCell{
                      .row = pose_row,
                      .column = pose_column,
                  },
                  std::span<const float>{
                      static_cast<const float *>(
                          truth_obstacle_ratio.data()),
                      static_cast<std::size_t>(
                          truth_obstacle_ratio.size())});
            }
            py::array_t<bool> result(py::array::ShapeContainer{
                static_cast<py::ssize_t>(shape.height),
                static_cast<py::ssize_t>(shape.width)});
            std::transform(visible.begin(), visible.end(),
                           result.mutable_data(),
                           [](const auto value) { return value != 0U; });
            return result;
          },
          py::arg("truth_obstacle_ratio"), py::arg("pose_row"),
          py::arg("pose_column"));
}

void BindRequest(py::module_ &module) {
  py::class_<training::TrainingPlanRequest>(module, "TrainingPlanRequest")
      .def(py::init<>())
      .def_readwrite("request_id", &training::TrainingPlanRequest::request_id)
      .def_readwrite("mission_id", &training::TrainingPlanRequest::mission_id)
      .def_readwrite("mission_revision",
                     &training::TrainingPlanRequest::mission_revision)
      .def_readwrite("platform_id",
                     &training::TrainingPlanRequest::platform_id)
      .def_readwrite("capability_version",
                     &training::TrainingPlanRequest::capability_version)
      .def_readwrite("global_map_generation",
                     &training::TrainingPlanRequest::global_map_generation)
      .def_readwrite("local_map_generation",
                     &training::TrainingPlanRequest::local_map_generation)
      .def_readwrite("map_from_odom_generation",
                     &training::TrainingPlanRequest::map_from_odom_generation)
      .def_readwrite("state_time", &training::TrainingPlanRequest::state_time)
      .def_readwrite("current_state",
                     &training::TrainingPlanRequest::current_state)
      .def_readwrite("goal", &training::TrainingPlanRequest::goal)
      .def_readwrite("world", &training::TrainingPlanRequest::world)
      .def_readwrite("capability", &training::TrainingPlanRequest::capability)
      .def_readwrite("config", &training::TrainingPlanRequest::config)
      .def_readwrite("previous_execution",
                     &training::TrainingPlanRequest::previous_execution)
      .def_readwrite("position_uncertainty_m",
                     &training::TrainingPlanRequest::position_uncertainty_m)
      .def_readwrite("velocity_uncertainty_mps",
                     &training::TrainingPlanRequest::velocity_uncertainty_mps);
  py::class_<training::PlannerBridge>(module, "PlannerBridge")
      .def(py::init<>())
      .def("plan", &training::PlannerBridge::Plan, py::arg("request"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "project_traversability",
          [](const training::PlannerBridge &self,
             const training::TrainingPlanRequest &request) {
            auto result = self.ProjectTraversability(request);
            if (!result.ok()) {
              throw std::runtime_error(result.reason_code);
            }
            return std::move(*result.projection);
          },
          py::arg("request"), py::call_guard<py::gil_scoped_release>())
      .def(
          "project_reachability",
          [](const training::PlannerBridge &self,
             const training::TrainingPlanRequest &request,
             const double maximum_edge_distance_m) {
            auto result = self.ProjectReachability(
                request, maximum_edge_distance_m);
            if (!result.ok()) {
              throw std::runtime_error(result.reason_code);
            }
            return std::move(*result.projection);
          },
          py::arg("request"), py::arg("maximum_edge_distance_m"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "project_reachability",
          [](const training::PlannerBridge &self,
             const training::TrainingPlanRequest &request,
             const double maximum_edge_distance_m,
             const planning::HopperLandingEvidenceGrid &evidence) {
            auto result = self.ProjectReachability(
                request, maximum_edge_distance_m, evidence);
            if (!result.ok()) {
              throw std::runtime_error(result.reason_code);
            }
            return std::move(*result.projection);
          },
          py::arg("request"), py::arg("maximum_edge_distance_m"),
          py::arg("hopper_landing_evidence"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "project_direct_hopper_reachability",
          [](const training::PlannerBridge &self,
             const training::TrainingPlanRequest &request,
             const double maximum_edge_distance_m,
             const planning::HopperLandingEvidenceGrid &evidence) {
            auto result = self.ProjectDirectHopperReachability(
                request, maximum_edge_distance_m, evidence);
            if (!result.ok()) {
              throw std::runtime_error(result.reason_code);
            }
            return std::move(*result.projection);
          },
          py::arg("request"), py::arg("maximum_edge_distance_m"),
          py::arg("hopper_landing_evidence"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "project_hopper_landing_evidence",
          [](const training::PlannerBridge &self,
             const training::TrainingPlanRequest &request,
             const py::array &target_positions_m) {
            RequireExactArray(
                target_positions_m, py::dtype::of<double>(), 2,
                "hopper landing targets", "float64");
            if (target_positions_m.shape(1) != 3) {
              throw py::value_error(
                  "hopper landing targets must have shape [N,3]");
            }
            RequireFiniteDoubleArray(
                target_positions_m, "hopper landing targets");
            const auto *data =
                static_cast<const double *>(target_positions_m.data());
            std::vector<planning::Vec3> targets;
            targets.reserve(
                static_cast<std::size_t>(target_positions_m.shape(0)));
            for (py::ssize_t index = 0;
                 index < target_positions_m.shape(0); ++index) {
              targets.push_back(planning::Vec3{
                  .x = data[index * 3],
                  .y = data[index * 3 + 1],
                  .z = data[index * 3 + 2],
              });
            }
            planning::HopperLandingEvidenceProjectionResult result;
            {
              py::gil_scoped_release release;
              result = self.ProjectHopperLandingEvidence(request, targets);
            }
            if (!result.ok()) {
              throw std::runtime_error(result.reason_code);
            }
            return std::move(*result.projection);
          },
          py::arg("request"), py::arg("target_positions_m"));
}

}  // namespace

PYBIND11_MODULE(_lunar_planner_training_bridge, module) {
  module.doc() = "In-process Python bindings for lunar::planning::Planner";
  module.def(
      "validate_urdf_geometry", &ValidateUrdfGeometry,
      py::arg("document"), py::arg("base_frame_id"));
  BindGeometry(module);
  BindWorld(module);
  BindGoalsAndState(module);
  BindCapabilities(module);
  BindConfig(module);
  BindExecution(module);
  BindOutput(module);
  BindProjection(module);
  BindVisibility(module);
  BindRequest(module);
  BindPrimitiveReachability(module);
}
