#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <urdf/model.h>
#include <urdf_model/link.h>

#include "lunar_planner_training_bridge/request.hpp"

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
                     &planning::WheelMotionPrimitive::relative_end_pose)
      .def_readwrite("nominal_duration",
                     &planning::WheelMotionPrimitive::nominal_duration)
      .def_property(
          "nominal_duration_ns",
          [](const planning::WheelMotionPrimitive &value) {
            return value.nominal_duration.count();
          },
          [](planning::WheelMotionPrimitive &value, const std::int64_t count) {
            value.nominal_duration = std::chrono::nanoseconds{count};
          });
  py::class_<planning::WheeledCapability>(module, "WheeledCapability")
      .def(py::init<>())
      .def_readwrite("footprint_xy_m",
                     &planning::WheeledCapability::footprint_xy_m)
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
                     &planning::LeggedBodyPrimitive::yaw_change_rad)
      .def_readwrite("nominal_duration",
                     &planning::LeggedBodyPrimitive::nominal_duration)
      .def_property(
          "nominal_duration_ns",
          [](const planning::LeggedBodyPrimitive &value) {
            return value.nominal_duration.count();
          },
          [](planning::LeggedBodyPrimitive &value, const std::int64_t count) {
            value.nominal_duration = std::chrono::nanoseconds{count};
          });
  py::class_<planning::LeggedCapability>(module, "LeggedCapability")
      .def(py::init<>())
      .def_readwrite("body_half_extent_m",
                     &planning::LeggedCapability::body_half_extent_m)
      .def_readwrite("maximum_slope_rad",
                     &planning::LeggedCapability::maximum_slope_rad)
      .def_readwrite("maximum_roughness_m",
                     &planning::LeggedCapability::maximum_roughness_m)
      .def_readwrite("maximum_step_height_m",
                     &planning::LeggedCapability::maximum_step_height_m)
      .def_readwrite("maximum_gap_width_m",
                     &planning::LeggedCapability::maximum_gap_width_m)
      .def_readwrite("minimum_confidence",
                     &planning::LeggedCapability::minimum_confidence)
      .def_readwrite("minimum_body_clearance_m",
                     &planning::LeggedCapability::minimum_body_clearance_m)
      .def_readwrite("body_height_m",
                     &planning::LeggedCapability::body_height_m)
      .def_readwrite("forward_speed_mps",
                     &planning::LeggedCapability::forward_speed_mps)
      .def_readwrite("lateral_speed_mps",
                     &planning::LeggedCapability::lateral_speed_mps)
      .def_readwrite("vertical_speed_mps",
                     &planning::LeggedCapability::vertical_speed_mps)
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
      .def_readwrite("body_half_extent_m",
                     &planning::HopperCapability::body_half_extent_m)
      .def_readwrite("platform_mass_kg",
                     &planning::HopperCapability::platform_mass_kg)
      .def_readwrite("gravity_mps2", &planning::HopperCapability::gravity_mps2)
      .def_readwrite("maximum_landing_slope_rad",
                     &planning::HopperCapability::maximum_landing_slope_rad)
      .def_readwrite("maximum_landing_roughness_m",
                     &planning::HopperCapability::maximum_landing_roughness_m)
      .def_readwrite("maximum_plane_residual_m",
                     &planning::HopperCapability::maximum_plane_residual_m)
      .def_readwrite("minimum_overhead_clearance_m",
                     &planning::HopperCapability::minimum_overhead_clearance_m)
      .def_readwrite("minimum_lateral_clearance_m",
                     &planning::HopperCapability::minimum_lateral_clearance_m)
      .def_readwrite(
          "minimum_landing_region_area_m2",
          &planning::HopperCapability::minimum_landing_region_area_m2)
      .def_readwrite("maximum_launch_speed_mps",
                     &planning::HopperCapability::maximum_launch_speed_mps)
      .def_readwrite(
          "maximum_launch_impulse_newton_seconds",
          &planning::HopperCapability::maximum_launch_impulse_newton_seconds)
      .def_readwrite("minimum_flight_time",
                     &planning::HopperCapability::minimum_flight_time)
      .def_property(
          "minimum_flight_time_ns",
          [](const planning::HopperCapability &value) {
            return value.minimum_flight_time.count();
          },
          [](planning::HopperCapability &value, const std::int64_t count) {
            value.minimum_flight_time = std::chrono::nanoseconds{count};
          })
      .def_readwrite("maximum_flight_time",
                     &planning::HopperCapability::maximum_flight_time)
      .def_property(
          "maximum_flight_time_ns",
          [](const planning::HopperCapability &value) {
            return value.maximum_flight_time.count();
          },
          [](planning::HopperCapability &value, const std::int64_t count) {
            value.maximum_flight_time = std::chrono::nanoseconds{count};
          })
      .def_readwrite("maximum_landing_speed_mps",
                     &planning::HopperCapability::maximum_landing_speed_mps)
      .def_readwrite(
          "minimum_downward_impact_speed_mps",
          &planning::HopperCapability::minimum_downward_impact_speed_mps)
      .def_readwrite("minimum_landing_clearance_m",
                     &planning::HopperCapability::minimum_landing_clearance_m)
      .def_readwrite("maximum_angular_speed_radps",
                     &planning::HopperCapability::maximum_angular_speed_radps)
      .def_readwrite(
          "maximum_angular_acceleration_radps2",
          &planning::HopperCapability::maximum_angular_acceleration_radps2)
      .def_readwrite(
          "maximum_initial_angular_speed_radps",
          &planning::HopperCapability::maximum_initial_angular_speed_radps)
      .def_readwrite("minimum_settle_guard",
                     &planning::HopperCapability::minimum_settle_guard)
      .def_property(
          "minimum_settle_guard_ns",
          [](const planning::HopperCapability &value) {
            return value.minimum_settle_guard.count();
          },
          [](planning::HopperCapability &value, const std::int64_t count) {
            value.minimum_settle_guard = std::chrono::nanoseconds{count};
          });
}

void BindConfig(py::module_ &module) {
  py::class_<planning::SearchResourceLimits>(module, "SearchResourceLimits")
      .def(py::init<>())
      .def_readwrite("maximum_expanded_states",
                     &planning::SearchResourceLimits::maximum_expanded_states)
      .def_readwrite("maximum_reopened_states",
                     &planning::SearchResourceLimits::maximum_reopened_states)
      .def_readwrite(
          "maximum_generated_candidates",
          &planning::SearchResourceLimits::maximum_generated_candidates)
      .def_readwrite("maximum_open_states",
                     &planning::SearchResourceLimits::maximum_open_states)
      .def_readwrite("maximum_memory_bytes",
                     &planning::SearchResourceLimits::maximum_memory_bytes);
  py::class_<planning::AraStarConfig>(module, "AraStarConfig")
      .def(py::init<>())
      .def_readwrite("initial_epsilon",
                     &planning::AraStarConfig::initial_epsilon)
      .def_readwrite("epsilon_decrement",
                     &planning::AraStarConfig::epsilon_decrement)
      .def_readwrite("target_epsilon", &planning::AraStarConfig::target_epsilon)
      .def_readwrite("resources", &planning::AraStarConfig::resources);
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
                     &planning::OptimizationConfig::constraint_tolerance);
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
                     &planning::WheelPlannerConfig::yaw_bin_count)
      .def_readwrite("maximum_terminal_candidates",
                     &planning::WheelPlannerConfig::maximum_terminal_candidates)
      .def_readwrite("continuous_validation_maximum_subdivisions",
                     &planning::WheelPlannerConfig::
                         continuous_validation_maximum_subdivisions);
  py::class_<planning::LeggedPlannerConfig>(module, "LeggedPlannerConfig")
      .def(py::init<>())
      .def_readwrite("xy_resolution_m",
                     &planning::LeggedPlannerConfig::xy_resolution_m)
      .def_readwrite("yaw_bin_count",
                     &planning::LeggedPlannerConfig::yaw_bin_count)
      .def_readwrite(
          "maximum_terminal_candidates",
          &planning::LeggedPlannerConfig::maximum_terminal_candidates)
      .def_readwrite(
          "maximum_height_interval_splits",
          &planning::LeggedPlannerConfig::maximum_height_interval_splits)
      .def_readwrite("continuous_validation_maximum_subdivisions",
                     &planning::LeggedPlannerConfig::
                         continuous_validation_maximum_subdivisions);
  py::class_<planning::HopperPlannerConfig>(module, "HopperPlannerConfig")
      .def(py::init<>())
      .def_readwrite("maximum_landing_regions",
                     &planning::HopperPlannerConfig::maximum_landing_regions)
      .def_readwrite("maximum_graph_nodes",
                     &planning::HopperPlannerConfig::maximum_graph_nodes)
      .def_readwrite("maximum_graph_out_degree",
                     &planning::HopperPlannerConfig::maximum_graph_out_degree)
      .def_readwrite(
          "maximum_nominal_aim_points_per_region",
          &planning::HopperPlannerConfig::maximum_nominal_aim_points_per_region)
      .def_readwrite(
          "maximum_certification_attempts",
          &planning::HopperPlannerConfig::maximum_certification_attempts)
      .def_readwrite(
          "maximum_flight_tube_sections",
          &planning::HopperPlannerConfig::maximum_flight_tube_sections)
      .def_readwrite("maximum_authorized_hops",
                     &planning::HopperPlannerConfig::maximum_authorized_hops);
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
                     &planning::HopSegment::flight_tube_radius_m);
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
}

void BindRequest(py::module_ &module) {
  py::class_<training::TrainingPlanRequest>(module, "TrainingPlanRequest")
      .def(py::init<>())
      .def_readwrite("request_id", &training::TrainingPlanRequest::request_id)
      .def_readwrite("state_time", &training::TrainingPlanRequest::state_time)
      .def_readwrite("current_state",
                     &training::TrainingPlanRequest::current_state)
      .def_readwrite("goal", &training::TrainingPlanRequest::goal)
      .def_readwrite("world", &training::TrainingPlanRequest::world)
      .def_readwrite("capability", &training::TrainingPlanRequest::capability)
      .def_readwrite("config", &training::TrainingPlanRequest::config)
      .def_readwrite("previous_execution",
                     &training::TrainingPlanRequest::previous_execution);
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
          py::arg("request"), py::call_guard<py::gil_scoped_release>());
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
  BindRequest(module);
}
