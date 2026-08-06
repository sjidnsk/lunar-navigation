cmake_minimum_required(VERSION 3.22)
project(lunar_navigation_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LocalizationStatus.msg"
  "msg/ScienceTargetRegion.msg"
  "msg/ExplorationTask.msg"
  "msg/MotionExecutionFeedback.msg"
  "msg/HopperPropellantState.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
