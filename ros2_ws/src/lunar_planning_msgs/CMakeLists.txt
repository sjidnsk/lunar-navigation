cmake_minimum_required(VERSION 3.22)
project(lunar_planning_msgs)

find_package(ament_cmake REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(unique_identifier_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DemoMapAck.msg"
  "msg/DemoPlanSegment.msg"
  "msg/GoalRegion.msg"
  "msg/HopSegment.msg"
  "msg/MotionReference.msg"
  "msg/PathReference.msg"
  "msg/PlannerDiagnostics.msg"
  "msg/TimedPath.msg"
  "action/PlanMotion.action"
  "action/NavigateToPose.action"
  DEPENDENCIES action_msgs builtin_interfaces geometry_msgs nav_msgs std_msgs trajectory_msgs unique_identifier_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
