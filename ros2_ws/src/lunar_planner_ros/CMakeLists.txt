cmake_minimum_required(VERSION 3.13)
project(lunar_planner_ros LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(lunar_navigation_msgs REQUIRED)
find_package(lunar_planner_core REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(urdf REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/capability_loader.cpp
  src/execution_feedback_tracker.cpp
  src/grid_map_adapter.cpp
  src/message_conversion.cpp
  src/plan_motion_server.cpp
  src/provisional_route_marker_publisher.cpp
  src/reference_guard.cpp
  src/route_marker_publisher.cpp
  src/snapshot_builder.cpp
  src/snapshot_store.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  builtin_interfaces
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  lifecycle_msgs
  lunar_navigation_msgs
  lunar_planner_core
  lunar_planning_msgs
  nav_msgs
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  std_msgs
  tf2_msgs
  trajectory_msgs
  urdf
  visualization_msgs)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

add_executable(lunar_planner_node src/main.cpp)
target_compile_features(lunar_planner_node PRIVATE cxx_std_20)
target_link_libraries(lunar_planner_node ${PROJECT_NAME})
ament_target_dependencies(lunar_planner_node rclcpp)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(TARGETS lunar_planner_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ament_index_cpp
  builtin_interfaces
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  lifecycle_msgs
  lunar_navigation_msgs
  lunar_planner_core
  lunar_planning_msgs
  nav_msgs
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  std_msgs
  tf2_msgs
  trajectory_msgs
  urdf
  visualization_msgs
  yaml-cpp)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(action_msgs REQUIRED)

  ament_add_gtest(lunar_planner_ros_grid_map_adapter_test
    test/grid_map_adapter_test.cpp)
  target_include_directories(lunar_planner_ros_grid_map_adapter_test PRIVATE test)
  target_link_libraries(lunar_planner_ros_grid_map_adapter_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_snapshot_builder_test
    test/snapshot_builder_test.cpp)
  target_include_directories(lunar_planner_ros_snapshot_builder_test PRIVATE test)
  target_link_libraries(lunar_planner_ros_snapshot_builder_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_execution_feedback_tracker_test
    test/execution_feedback_tracker_test.cpp)
  target_link_libraries(
    lunar_planner_ros_execution_feedback_tracker_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_route_marker_publisher_test
    test/route_marker_publisher_test.cpp)
  target_link_libraries(
    lunar_planner_ros_route_marker_publisher_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_provisional_route_marker_publisher_test
    test/provisional_route_marker_publisher_test.cpp)
  target_link_libraries(
    lunar_planner_ros_provisional_route_marker_publisher_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_capability_loader_test
    test/capability_loader_test.cpp)
  target_link_libraries(lunar_planner_ros_capability_loader_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_reference_guard_test
    test/reference_guard_test.cpp)
  target_link_libraries(lunar_planner_ros_reference_guard_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_message_conversion_test
    test/message_conversion_test.cpp)
  target_link_libraries(lunar_planner_ros_message_conversion_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_plan_motion_server_test
    test/plan_motion_server_test.cpp)
  target_include_directories(lunar_planner_ros_plan_motion_server_test PRIVATE test)
  target_link_libraries(lunar_planner_ros_plan_motion_server_test ${PROJECT_NAME})
  ament_target_dependencies(
    lunar_planner_ros_plan_motion_server_test action_msgs)
  install(
    TARGETS lunar_planner_ros_plan_motion_server_test
    DESTINATION lib/${PROJECT_NAME})
endif()

ament_package()
