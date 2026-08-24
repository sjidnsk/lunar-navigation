cmake_minimum_required(VERSION 3.22)
project(lunar_pure_exploration_ros LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(lunar_pure_exploration_core REQUIRED)
find_package(lunar_pure_exploration_msgs REQUIRED)
find_package(lunar_pure_planner_core REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/exploration_node.cpp
  src/marker_builder.cpp
  src/planner_client.cpp
  src/planner_timing_accumulator.cpp
  src/platform_config_loader.cpp
  src/pose_resolver.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(${PROJECT_NAME}
  diagnostic_msgs
  geometry_msgs
  lunar_planning_msgs
  lunar_pure_exploration_core
  lunar_pure_exploration_msgs
  lunar_pure_planner_core
  nav_msgs
  rclcpp
  rclcpp_action
  std_msgs
  tf2_msgs
  visualization_msgs)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

add_executable(pure_exploration_node src/main.cpp)
target_compile_features(pure_exploration_node PRIVATE cxx_std_20)
target_link_libraries(pure_exploration_node ${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME} pure_exploration_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)

get_filename_component(
  LUNAR_PURE_EXPLORATION_CONFIG_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../config"
  ABSOLUTE)
get_filename_component(
  LUNAR_PURE_EXPLORATION_LAUNCH_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../launch"
  ABSOLUTE)
install(
  FILES "${LUNAR_PURE_EXPLORATION_CONFIG_SOURCE_DIR}/pure_exploration.yaml"
  DESTINATION share/${PROJECT_NAME}/config)
install(
  FILES "${LUNAR_PURE_EXPLORATION_LAUNCH_SOURCE_DIR}/pure_exploration.launch.py"
  DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(action_msgs REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_index_cpp REQUIRED)
  find_package(lunar_pure_planner_ros REQUIRED)

  # This test-only source digest gate binds the explorer compatibility test to
  # the planner snapshot integrated in lunar_pure_planner_orin. Any later
  # planner-source change requires an explicit compatibility re-review.
  set(_planner_ros_source_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/../lunar_pure_planner_ros/src")
  file(SHA256 "${_planner_ros_source_dir}/message_conversion.cpp"
    _message_conversion_sha)
  file(SHA256 "${_planner_ros_source_dir}/accepted_goal_finalizer.hpp"
    _accepted_goal_finalizer_sha)
  file(SHA256 "${_planner_ros_source_dir}/pure_plan_motion_server.cpp"
    _pure_plan_motion_server_sha)
  if(NOT _message_conversion_sha STREQUAL
      "52633bc5352cd303cc27da65db33aebbc9ad58d56537cfd1e149523a8797e3db" OR
     NOT _accepted_goal_finalizer_sha STREQUAL
      "bcd4dd1be9e56d6495b63171651b8449994d292d6451f3be960fd0fa72235d63" OR
     NOT _pure_plan_motion_server_sha STREQUAL
      "e9c2e67e1522d9d509a7222240a253b23a01547df3583efe1347a311aeb4885e")
    message(FATAL_ERROR
      "final planner source changed; re-review Task 10 compatibility")
  endif()

  ament_add_gtest(test_platform_config_loader
    test/test_platform_config_loader.cpp)
  target_compile_features(test_platform_config_loader PRIVATE cxx_std_20)
  ament_target_dependencies(test_platform_config_loader ament_index_cpp)
  target_link_libraries(test_platform_config_loader ${PROJECT_NAME})

  ament_add_gtest(test_pose_resolver test/test_pose_resolver.cpp)
  target_compile_features(test_pose_resolver PRIVATE cxx_std_20)
  target_link_libraries(test_pose_resolver ${PROJECT_NAME})

  ament_add_gtest(test_planner_timing_accumulator
    test/test_planner_timing_accumulator.cpp)
  target_compile_features(test_planner_timing_accumulator PRIVATE cxx_std_20)
  target_link_libraries(test_planner_timing_accumulator ${PROJECT_NAME})

  ament_add_gtest(test_planner_client test/test_planner_client.cpp)
  target_compile_features(test_planner_client PRIVATE cxx_std_20)
  target_include_directories(test_planner_client PRIVATE
    "${_planner_ros_source_dir}")
  ament_target_dependencies(test_planner_client
    action_msgs lunar_pure_planner_ros rclcpp rclcpp_action)
  target_link_libraries(test_planner_client ${PROJECT_NAME})

  ament_add_gtest(test_exploration_node test/test_exploration_node.cpp)
  target_compile_features(test_exploration_node PRIVATE cxx_std_20)
  ament_target_dependencies(test_exploration_node
    action_msgs lunar_pure_exploration_msgs rclcpp rclcpp_action std_msgs)
  target_link_libraries(test_exploration_node ${PROJECT_NAME})

  ament_add_gtest(test_synthetic_scenarios
    test/test_synthetic_scenarios.cpp)
  target_compile_features(test_synthetic_scenarios PRIVATE cxx_std_20)
  target_compile_definitions(test_synthetic_scenarios PRIVATE
    TASK15_FIXTURE_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/scenarios/fixtures")
  ament_target_dependencies(test_synthetic_scenarios
    action_msgs ament_index_cpp lunar_pure_exploration_msgs rclcpp rclcpp_action std_msgs)
  target_link_libraries(test_synthetic_scenarios ${PROJECT_NAME} yaml-cpp)

  ament_add_gtest(test_exploration_outputs test/test_exploration_outputs.cpp)
  target_compile_features(test_exploration_outputs PRIVATE cxx_std_20)
  target_link_libraries(test_exploration_outputs ${PROJECT_NAME})

  ament_add_gtest(test_execution_monitor test/test_execution_monitor.cpp)
  target_compile_features(test_execution_monitor PRIVATE cxx_std_20)
  target_link_libraries(test_execution_monitor ${PROJECT_NAME})
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  diagnostic_msgs
  geometry_msgs
  lunar_planning_msgs
  lunar_pure_exploration_core
  lunar_pure_exploration_msgs
  lunar_pure_planner_core
  nav_msgs
  rclcpp
  rclcpp_action
  std_msgs
  tf2_msgs
  visualization_msgs
  yaml-cpp)
ament_package()
