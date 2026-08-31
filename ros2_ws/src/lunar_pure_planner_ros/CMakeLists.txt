cmake_minimum_required(VERSION 3.13)
project(lunar_pure_planner_ros LANGUAGES CXX)

option(LUNAR_BUILD_DEMO "Build test-only lunar surface RViz demo" ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(lunar_pure_planner_core REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
if(LUNAR_BUILD_DEMO)
  find_package(visualization_msgs REQUIRED)
endif()

set(LUNAR_PURE_PLANNER_ROS_SOURCES
  src/platform_config.cpp
  src/input_store.cpp
  src/traversability_input.cpp
  src/map_adapters.cpp
  src/incremental_traversability.cpp
  src/elevation_occupancy.cpp
  src/center_distance_transform.cpp
  src/traversability_qos.cpp
  src/trusted_bridge.cpp
  src/state_adapter.cpp
  src/message_conversion.cpp
  src/request_diagnostics.cpp
  src/pure_plan_motion_server.cpp
  src/rviz_goal_bridge.cpp)
if(LUNAR_BUILD_DEMO)
  list(APPEND LUNAR_PURE_PLANNER_ROS_SOURCES
    src/lunar_surface_demo_state.cpp
    src/lunar_surface_demo_motion.cpp
    src/lunar_surface_scenario.cpp
    src/lunar_surface_local_map.cpp
    src/lunar_surface_reporter.cpp
    src/lunar_surface_traversability_viz.cpp)
endif()
add_library(${PROJECT_NAME} SHARED ${LUNAR_PURE_PLANNER_ROS_SOURCES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../lunar_pure_planner_core/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../lunar_pure_planner_core/src")
ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  lunar_planning_msgs
  lunar_pure_planner_core
  nav_msgs
  rclcpp
  rclcpp_action
  tf2_msgs)
if(LUNAR_BUILD_DEMO)
  ament_target_dependencies(${PROJECT_NAME} visualization_msgs)
endif()
target_link_libraries(${PROJECT_NAME} yaml-cpp)

add_executable(lunar_pure_planner_node src/main.cpp)
target_compile_features(lunar_pure_planner_node PUBLIC cxx_std_20)
target_link_libraries(lunar_pure_planner_node ${PROJECT_NAME})
ament_target_dependencies(lunar_pure_planner_node rclcpp)

add_executable(lunar_rviz_goal_bridge src/rviz_goal_bridge_main.cpp)
target_compile_features(lunar_rviz_goal_bridge PUBLIC cxx_std_20)
target_link_libraries(lunar_rviz_goal_bridge ${PROJECT_NAME})
ament_target_dependencies(lunar_rviz_goal_bridge rclcpp)

add_executable(lunar_local_traversability_node
  src/local_traversability_main.cpp
  src/local_traversability_node.cpp)
target_compile_features(lunar_local_traversability_node PUBLIC cxx_std_20)
target_link_libraries(lunar_local_traversability_node ${PROJECT_NAME})
ament_target_dependencies(lunar_local_traversability_node
  ament_index_cpp grid_map_msgs rclcpp)

add_executable(lunar_elevation_occupancy_node
  src/elevation_occupancy_main.cpp
  src/elevation_occupancy_node.cpp)
target_compile_features(lunar_elevation_occupancy_node PUBLIC cxx_std_20)
target_link_libraries(lunar_elevation_occupancy_node ${PROJECT_NAME})
ament_target_dependencies(lunar_elevation_occupancy_node grid_map_msgs rclcpp)

if(LUNAR_BUILD_DEMO)
  add_executable(lunar_surface_demo_node src/lunar_surface_demo_node.cpp)
  target_compile_features(lunar_surface_demo_node PUBLIC cxx_std_20)
  target_link_libraries(lunar_surface_demo_node ${PROJECT_NAME})
  ament_target_dependencies(lunar_surface_demo_node
    geometry_msgs grid_map_msgs nav_msgs rclcpp tf2_msgs)

  add_executable(lunar_surface_visualizer_node
    src/lunar_surface_visualizer_node.cpp)
  target_compile_features(lunar_surface_visualizer_node PUBLIC cxx_std_20)
  target_link_libraries(lunar_surface_visualizer_node ${PROJECT_NAME})
  ament_target_dependencies(lunar_surface_visualizer_node
    grid_map_msgs lunar_planning_msgs nav_msgs rclcpp visualization_msgs)

  add_executable(lunar_surface_reporter_node
    src/lunar_surface_reporter_main.cpp)
  target_compile_features(lunar_surface_reporter_node PUBLIC cxx_std_20)
  target_link_libraries(lunar_surface_reporter_node ${PROJECT_NAME})
  ament_target_dependencies(lunar_surface_reporter_node rclcpp)
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(
  TARGETS lunar_pure_planner_node lunar_rviz_goal_bridge
    lunar_local_traversability_node lunar_elevation_occupancy_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
if(LUNAR_BUILD_DEMO)
  install(
    TARGETS lunar_surface_demo_node lunar_surface_visualizer_node
      lunar_surface_reporter_node
    RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()
install(DIRECTORY include/ DESTINATION include)

get_filename_component(
  LUNAR_PURE_PLANNER_CONFIG_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../config"
  ABSOLUTE)
get_filename_component(
  LUNAR_PURE_PLANNER_LAUNCH_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../launch"
  ABSOLUTE)
install(
  DIRECTORY "${LUNAR_PURE_PLANNER_CONFIG_SOURCE_DIR}/"
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING
    PATTERN "*.yaml"
    PATTERN "pure_exploration.yaml" EXCLUDE)
install(
  DIRECTORY "${LUNAR_PURE_PLANNER_LAUNCH_SOURCE_DIR}/"
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING
    PATTERN "*.launch.py"
    PATTERN "pure_exploration.launch.py" EXCLUDE)
if(LUNAR_BUILD_DEMO)
  get_filename_component(
    LUNAR_PURE_PLANNER_RVIZ_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../rviz"
    ABSOLUTE)
  install(
    DIRECTORY "${LUNAR_PURE_PLANNER_RVIZ_SOURCE_DIR}/"
    DESTINATION share/${PROJECT_NAME}/rviz
    FILES_MATCHING PATTERN "*.rviz")
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(action_msgs REQUIRED)
  get_filename_component(
    LUNAR_PURE_PLANNER_CONFIG_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../config"
    ABSOLUTE)
  ament_add_gtest(platform_config_test test/platform_config_test.cpp)
  target_compile_definitions(platform_config_test PRIVATE
    LUNAR_PURE_PLANNER_CONFIG_DIR="${LUNAR_PURE_PLANNER_CONFIG_DIR}")
  target_link_libraries(platform_config_test ${PROJECT_NAME})

  ament_add_gtest(input_store_test test/input_store_test.cpp)
  target_link_libraries(input_store_test ${PROJECT_NAME})

  ament_add_gtest(map_adapters_test test/map_adapters_test.cpp)
  target_include_directories(map_adapters_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../lunar_pure_planner_core/src")
  target_link_libraries(map_adapters_test ${PROJECT_NAME})

  ament_add_gtest(state_adapter_test test/state_adapter_test.cpp)
  target_link_libraries(state_adapter_test ${PROJECT_NAME})

  ament_add_gtest(message_conversion_test test/message_conversion_test.cpp)
  target_link_libraries(message_conversion_test ${PROJECT_NAME})

  ament_add_gtest(request_diagnostics_test test/request_diagnostics_test.cpp)
  target_link_libraries(request_diagnostics_test ${PROJECT_NAME})

  ament_add_gtest(traversability_input_test test/traversability_input_test.cpp)
  target_link_libraries(traversability_input_test ${PROJECT_NAME})

  ament_add_gtest(pure_plan_motion_server_test
    test/pure_plan_motion_server_test.cpp)
  target_compile_definitions(pure_plan_motion_server_test PRIVATE
    LUNAR_PURE_PLANNER_CONFIG_DIR="${LUNAR_PURE_PLANNER_CONFIG_DIR}")
  target_include_directories(pure_plan_motion_server_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(pure_plan_motion_server_test ${PROJECT_NAME})
  ament_target_dependencies(pure_plan_motion_server_test
    action_msgs rclcpp rclcpp_action)

  ament_add_gtest(rviz_goal_bridge_test test/rviz_goal_bridge_test.cpp)
  target_link_libraries(rviz_goal_bridge_test ${PROJECT_NAME})
  ament_target_dependencies(rviz_goal_bridge_test
    action_msgs rclcpp rclcpp_action)

  ament_add_gtest(incremental_traversability_test
    test/incremental_traversability_test.cpp)
  target_link_libraries(incremental_traversability_test ${PROJECT_NAME})

  ament_add_gtest(elevation_occupancy_test test/elevation_occupancy_test.cpp)
  target_link_libraries(elevation_occupancy_test ${PROJECT_NAME})

  ament_add_gtest(traversability_qos_test test/traversability_qos_test.cpp)
  target_link_libraries(traversability_qos_test ${PROJECT_NAME})
  ament_target_dependencies(traversability_qos_test rclcpp)

  ament_add_gtest(center_distance_transform_test
    test/center_distance_transform_test.cpp)
  target_link_libraries(center_distance_transform_test ${PROJECT_NAME})

  ament_add_gtest(trusted_bridge_test test/trusted_bridge_test.cpp)
  target_link_libraries(trusted_bridge_test ${PROJECT_NAME})

  if(LUNAR_BUILD_DEMO)
    ament_add_gtest(lunar_surface_scenario_test
      test/lunar_surface_scenario_test.cpp)
    target_link_libraries(lunar_surface_scenario_test ${PROJECT_NAME})

    ament_add_gtest(lunar_surface_demo_state_test
      test/lunar_surface_demo_state_test.cpp)
    target_link_libraries(lunar_surface_demo_state_test ${PROJECT_NAME})

    ament_add_gtest(lunar_surface_reporter_test
      test/lunar_surface_reporter_test.cpp)
    target_link_libraries(lunar_surface_reporter_test ${PROJECT_NAME})

    ament_add_gtest(lunar_surface_traversability_viz_test
      test/lunar_surface_traversability_viz_test.cpp)
    target_link_libraries(lunar_surface_traversability_viz_test ${PROJECT_NAME})
  endif()

  get_filename_component(
    LUNAR_PURE_PLANNER_LAUNCH_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/launch/test_pure_plan_motion_server.py"
    ABSOLUTE)
  ament_add_pytest_test(pure_plan_motion_server_launch_test
    "${LUNAR_PURE_PLANNER_LAUNCH_TEST}"
    TIMEOUT 120)
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ament_index_cpp
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  lunar_planning_msgs
  lunar_pure_planner_core
  nav_msgs
  rclcpp
  rclcpp_action
  tf2_msgs
  yaml-cpp)
if(LUNAR_BUILD_DEMO)
  ament_export_dependencies(visualization_msgs)
endif()
ament_package()
