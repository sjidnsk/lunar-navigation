cmake_minimum_required(VERSION 3.13)
project(lunar_incremental_navigation_ros LANGUAGES CXX)

option(LUNAR_INCREMENTAL_NAVIGATION_BUILD_DEBUG
  "Build optional incremental-navigation debug publishers" ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(action_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(lunar_incremental_navigation_core REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

set(LUNAR_INCREMENTAL_NAVIGATION_ROS_SOURCES
  src/elevation_pipeline.cpp
  src/exploration_map_projector.cpp
  src/demo_motion_follower.cpp
  src/incremental_demo_scenario.cpp
  src/incremental_map_publisher.cpp
  src/incremental_navigation_node.cpp
  src/input_store.cpp
  src/map_adapters.cpp
  src/message_conversion.cpp
  src/platform_config.cpp
  src/request_diagnostics.cpp
  src/rviz_goal_bridge.cpp
  src/state_adapter.cpp
  src/traversability_qos.cpp
  src/planning_debug_publisher.cpp
  src/planning_snapshot_visualization.cpp)

add_library(${PROJECT_NAME} SHARED
  ${LUNAR_INCREMENTAL_NAVIGATION_ROS_SOURCES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  action_msgs
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  lunar_incremental_navigation_core
  lunar_planning_msgs
  nav_msgs
  rclcpp
  rclcpp_action
  tf2_msgs
  visualization_msgs)
target_link_libraries(${PROJECT_NAME} yaml-cpp)
if(LUNAR_INCREMENTAL_NAVIGATION_BUILD_DEBUG)
  target_compile_definitions(${PROJECT_NAME} PRIVATE LUNAR_BUILD_DEMO=1)
endif()

add_executable(lunar_incremental_navigation_node
  src/incremental_navigation_main.cpp)
target_compile_features(lunar_incremental_navigation_node PUBLIC cxx_std_20)
target_link_libraries(lunar_incremental_navigation_node ${PROJECT_NAME})
ament_target_dependencies(lunar_incremental_navigation_node rclcpp)

add_executable(lunar_incremental_rviz_goal_bridge
  src/incremental_rviz_goal_bridge_main.cpp)
target_compile_features(lunar_incremental_rviz_goal_bridge PUBLIC cxx_std_20)
target_link_libraries(lunar_incremental_rviz_goal_bridge ${PROJECT_NAME})
ament_target_dependencies(lunar_incremental_rviz_goal_bridge rclcpp)

add_executable(incremental_demo_scenario_node
  src/incremental_demo_scenario_node.cpp)
target_compile_features(incremental_demo_scenario_node PUBLIC cxx_std_20)
target_link_libraries(incremental_demo_scenario_node ${PROJECT_NAME})
ament_target_dependencies(incremental_demo_scenario_node
  geometry_msgs grid_map_msgs lunar_planning_msgs nav_msgs rclcpp tf2_msgs
  visualization_msgs)

add_executable(incremental_demo_motion_emulator_node
  src/incremental_demo_motion_emulator_node.cpp)
target_compile_features(incremental_demo_motion_emulator_node PUBLIC cxx_std_20)
target_link_libraries(incremental_demo_motion_emulator_node ${PROJECT_NAME})
ament_target_dependencies(incremental_demo_motion_emulator_node
  geometry_msgs lunar_planning_msgs nav_msgs rclcpp visualization_msgs)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(
  TARGETS lunar_incremental_navigation_node
    lunar_incremental_rviz_goal_bridge
    incremental_demo_scenario_node
    incremental_demo_motion_emulator_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

get_filename_component(
  LUNAR_INCREMENTAL_NAVIGATION_CONFIG_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../config"
  ABSOLUTE)
get_filename_component(
  LUNAR_INCREMENTAL_NAVIGATION_LAUNCH_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../launch"
  ABSOLUTE)
get_filename_component(
  LUNAR_INCREMENTAL_NAVIGATION_RVIZ_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../rviz"
  ABSOLUTE)
install(
  DIRECTORY "${LUNAR_INCREMENTAL_NAVIGATION_CONFIG_SOURCE_DIR}/"
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING
    PATTERN "wheel.yaml"
    PATTERN "wheel_orin.yaml"
    PATTERN "orin_hardware.yaml"
    PATTERN "orin_controller.yaml"
    PATTERN "legged.yaml"
    PATTERN "incremental_navigation_interfaces.yaml")
install(
  DIRECTORY "${LUNAR_INCREMENTAL_NAVIGATION_LAUNCH_SOURCE_DIR}/"
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING
    PATTERN "incremental_exploration_navigation_rviz.launch.py"
    PATTERN "incremental_rviz_goal_bridge.launch.py"
    PATTERN "navigation_rviz.launch.py")
install(
  DIRECTORY "${LUNAR_INCREMENTAL_NAVIGATION_RVIZ_SOURCE_DIR}/"
  DESTINATION share/${PROJECT_NAME}/rviz
  FILES_MATCHING
    PATTERN "incremental_exploration_navigation.rviz"
    PATTERN "navigation.rviz")

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  function(lunar_add_ros_gtest target source)
    ament_add_gtest(${target} ${source})
    target_link_libraries(${target} ${PROJECT_NAME})
  endfunction()

  lunar_add_ros_gtest(platform_config_test test/platform_config_test.cpp)
  target_compile_definitions(platform_config_test PRIVATE
    PLATFORM_CONFIG_DIR="${LUNAR_INCREMENTAL_NAVIGATION_CONFIG_SOURCE_DIR}")

  lunar_add_ros_gtest(elevation_pipeline_test
    test/elevation_pipeline_test.cpp)
  lunar_add_ros_gtest(exploration_map_projector_test
    test/exploration_map_projector_test.cpp)
  lunar_add_ros_gtest(incremental_map_publisher_test
    test/incremental_map_publisher_test.cpp)
  lunar_add_ros_gtest(incremental_demo_scenario_test
    test/incremental_demo_scenario_test.cpp)
  lunar_add_ros_gtest(demo_motion_follower_test
    test/demo_motion_follower_test.cpp)
  lunar_add_ros_gtest(map_adapters_test test/map_adapters_test.cpp)
  lunar_add_ros_gtest(message_conversion_test
    test/message_conversion_test.cpp)
  lunar_add_ros_gtest(incremental_navigation_node_test
    test/incremental_navigation_node_test.cpp)
  target_compile_definitions(incremental_navigation_node_test PRIVATE
    LUNAR_INCREMENTAL_NAVIGATION_CONFIG_DIR="${LUNAR_INCREMENTAL_NAVIGATION_CONFIG_SOURCE_DIR}")
  if(LUNAR_INCREMENTAL_NAVIGATION_BUILD_DEBUG)
    target_compile_definitions(incremental_navigation_node_test PRIVATE
      LUNAR_BUILD_DEMO=1)
  endif()
  ament_target_dependencies(incremental_navigation_node_test
    action_msgs rclcpp rclcpp_action)
  lunar_add_ros_gtest(planning_debug_publisher_test
    test/planning_debug_publisher_test.cpp)
  lunar_add_ros_gtest(state_adapter_test test/state_adapter_test.cpp)
  lunar_add_ros_gtest(traversability_qos_test
    test/traversability_qos_test.cpp)
  lunar_add_ros_gtest(rviz_goal_bridge_test
    test/rviz_goal_bridge_test.cpp)
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_package()
