cmake_minimum_required(VERSION 3.22)
project(lunar_pure_exploration_ros LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(lunar_pure_exploration_core REQUIRED)
find_package(lunar_pure_exploration_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/incremental_exploration_node.cpp
  src/marker_builder.cpp
  src/navigation_client.cpp
  src/platform_config_loader.cpp
  src/pose_resolver.cpp
  src/task_map_marker_builder.cpp)
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
  nav_msgs
  rclcpp
  rclcpp_action
  tf2_msgs
  visualization_msgs)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

add_executable(incremental_exploration_node
  src/incremental_exploration_main.cpp)
target_compile_features(incremental_exploration_node PRIVATE cxx_std_20)
target_link_libraries(incremental_exploration_node ${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME} incremental_exploration_node
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
  FILES "${LUNAR_PURE_EXPLORATION_CONFIG_SOURCE_DIR}/exploration_navigation.yaml"
  DESTINATION share/${PROJECT_NAME}/config)
install(
  FILES "${LUNAR_PURE_EXPLORATION_LAUNCH_SOURCE_DIR}/exploration_navigation.launch.py"
  DESTINATION share/${PROJECT_NAME}/launch)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  diagnostic_msgs
  geometry_msgs
  lunar_planning_msgs
  lunar_pure_exploration_core
  lunar_pure_exploration_msgs
  nav_msgs
  rclcpp
  rclcpp_action
  tf2_msgs
  visualization_msgs
  yaml-cpp)
ament_package()
