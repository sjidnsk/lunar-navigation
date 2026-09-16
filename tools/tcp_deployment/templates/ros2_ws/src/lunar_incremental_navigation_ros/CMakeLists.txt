cmake_minimum_required(VERSION 3.13)
project(lunar_incremental_navigation_ros LANGUAGES CXX)


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
find_package(yaml-cpp REQUIRED)
find_package(visualization_msgs REQUIRED)

set(LUNAR_INCREMENTAL_NAVIGATION_ROS_SOURCES
  src/elevation_pipeline.cpp
  src/exploration_map_projector.cpp
  src/incremental_map_publisher.cpp
  src/incremental_navigation_node.cpp
  src/input_store.cpp
  src/map_adapters.cpp
  src/message_conversion.cpp
  src/platform_config.cpp
  src/request_diagnostics.cpp
  src/state_adapter.cpp
  src/rviz_goal_bridge.cpp
  src/planning_debug_publisher.cpp
  src/planning_snapshot_visualization.cpp
  src/traversability_qos.cpp
)

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
target_compile_definitions(${PROJECT_NAME} PRIVATE LUNAR_BUILD_DEMO=1)

add_executable(lunar_incremental_navigation_node
  src/incremental_navigation_main.cpp)
target_compile_features(lunar_incremental_navigation_node PUBLIC cxx_std_20)
target_link_libraries(lunar_incremental_navigation_node ${PROJECT_NAME})
ament_target_dependencies(lunar_incremental_navigation_node rclcpp)

add_executable(lunar_incremental_rviz_goal_bridge src/incremental_rviz_goal_bridge_main.cpp)
target_compile_features(lunar_incremental_rviz_goal_bridge PUBLIC cxx_std_20)
target_link_libraries(lunar_incremental_rviz_goal_bridge ${PROJECT_NAME})
ament_target_dependencies(lunar_incremental_rviz_goal_bridge rclcpp)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(
  TARGETS lunar_incremental_navigation_node lunar_incremental_rviz_goal_bridge
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

get_filename_component(
  LUNAR_INCREMENTAL_NAVIGATION_CONFIG_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../config"
  ABSOLUTE)
install(
  DIRECTORY "${LUNAR_INCREMENTAL_NAVIGATION_CONFIG_SOURCE_DIR}/"
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING
    PATTERN "wheel.yaml"
    PATTERN "legged.yaml"
    PATTERN "incremental_navigation_interfaces.yaml")
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_package()

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../launch/navigation_rviz.launch.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../launch/incremental_rviz_goal_bridge.launch.py"
  DESTINATION share/${PROJECT_NAME}/launch)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../../rviz/navigation.rviz"
  DESTINATION share/${PROJECT_NAME}/rviz)
