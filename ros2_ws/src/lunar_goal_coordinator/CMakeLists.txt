cmake_minimum_required(VERSION 3.16)
project(lunar_goal_coordinator LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(lunar_navigation_msgs REQUIRED)
find_package(lunar_planner_core REQUIRED)
find_package(lunar_planner_ros REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/goal_state_machine.cpp
  src/goal_coordinator_node.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(
  ${PROJECT_NAME}
  lunar_planner_core::lunar_planner_core
  lunar_planner_ros::lunar_planner_ros)
ament_target_dependencies(${PROJECT_NAME}
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  lunar_navigation_msgs
  lunar_planning_msgs
  nav_msgs
  rcl_interfaces
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  std_srvs
  tf2_msgs)

add_executable(lunar_goal_coordinator_node src/main.cpp)
target_compile_features(lunar_goal_coordinator_node PRIVATE cxx_std_20)
target_link_libraries(lunar_goal_coordinator_node ${PROJECT_NAME})
ament_target_dependencies(lunar_goal_coordinator_node rclcpp)

install(
  TARGETS ${PROJECT_NAME} lunar_goal_coordinator_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(lunar_planner_core lunar_planner_ros)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(lifecycle_msgs REQUIRED)

  ament_add_gtest(goal_state_machine_test test/goal_state_machine_test.cpp)
  target_link_libraries(goal_state_machine_test ${PROJECT_NAME})

  ament_add_gtest(goal_coordinator_node_test test/goal_coordinator_node_test.cpp)
  target_link_libraries(goal_coordinator_node_test ${PROJECT_NAME})
  ament_target_dependencies(goal_coordinator_node_test lifecycle_msgs)
endif()

ament_package()
