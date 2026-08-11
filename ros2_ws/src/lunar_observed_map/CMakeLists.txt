cmake_minimum_required(VERSION 3.16)
project(lunar_observed_map LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/grid_map_products.cpp
  src/observed_map_node.cpp
  src/sparse_observed_map.cpp
  src/obstacle_classifier.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(${PROJECT_NAME}
  builtin_interfaces
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  nav_msgs
  rcl_interfaces
  rclcpp
  rclcpp_lifecycle
  std_msgs)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

add_executable(lunar_observed_map_node src/main.cpp)
target_compile_features(lunar_observed_map_node PRIVATE cxx_std_20)
target_link_libraries(lunar_observed_map_node ${PROJECT_NAME})
ament_target_dependencies(lunar_observed_map_node rclcpp)
install(TARGETS lunar_observed_map_node DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  builtin_interfaces
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  nav_msgs
  rcl_interfaces
  rclcpp
  rclcpp_lifecycle
  std_msgs)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(lunar_planner_ros REQUIRED)
  find_package(OpenSSL REQUIRED)

  ament_add_gtest(sparse_observed_map_test
    test/sparse_observed_map_test.cpp)
  target_link_libraries(sparse_observed_map_test ${PROJECT_NAME})

  ament_add_gtest(obstacle_classifier_test
    test/obstacle_classifier_test.cpp)
  target_link_libraries(obstacle_classifier_test ${PROJECT_NAME})

  ament_add_gtest(grid_map_products_test
    test/grid_map_products_test.cpp)
  target_link_libraries(
    grid_map_products_test ${PROJECT_NAME} lunar_planner_ros::lunar_planner_ros)

  find_package(lifecycle_msgs REQUIRED)
  ament_add_gtest(observed_map_node_test
    test/observed_map_node_test.cpp)
  target_link_libraries(
    observed_map_node_test ${PROJECT_NAME} lunar_planner_ros::lunar_planner_ros)
  ament_target_dependencies(observed_map_node_test lifecycle_msgs)
endif()

ament_package()
