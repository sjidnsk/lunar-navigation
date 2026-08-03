cmake_minimum_required(VERSION 3.13)
project(lunar_planner_ros LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(lunar_navigation_msgs REQUIRED)
find_package(lunar_planner_core REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(urdf REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/capability_loader.cpp
  src/grid_map_adapter.cpp
  src/snapshot_builder.cpp
  src/snapshot_store.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  grid_map_msgs
  lunar_navigation_msgs
  lunar_planner_core
  nav_msgs
  rclcpp
  tf2_msgs
  urdf)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ament_index_cpp
  grid_map_msgs
  lunar_navigation_msgs
  lunar_planner_core
  nav_msgs
  rclcpp
  tf2_msgs
  urdf)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(lunar_planner_ros_grid_map_adapter_test
    test/grid_map_adapter_test.cpp)
  target_include_directories(lunar_planner_ros_grid_map_adapter_test PRIVATE test)
  target_link_libraries(lunar_planner_ros_grid_map_adapter_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_snapshot_builder_test
    test/snapshot_builder_test.cpp)
  target_include_directories(lunar_planner_ros_snapshot_builder_test PRIVATE test)
  target_link_libraries(lunar_planner_ros_snapshot_builder_test ${PROJECT_NAME})

  ament_add_gtest(lunar_planner_ros_capability_loader_test
    test/capability_loader_test.cpp)
  target_link_libraries(lunar_planner_ros_capability_loader_test ${PROJECT_NAME})
endif()

ament_package()
