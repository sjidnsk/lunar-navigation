cmake_minimum_required(VERSION 3.22)
project(lunar_pure_exploration_sim LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(lunar_pure_exploration_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/coordinated_steering_plant.cpp
  src/lunar_scene.cpp
  src/map_messages.cpp
  src/run_coordinator.cpp
  src/simulation_node.cpp
  src/visibility.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(${PROJECT_NAME}
  geometry_msgs
  grid_map_msgs
  lunar_planning_msgs
  lunar_pure_exploration_msgs
  nav_msgs
  rclcpp
  rclcpp_action
  std_msgs
  tf2_msgs
  visualization_msgs)

add_executable(simulation_node src/simulation_main.cpp)
target_compile_features(simulation_node PRIVATE cxx_std_20)
target_link_libraries(simulation_node ${PROJECT_NAME})

add_executable(run_coordinator src/run_coordinator_main.cpp)
target_compile_features(run_coordinator PRIVATE cxx_std_20)
target_link_libraries(run_coordinator ${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(TARGETS simulation_node run_coordinator DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(lunar_scene_test test/lunar_scene_test.cpp)
  target_compile_features(lunar_scene_test PRIVATE cxx_std_20)
  target_link_libraries(lunar_scene_test ${PROJECT_NAME})

  ament_add_gtest(visibility_test test/visibility_test.cpp)
  target_compile_features(visibility_test PRIVATE cxx_std_20)
  target_link_libraries(visibility_test ${PROJECT_NAME})

  ament_add_gtest(map_messages_test test/map_messages_test.cpp)
  target_compile_features(map_messages_test PRIVATE cxx_std_20)
  target_link_libraries(map_messages_test ${PROJECT_NAME})

  ament_add_gtest(coordinated_steering_plant_test
    test/coordinated_steering_plant_test.cpp)
  target_compile_features(coordinated_steering_plant_test PRIVATE cxx_std_20)
  target_link_libraries(coordinated_steering_plant_test ${PROJECT_NAME})

  ament_add_gtest(simulation_node_test test/simulation_node_test.cpp)
  target_compile_features(simulation_node_test PRIVATE cxx_std_20)
  target_link_libraries(simulation_node_test ${PROJECT_NAME})

  ament_add_gtest(run_coordinator_test test/run_coordinator_test.cpp)
  target_compile_features(run_coordinator_test PRIVATE cxx_std_20)
  target_link_libraries(run_coordinator_test ${PROJECT_NAME})
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  geometry_msgs
  grid_map_msgs
  lunar_planning_msgs
  lunar_pure_exploration_msgs
  nav_msgs
  rclcpp
  rclcpp_action
  std_msgs
  tf2_msgs
  visualization_msgs)
ament_package()
