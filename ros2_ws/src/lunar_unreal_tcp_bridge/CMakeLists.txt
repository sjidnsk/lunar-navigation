cmake_minimum_required(VERSION 3.22)
project(lunar_unreal_tcp_bridge LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(lunar_navigation_msgs REQUIRED)
find_package(lunar_planning_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/bridge_node.cpp
  src/coordinate_transform.cpp
  src/metadata_validation.cpp
  src/outbound_queue.cpp
  src/protocol.cpp
  src/ros_conversions.cpp
  src/session_protocol.cpp
  src/stream_decoder.cpp
  src/tcp_client.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  nlohmann_json::nlohmann_json)
ament_target_dependencies(${PROJECT_NAME}
  builtin_interfaces
  diagnostic_msgs
  geometry_msgs
  grid_map_msgs
  lunar_navigation_msgs
  lunar_planning_msgs
  nav_msgs
  rcl_interfaces
  rclcpp
  rclcpp_lifecycle
  rosgraph_msgs
  std_msgs
  std_srvs
  tf2_msgs
  trajectory_msgs)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

add_executable(lunar_unreal_tcp_bridge_node src/main.cpp)
target_link_libraries(lunar_unreal_tcp_bridge_node ${PROJECT_NAME})
ament_target_dependencies(lunar_unreal_tcp_bridge_node rclcpp)
install(TARGETS lunar_unreal_tcp_bridge_node DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  builtin_interfaces
  diagnostic_msgs
  Eigen3
  geometry_msgs
  grid_map_msgs
  lunar_navigation_msgs
  lunar_planning_msgs
  nav_msgs
  nlohmann_json
  rcl_interfaces
  rclcpp
  rclcpp_lifecycle
  rosgraph_msgs
  std_msgs
  std_srvs
  tf2_msgs
  trajectory_msgs)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(unreal_tcp_protocol_test test/protocol_test.cpp)
  target_link_libraries(unreal_tcp_protocol_test ${PROJECT_NAME})
  target_compile_definitions(unreal_tcp_protocol_test PRIVATE
    UNREAL_TCP_VECTOR_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/fixtures/unreal_tcp/protocol_v1_vectors.json")

  ament_add_gtest(unreal_tcp_metadata_validation_test
    test/metadata_validation_test.cpp)
  target_link_libraries(unreal_tcp_metadata_validation_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_stream_decoder_test test/stream_decoder_test.cpp)
  target_link_libraries(unreal_tcp_stream_decoder_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_outbound_queue_test test/outbound_queue_test.cpp)
  target_link_libraries(unreal_tcp_outbound_queue_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_session_protocol_test test/session_protocol_test.cpp)
  target_link_libraries(unreal_tcp_session_protocol_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_coordinate_transform_test
    test/coordinate_transform_test.cpp)
  target_link_libraries(unreal_tcp_coordinate_transform_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_ros_conversions_test
    test/ros_conversions_test.cpp)
  target_link_libraries(unreal_tcp_ros_conversions_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_client_test test/tcp_client_test.cpp)
  target_link_libraries(unreal_tcp_client_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_bridge_node_test test/bridge_node_test.cpp)
  target_link_libraries(unreal_tcp_bridge_node_test ${PROJECT_NAME})
  ament_target_dependencies(unreal_tcp_bridge_node_test lifecycle_msgs)
endif()

ament_package()
