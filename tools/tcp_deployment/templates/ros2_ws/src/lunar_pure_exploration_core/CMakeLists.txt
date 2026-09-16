cmake_minimum_required(VERSION 3.22)
project(lunar_pure_exploration_core LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/boundary_guidance.cpp
  src/candidate_generator.cpp
  src/candidate_ranker.cpp
  src/coverage.cpp
  src/exploration_state_machine.cpp
  src/failure_memory.cpp
  src/frontier_detector.cpp
  src/information_gain.cpp
  src/occupancy_grid.cpp
  src/progress_monitor.cpp
  src/safe_pose_validator.cpp
  src/task_raster.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)



ament_package()
