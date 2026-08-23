cmake_minimum_required(VERSION 3.22)
project(lunar_pure_exploration_core LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/candidate_generator.cpp
  src/candidate_ranker.cpp
  src/coverage.cpp
  src/exploration_state_machine.cpp
  src/failure_memory.cpp
  src/frontier_detector.cpp
  src/information_gain.cpp
  src/occupancy_grid.cpp
  src/progress_monitor.cpp
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

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_task_raster test/test_task_raster.cpp)
  target_compile_features(test_task_raster PRIVATE cxx_std_20)
  target_link_libraries(test_task_raster ${PROJECT_NAME})

  ament_add_gtest(test_frontier_detector test/test_frontier_detector.cpp)
  target_compile_features(test_frontier_detector PRIVATE cxx_std_20)
  target_link_libraries(test_frontier_detector ${PROJECT_NAME})

  ament_add_gtest(test_occupancy_grid test/test_occupancy_grid.cpp)
  target_compile_features(test_occupancy_grid PRIVATE cxx_std_20)
  target_link_libraries(test_occupancy_grid ${PROJECT_NAME})

  ament_add_gtest(test_candidate_generator test/test_candidate_generator.cpp)
  target_compile_features(test_candidate_generator PRIVATE cxx_std_20)
  target_link_libraries(test_candidate_generator ${PROJECT_NAME})

  ament_add_gtest(test_exploration_state_machine test/test_exploration_state_machine.cpp)
  target_compile_features(test_exploration_state_machine PRIVATE cxx_std_20)
  target_link_libraries(test_exploration_state_machine ${PROJECT_NAME})

  ament_add_gtest(test_information_gain test/test_information_gain.cpp)
  target_compile_features(test_information_gain PRIVATE cxx_std_20)
  target_link_libraries(test_information_gain ${PROJECT_NAME})

  ament_add_gtest(test_progress_monitor test/test_progress_monitor.cpp)
  target_compile_features(test_progress_monitor PRIVATE cxx_std_20)
  target_link_libraries(test_progress_monitor ${PROJECT_NAME})

  ament_add_gtest(test_visibility_traversal test/test_visibility_traversal.cpp)
  target_compile_features(test_visibility_traversal PRIVATE cxx_std_20)
  target_include_directories(test_visibility_traversal PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(test_visibility_traversal ${PROJECT_NAME})

  ament_add_gtest(test_candidate_ranker test/test_candidate_ranker.cpp)
  target_compile_features(test_candidate_ranker PRIVATE cxx_std_20)
  target_link_libraries(test_candidate_ranker ${PROJECT_NAME})

  ament_add_gtest(test_failure_memory test/test_failure_memory.cpp)
  target_compile_features(test_failure_memory PRIVATE cxx_std_20)
  target_link_libraries(test_failure_memory ${PROJECT_NAME})

  ament_add_gtest(test_coverage test/test_coverage.cpp)
  target_compile_features(test_coverage PRIVATE cxx_std_20)
  target_include_directories(test_coverage PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(test_coverage ${PROJECT_NAME})
endif()

ament_package()
