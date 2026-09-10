cmake_minimum_required(VERSION 3.13)
project(lunar_incremental_navigation_core LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

option(LUNAR_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)

function(lunar_enable_sanitizers target_name)
  if(NOT LUNAR_ENABLE_SANITIZERS)
    return()
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "LUNAR_ENABLE_SANITIZERS requires GCC or Clang")
  endif()
  target_compile_options(${target_name} PRIVATE
    -fsanitize=address,undefined
    -fno-omit-frame-pointer)
  target_link_options(${target_name} PRIVATE
    -fsanitize=address,undefined)
endfunction()

set(LUNAR_INCREMENTAL_NAVIGATION_PRODUCTION_SOURCES
  src/map/global_guidance_builder.cpp
  src/map/persistent_elevation_map.cpp
  src/map/platform_elevation_physics.cpp
  src/map/fine_traversability_builder.cpp
  src/global/global_route_planner.cpp
  src/local/local_planning_window.cpp
  src/local/local_target_selector.cpp
  src/local/local_goal_region.cpp
  src/local/request_local_start_patch.cpp
  src/local/phase_aware_path_simplifier.cpp
  src/session/planning_session_coordinator.cpp
  src/legged/legged_local_planner.cpp
  src/shared/search_control.cpp
  src/wheel/wheel_local_astar.cpp)

add_library(${PROJECT_NAME} SHARED
  ${LUNAR_INCREMENTAL_NAVIGATION_PRODUCTION_SOURCES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
lunar_enable_sanitizers(${PROJECT_NAME})

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

  add_library(${PROJECT_NAME}_test_support STATIC
    test/support/complex_terrain_benchmark.cpp
    test/support/complex_terrain_scenarios.cpp)
  target_compile_features(${PROJECT_NAME}_test_support PUBLIC cxx_std_20)
  target_include_directories(${PROJECT_NAME}_test_support
    PUBLIC test
    PRIVATE src)
  target_link_libraries(${PROJECT_NAME}_test_support PUBLIC ${PROJECT_NAME})
  lunar_enable_sanitizers(${PROJECT_NAME}_test_support)

  function(lunar_add_core_gtest target source)
    ament_add_gtest(${target} ${source})
    target_include_directories(${target} PRIVATE src test)
    target_link_libraries(${target} ${PROJECT_NAME})
    lunar_enable_sanitizers(${target})
  endfunction()

  lunar_add_core_gtest(
    lunar_incremental_navigation_core_planning_contract_test
    test/planning_contract_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_persistent_elevation_map_test
    test/persistent_elevation_map_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_planning_snapshot_contract_test
    test/planning_snapshot_contract_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_fine_traversability_builder_test
    test/fine_traversability_builder_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_request_local_start_patch_test
    test/request_local_start_patch_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_global_guidance_builder_test
    test/global_guidance_builder_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_global_route_planner_v2_test
    test/global_route_planner_v2_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_reachable_local_goals_test
    test/reachable_local_goals_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_local_target_selector_test
    test/local_target_selector_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_local_planning_window_test
    test/local_planning_window_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_wheel_local_planner_test
    test/wheel_local_planner_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_phase_aware_path_simplifier_test
    test/phase_aware_path_simplifier_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_legged_local_planner_v2_test
    test/legged_local_planner_v2_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_planning_session_coordinator_test
    test/planning_session_coordinator_test.cpp)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_complex_terrain_scenarios_test
    test/complex_terrain_scenarios_test.cpp)
  target_link_libraries(
    lunar_incremental_navigation_core_complex_terrain_scenarios_test
    ${PROJECT_NAME}_test_support)
  lunar_add_core_gtest(
    lunar_incremental_navigation_core_complex_terrain_benchmark_test
    test/complex_terrain_benchmark_test.cpp)
  target_link_libraries(
    lunar_incremental_navigation_core_complex_terrain_benchmark_test
    ${PROJECT_NAME}_test_support)

  add_executable(${PROJECT_NAME}_complex_terrain_benchmark
    benchmark/complex_terrain_benchmark_main.cpp)
  target_compile_features(
    ${PROJECT_NAME}_complex_terrain_benchmark PRIVATE cxx_std_20)
  target_include_directories(
    ${PROJECT_NAME}_complex_terrain_benchmark PRIVATE test)
  target_link_libraries(
    ${PROJECT_NAME}_complex_terrain_benchmark
    ${PROJECT_NAME}_test_support)
  lunar_enable_sanitizers(${PROJECT_NAME}_complex_terrain_benchmark)
endif()

ament_package()
