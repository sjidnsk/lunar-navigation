cmake_minimum_required(VERSION 3.13)
project(lunar_pure_planner_core LANGUAGES CXX)

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

# This is the complete installed production closure. Copied facade,
# continuation, execution-feedback and ten-layer admission sources are not
# linked into the pure planner.
set(LUNAR_ANYTIME_PRODUCTION_SOURCES
  src/planner.cpp
  src/hierarchical/frame_transform.cpp
  src/hierarchical/global_route_planner.cpp
  src/hierarchical/reference_composer.cpp
  src/hierarchical/surface_portal_set.cpp
  src/hierarchical/surface_rolling_session.cpp
  src/hierarchical/surface_global_search.cpp
  src/hopper/anytime_hopper_planner.cpp
  src/hopper/ballistic_envelope.cpp
  src/hopper/ballistic_kinematics.cpp
  src/legged/anytime_legged_planner.cpp
  src/shared/active_planner_cache.cpp
  src/shared/anytime_ara_star.cpp
  src/shared/cell_area_distance_transform.cpp
  src/shared/global_occupancy_projection.cpp
  src/shared/goal_distance_field.cpp
  src/shared/local_terrain_projection.cpp
  src/shared/map_snapshot.cpp
  src/shared/obstacle_height_estimator.cpp
  src/shared/planning_timing.cpp
  src/shared/request_local_start_patch.cpp
  src/shared/search_control.cpp
  src/wheel/anytime_wheel_planner.cpp)

add_library(${PROJECT_NAME} SHARED ${LUNAR_ANYTIME_PRODUCTION_SOURCES})
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
install(FILES
  include/lunar_pure_planner_core/planner.hpp
  include/lunar_pure_planner_core/planning_timing.hpp
  include/lunar_pure_planner_core/search_control.hpp
  DESTINATION include/lunar_pure_planner_core)
install(FILES
  include/lunar_pure_planner_core/types/geometry.hpp
  include/lunar_pure_planner_core/types/goal.hpp
  include/lunar_pure_planner_core/types/motion_reference.hpp
  include/lunar_pure_planner_core/types/planning_request.hpp
  include/lunar_pure_planner_core/types/platform_capability.hpp
  include/lunar_pure_planner_core/types/world_snapshot.hpp
  DESTINATION include/lunar_pure_planner_core/types)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  function(lunar_add_core_gtest target source)
    ament_add_gtest(${target} ${source})
    target_include_directories(${target} PRIVATE src test)
    target_link_libraries(${target} ${PROJECT_NAME})
    lunar_enable_sanitizers(${target})
  endfunction()

  lunar_add_core_gtest(lunar_pure_planner_core_public_api_test
    test/public_api_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_dual_mode_planner_test
    test/dual_mode_planner_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_planning_timing_test
    test/planning_timing_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_ara_star_anytime_test
    test/ara_star_anytime_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_map_contract_test
    test/map_contract_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_global_occupancy_projection_test
    test/global_occupancy_projection_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_surface_global_search_test
    test/surface_global_search_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_surface_rolling_session_test
    test/surface_rolling_session_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_surface_portal_set_test
    test/surface_portal_set_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_local_terrain_projection_test
    test/local_terrain_projection_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_obstacle_height_estimator_test
    test/obstacle_height_estimator_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_anytime_wheel_planner_test
    test/anytime_wheel_planner_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_anytime_legged_planner_test
    test/anytime_legged_planner_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_anytime_hopper_planner_test
    test/anytime_hopper_planner_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_ballistic_envelope_test
    test/ballistic_envelope_test.cpp)
  lunar_add_core_gtest(lunar_pure_planner_core_active_planner_cache_test
    test/active_planner_cache_test.cpp)

  add_test(
    NAME lunar_pure_planner_core_no_legacy_contract
    COMMAND Python3::Interpreter
      -m pytest -q -p no:cacheprovider
      "${CMAKE_CURRENT_SOURCE_DIR}/test/no_legacy_contract_test.py")
  set_tests_properties(
    lunar_pure_planner_core_no_legacy_contract PROPERTIES
    LABELS "boundary;native"
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

  add_test(
    NAME lunar_pure_planner_core_public_header_boundary
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/test/public_header_boundary_test.py"
      --include-root "${CMAKE_INSTALL_PREFIX}/include"
      --compiler "${CMAKE_CXX_COMPILER}")
  set_tests_properties(
    lunar_pure_planner_core_public_header_boundary PROPERTIES
    LABELS "boundary;native")

  add_test(
    NAME lunar_pure_planner_core_minimal_map_entrypoint
    COMMAND Python3::Interpreter
      -m pytest -q -p no:cacheprovider
      "${CMAKE_CURRENT_SOURCE_DIR}/test/minimal_map_entrypoint_test.py")
  set_tests_properties(
    lunar_pure_planner_core_minimal_map_entrypoint PROPERTIES
    LABELS "boundary;native"
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endif()

ament_package()
