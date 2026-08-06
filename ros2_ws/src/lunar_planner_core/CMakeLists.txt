cmake_minimum_required(VERSION 3.13)
project(lunar_planner_core LANGUAGES CXX)

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

add_library(${PROJECT_NAME} SHARED
  src/planner.cpp
  src/hierarchical/frame_transform.cpp
  src/hierarchical/global_route.cpp
  src/hierarchical/global_route_planner.cpp
  src/hierarchical/grid_search.cpp
  src/hierarchical/hopper_route_planner.cpp
  src/hierarchical/landing_support_field.cpp
  src/hierarchical/local_frontier.cpp
  src/hierarchical/map_level.cpp
  src/hierarchical/reference_composer.cpp
  src/hopper/ballistic_envelope.cpp
  src/hopper/ballistic_kinematics.cpp
  src/hopper/commitment_state_machine.cpp
  src/hopper/flight_tube_certifier.cpp
  src/hopper/hop_certifier.cpp
  src/hopper/hopper_planner.cpp
  src/hopper/landing_region.cpp
  src/legged/legged_lattice.cpp
  src/legged/legged_planner.cpp
  src/legged/legged_spline_optimizer.cpp
  src/legged/legged_terrain.cpp
  src/legged/legged_timing.cpp
  src/shared/ara_star.cpp
  src/shared/bounded_qp_solver.cpp
  src/shared/candidate_ranker.cpp
  src/shared/convex_corridor.cpp
  src/shared/map_snapshot.cpp
  src/shared/safe_projection.cpp
  src/shared/terrain_checks.cpp
  src/wheel/wheel_lattice.cpp
  src/wheel/wheel_planner.cpp
  src/wheel/wheel_spline_optimizer.cpp
  src/wheel/wheel_sweep_validator.cpp
  src/wheel/wheel_timing.cpp)
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
  find_package(nlohmann_json REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  ament_add_gtest(lunar_planner_core_public_api_test
    test/public_api_test.cpp)
  target_include_directories(lunar_planner_core_public_api_test PRIVATE test)
  target_link_libraries(lunar_planner_core_public_api_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_public_api_test)

  ament_add_gtest(lunar_planner_core_shared_core_test
    test/shared_core_test.cpp)
  target_include_directories(lunar_planner_core_shared_core_test PRIVATE
    src test)
  target_link_libraries(lunar_planner_core_shared_core_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_shared_core_test)

  ament_add_gtest(lunar_planner_core_shared_core_determinism_test
    test/shared_core_determinism_test.cpp)
  target_include_directories(
    lunar_planner_core_shared_core_determinism_test PRIVATE src test)
  target_link_libraries(
    lunar_planner_core_shared_core_determinism_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_shared_core_determinism_test)

  ament_add_gtest(lunar_planner_core_hierarchical_types_test
    test/hierarchical_types_test.cpp)
  target_include_directories(lunar_planner_core_hierarchical_types_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_hierarchical_types_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_hierarchical_types_test)

  ament_add_gtest(lunar_planner_core_hierarchical_planner_test
    test/hierarchical_planner_test.cpp)
  target_include_directories(
    lunar_planner_core_hierarchical_planner_test PRIVATE src test)
  target_link_libraries(
    lunar_planner_core_hierarchical_planner_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_hierarchical_planner_test)

  ament_add_gtest(lunar_planner_core_hierarchical_regression_test
    test/hierarchical_regression_test.cpp)
  target_include_directories(
    lunar_planner_core_hierarchical_regression_test PRIVATE src test)
  target_link_libraries(
    lunar_planner_core_hierarchical_regression_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_hierarchical_regression_test)

  ament_add_gtest(lunar_planner_core_global_grid_search_test
    test/global_grid_search_test.cpp)
  target_include_directories(lunar_planner_core_global_grid_search_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_global_grid_search_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_global_grid_search_test)

  ament_add_gtest(lunar_planner_core_global_route_planner_test
    test/global_route_planner_test.cpp)
  target_include_directories(lunar_planner_core_global_route_planner_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_global_route_planner_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_global_route_planner_test)

  ament_add_gtest(lunar_planner_core_local_frontier_test
    test/local_frontier_test.cpp)
  target_include_directories(lunar_planner_core_local_frontier_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_local_frontier_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_local_frontier_test)

  ament_add_gtest(lunar_planner_core_wheel_planner_test
    test/wheel_planner_test.cpp)
  target_include_directories(lunar_planner_core_wheel_planner_test PRIVATE
    src test)
  target_link_libraries(lunar_planner_core_wheel_planner_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_wheel_planner_test)

  ament_add_gtest(lunar_planner_core_wheel_fault_matrix_test
    test/wheel_fault_matrix_test.cpp)
  target_include_directories(lunar_planner_core_wheel_fault_matrix_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_wheel_fault_matrix_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_wheel_fault_matrix_test)

  ament_add_gtest(lunar_planner_core_legged_planner_test
    test/legged_planner_test.cpp)
  target_include_directories(lunar_planner_core_legged_planner_test PRIVATE
    src test)
  target_link_libraries(lunar_planner_core_legged_planner_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_legged_planner_test)

  ament_add_gtest(lunar_planner_core_legged_fault_matrix_test
    test/legged_fault_matrix_test.cpp)
  target_include_directories(lunar_planner_core_legged_fault_matrix_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_legged_fault_matrix_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_legged_fault_matrix_test)

  ament_add_gtest(lunar_planner_core_hopper_planner_test
    test/hopper_planner_test.cpp)
  target_include_directories(lunar_planner_core_hopper_planner_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_hopper_planner_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_hopper_planner_test)

  ament_add_gtest(lunar_planner_core_ballistic_envelope_test
    test/ballistic_envelope_test.cpp)
  target_include_directories(
    lunar_planner_core_ballistic_envelope_test PRIVATE src test)
  target_link_libraries(
    lunar_planner_core_ballistic_envelope_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_ballistic_envelope_test)

  ament_add_gtest(lunar_planner_core_hopper_route_planner_test
    test/hopper_route_planner_test.cpp)
  target_include_directories(
    lunar_planner_core_hopper_route_planner_test PRIVATE src test)
  target_link_libraries(
    lunar_planner_core_hopper_route_planner_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_hopper_route_planner_test)

  ament_add_gtest(lunar_planner_core_hopper_commitment_test
    test/hopper_commitment_test.cpp)
  target_include_directories(lunar_planner_core_hopper_commitment_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_hopper_commitment_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_hopper_commitment_test)

  ament_add_gtest(lunar_planner_core_hopper_fault_matrix_test
    test/hopper_fault_matrix_test.cpp)
  target_include_directories(lunar_planner_core_hopper_fault_matrix_test PRIVATE
    src test)
  target_link_libraries(
    lunar_planner_core_hopper_fault_matrix_test ${PROJECT_NAME})
  lunar_enable_sanitizers(lunar_planner_core_hopper_fault_matrix_test)

  add_executable(lunar_planner_facade_summary
    test/facade_summary.cpp)
  target_compile_features(lunar_planner_facade_summary PRIVATE cxx_std_20)
  target_include_directories(lunar_planner_facade_summary PRIVATE src test)
  target_link_libraries(lunar_planner_facade_summary PRIVATE
    ${PROJECT_NAME}
    nlohmann_json::nlohmann_json)
  lunar_enable_sanitizers(lunar_planner_facade_summary)

  add_executable(lunar_planner_core_benchmark
    ../../../tests/performance/planner_core_benchmark.cpp)
  target_compile_features(lunar_planner_core_benchmark PRIVATE cxx_std_20)
  target_include_directories(lunar_planner_core_benchmark PRIVATE test)
  target_link_libraries(lunar_planner_core_benchmark PRIVATE
    ${PROJECT_NAME}
    nlohmann_json::nlohmann_json)
  lunar_enable_sanitizers(lunar_planner_core_benchmark)

  add_executable(lunar_hierarchical_planner_benchmark
    ../../../tests/performance/hierarchical_planner_benchmark.cpp)
  target_compile_features(
    lunar_hierarchical_planner_benchmark PRIVATE cxx_std_20)
  target_include_directories(
    lunar_hierarchical_planner_benchmark PRIVATE src test)
  target_compile_definitions(
    lunar_hierarchical_planner_benchmark PRIVATE
      LUNAR_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\")
  target_link_libraries(lunar_hierarchical_planner_benchmark PRIVATE
    ${PROJECT_NAME}
    nlohmann_json::nlohmann_json)
  lunar_enable_sanitizers(lunar_hierarchical_planner_benchmark)

  add_test(
    NAME lunar_planner_core_v3_semantic_equivalence
    COMMAND "${CMAKE_COMMAND}" -E env
      "LUNAR_PLANNER_FACADE_SUMMARY=$<TARGET_FILE:lunar_planner_facade_summary>"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/differential/test_v3_semantic_equivalence.py")
  set_tests_properties(
    lunar_planner_core_v3_semantic_equivalence PROPERTIES
    LABELS "differential;native")

  add_test(
    NAME lunar_planner_core_benchmark_contract
    COMMAND "${CMAKE_COMMAND}" -E env
      "LUNAR_PLANNER_CORE_BENCHMARK=$<TARGET_FILE:lunar_planner_core_benchmark>"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/performance/test_planner_core_benchmark.py")
  set_tests_properties(
    lunar_planner_core_benchmark_contract PROPERTIES
    LABELS "performance;native")

  add_test(
    NAME lunar_planner_core_hierarchical_benchmark_contract
    COMMAND "${CMAKE_COMMAND}" -E env
      "LUNAR_HIERARCHICAL_PLANNER_BENCHMARK=$<TARGET_FILE:lunar_hierarchical_planner_benchmark>"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/performance/test_hierarchical_planner_benchmark.py")
  set_tests_properties(
    lunar_planner_core_hierarchical_benchmark_contract PROPERTIES
    LABELS "performance;native"
    TIMEOUT 900)

  add_test(
    NAME lunar_planner_core_no_legacy_contract
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/test/no_legacy_contract_test.py")
  set_tests_properties(
    lunar_planner_core_no_legacy_contract PROPERTIES
    LABELS "boundary;native")

  add_test(
    NAME lunar_planner_core_public_header_boundary
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/test/public_header_boundary_test.py"
      --include-root "${CMAKE_INSTALL_PREFIX}/include"
      --compiler "${CMAKE_CXX_COMPILER}")
endif()

ament_package()
