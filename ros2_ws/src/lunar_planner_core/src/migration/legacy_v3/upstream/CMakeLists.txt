cmake_minimum_required(VERSION 3.28)
project(lunar_path_planner_v3 VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(LPP_V3_BUILD_TESTS "Build C++ tests" ON)
option(LPP_V3_BUILD_BENCHMARKS "Build C++ benchmarks" ON)
option(LPP_V3_ENABLE_OSQP "Build the optional OSQP backend" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 5.0.1 EXACT CONFIG REQUIRED)
find_package(nlohmann_json 3.12.0 EXACT CONFIG REQUIRED)
find_package(double-conversion 3.4.0 EXACT CONFIG REQUIRED)
find_path(LPP_V3_PICOSHA2_INCLUDE_DIR NAMES picosha2.h REQUIRED)

add_library(lpp_v3_project_options INTERFACE)
target_compile_features(lpp_v3_project_options INTERFACE cxx_std_20)
add_library(lpp_v3_project_warnings INTERFACE)

include(cmake/ProjectOptions.cmake)
include(cmake/CompilerWarnings.cmake)
lpp_v3_set_project_options(lpp_v3_project_options)
lpp_v3_set_project_warnings(lpp_v3_project_warnings)

add_library(lpp_v3_contracts src/contracts/base_types.cpp)
target_include_directories(
  lpp_v3_contracts
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_contracts
  PRIVATE Eigen3::Eigen lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_wire_primitives
  src/codec/jcs_canonicalizer.cpp
  src/crypto/sha256.cpp)
target_include_directories(
  lpp_v3_wire_primitives
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${LPP_V3_PICOSHA2_INCLUDE_DIR}")
target_link_libraries(
  lpp_v3_wire_primitives
  PUBLIC lpp_v3_contracts nlohmann_json::nlohmann_json
  PRIVATE double-conversion::double-conversion
          lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_map_snapshot
  src/map/immutable_snapshot.cpp)
target_include_directories(
  lpp_v3_map_snapshot
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_map_snapshot
  PUBLIC lpp_v3_contracts lpp_v3_wire_primitives
         nlohmann_json::nlohmann_json
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_cost
  src/cost/learned_cost_snapshot.cpp)
target_include_directories(
  lpp_v3_cost
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_cost
  PUBLIC lpp_v3_contracts lpp_v3_map_snapshot
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_map
  src/map/safe_projection.cpp)
target_include_directories(
  lpp_v3_map
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_map
  PUBLIC lpp_v3_contracts lpp_v3_cost lpp_v3_map_snapshot
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_search
  src/search/candidate_ranker.cpp)
target_include_directories(
  lpp_v3_search
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_search
  PUBLIC lpp_v3_contracts
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_goal
  src/goal/terminal_resolver.cpp)
target_include_directories(
  lpp_v3_goal
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_goal
  PUBLIC lpp_v3_contracts lpp_v3_map lpp_v3_search
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_corridor
  src/corridor/convex_corridor.cpp)
target_include_directories(
  lpp_v3_corridor
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_corridor
  PUBLIC lpp_v3_contracts lpp_v3_map
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_optimization
  src/optimization/bounded_qp_solver.cpp)
target_include_directories(
  lpp_v3_optimization
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_optimization
  PUBLIC Eigen3::Eigen lpp_v3_contracts
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(lpp_v3_cache INTERFACE)
target_include_directories(
  lpp_v3_cache
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_cache
  INTERFACE lpp_v3_contracts lpp_v3_project_options)

if(LPP_V3_ENABLE_OSQP)
  # The vcpkg manifest and override pin OSQP to 1.0.0. Its upstream
  # generated package-version file reports 0.0.0 when built from the
  # release archive, so version enforcement must remain at the registry
  # boundary instead of using find_package(... EXACT).
  find_package(osqp CONFIG REQUIRED)
  add_library(
    lpp_v3_osqp_backend
    src/optimization/osqp_bounded_qp_solver.cpp)
  target_compile_definitions(
    lpp_v3_osqp_backend
    PUBLIC LPP_V3_HAS_OSQP=1)
  target_include_directories(
    lpp_v3_osqp_backend
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(
    lpp_v3_osqp_backend
    PUBLIC lpp_v3_optimization osqp::osqp
    PRIVATE lpp_v3_project_options lpp_v3_project_warnings)
endif()

add_library(
  lpp_v3_codec
  src/codec/json_codec.cpp
  src/codec/semantic_validator.cpp)
target_include_directories(
  lpp_v3_codec
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_codec
  PUBLIC lpp_v3_contracts lpp_v3_cost lpp_v3_map
         lpp_v3_wire_primitives
         nlohmann_json::nlohmann_json
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(lpp_v3_common INTERFACE)
target_link_libraries(
  lpp_v3_common
  INTERFACE
    lpp_v3_contracts
    lpp_v3_codec
    lpp_v3_map
    lpp_v3_cost
    lpp_v3_goal
    lpp_v3_search
    lpp_v3_corridor
    lpp_v3_optimization
    lpp_v3_cache)

add_library(
  lpp_v3_wheel
  src/wheel/wheel_capability.cpp
  src/wheel/wheel_corridor.cpp
  src/wheel/wheel_lattice.cpp
  src/wheel/wheel_planner.cpp
  src/wheel/wheel_primitive.cpp
  src/wheel/wheel_search.cpp
  src/wheel/wheel_spline_optimizer.cpp
  src/wheel/wheel_sweep_validator.cpp
  src/wheel/wheel_timing.cpp)
target_include_directories(
  lpp_v3_wheel
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_wheel
  PUBLIC lpp_v3_common
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_legged
  src/legged/body_motion_primitive.cpp
  src/legged/height_interval.cpp
  src/legged/legged_capability.cpp
  src/legged/legged_corridor.cpp
  src/legged/legged_lattice.cpp
  src/legged/legged_planner.cpp
  src/legged/legged_search.cpp
  src/legged/legged_spline_optimizer.cpp
  src/legged/legged_terrain.cpp
  src/legged/legged_timing.cpp)
target_include_directories(
  lpp_v3_legged
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_legged
  PUBLIC lpp_v3_common
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_hopper
  src/hopper/aim_point_generator.cpp
  src/hopper/attitude_certifier.cpp
  src/hopper/ballistic_kinematics.cpp
  src/hopper/ballistic_time_solver.cpp
  src/hopper/flight_tube_certifier.cpp
  src/hopper/hop_certifier.cpp
  src/hopper/hopper_commitment_state_machine.cpp
  src/hopper/hopper_config.cpp
  src/hopper/hopper_planner.cpp
  src/hopper/landing_geometry.cpp
  src/hopper/landing_graph.cpp
  src/hopper/landing_region.cpp
  src/hopper/landing_set_propagator.cpp
  src/hopper/safe_pose_mask.cpp
  src/hopper/swept_footprint.cpp)
target_include_directories(
  lpp_v3_hopper
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_hopper
  PUBLIC lpp_v3_common
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_api
  src/api/execution_arbitrator.cpp
  src/api/planner_v3.cpp
  src/api/reference_bundle_builder.cpp)
target_include_directories(
  lpp_v3_api
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_api
  PUBLIC lpp_v3_common
  PRIVATE lpp_v3_wheel lpp_v3_legged lpp_v3_hopper
          lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_python_projection
  src/bindings/python_projection.cpp)
target_include_directories(
  lpp_v3_python_projection
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  lpp_v3_python_projection
  PUBLIC lpp_v3_common
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_benchmark_report
  benchmarks/benchmark_report.cpp)
target_include_directories(
  lpp_v3_benchmark_report
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
target_link_libraries(
  lpp_v3_benchmark_report
  PUBLIC lpp_v3_common
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_benchmark_runner
  benchmarks/benchmark_runner.cpp)
target_include_directories(
  lpp_v3_benchmark_runner
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
target_link_libraries(
  lpp_v3_benchmark_runner
  PUBLIC lpp_v3_api lpp_v3_benchmark_report
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_multiscale_experiment_lib
  benchmarks/multiscale_experiment.cpp
  tests/fixtures/system/planner_v3_system_fixture.cpp)
target_include_directories(
  lpp_v3_multiscale_experiment_lib
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(
  lpp_v3_multiscale_experiment_lib
  PUBLIC lpp_v3_api
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

set_target_properties(
  lpp_v3_project_options PROPERTIES EXPORT_NAME project_options)
set_target_properties(
  lpp_v3_project_warnings PROPERTIES EXPORT_NAME project_warnings)
set_target_properties(lpp_v3_contracts PROPERTIES EXPORT_NAME contracts)
set_target_properties(
  lpp_v3_wire_primitives PROPERTIES EXPORT_NAME wire_primitives)
set_target_properties(
  lpp_v3_map_snapshot PROPERTIES EXPORT_NAME map_snapshot)
set_target_properties(lpp_v3_cost PROPERTIES EXPORT_NAME cost)
set_target_properties(lpp_v3_map PROPERTIES EXPORT_NAME map)
set_target_properties(lpp_v3_search PROPERTIES EXPORT_NAME search)
set_target_properties(lpp_v3_goal PROPERTIES EXPORT_NAME goal)
set_target_properties(lpp_v3_corridor PROPERTIES EXPORT_NAME corridor)
set_target_properties(
  lpp_v3_optimization PROPERTIES EXPORT_NAME optimization)
set_target_properties(lpp_v3_cache PROPERTIES EXPORT_NAME cache)
set_target_properties(lpp_v3_codec PROPERTIES EXPORT_NAME codec)
set_target_properties(lpp_v3_common PROPERTIES EXPORT_NAME common)
set_target_properties(lpp_v3_wheel PROPERTIES EXPORT_NAME wheel)
set_target_properties(lpp_v3_legged PROPERTIES EXPORT_NAME legged)
set_target_properties(lpp_v3_hopper PROPERTIES EXPORT_NAME hopper)
set_target_properties(lpp_v3_api PROPERTIES EXPORT_NAME api)
set_target_properties(
  lpp_v3_python_projection
  PROPERTIES EXPORT_NAME python_projection)

set(
  LPP_V3_INSTALL_TARGETS
  lpp_v3_project_options
  lpp_v3_project_warnings
  lpp_v3_contracts
  lpp_v3_wire_primitives
  lpp_v3_map_snapshot
  lpp_v3_cost
  lpp_v3_map
  lpp_v3_search
  lpp_v3_goal
  lpp_v3_corridor
  lpp_v3_optimization
  lpp_v3_cache
  lpp_v3_codec
  lpp_v3_common
  lpp_v3_wheel
  lpp_v3_legged
  lpp_v3_hopper
  lpp_v3_api
  lpp_v3_python_projection)
if(TARGET lpp_v3_osqp_backend)
  set_target_properties(
    lpp_v3_osqp_backend PROPERTIES EXPORT_NAME osqp_backend)
  list(APPEND LPP_V3_INSTALL_TARGETS lpp_v3_osqp_backend)
endif()

install(
  TARGETS ${LPP_V3_INSTALL_TARGETS}
  EXPORT LunarPathPlannerV3Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
  EXPORT LunarPathPlannerV3Targets
  FILE LunarPathPlannerV3Targets.cmake
  NAMESPACE LunarPathPlannerV3::
  DESTINATION
    "${CMAKE_INSTALL_LIBDIR}/cmake/LunarPathPlannerV3")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PlannerV3Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/LunarPathPlannerV3Config.cmake"
  INSTALL_DESTINATION
    "${CMAKE_INSTALL_LIBDIR}/cmake/LunarPathPlannerV3")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LunarPathPlannerV3ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LunarPathPlannerV3Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LunarPathPlannerV3ConfigVersion.cmake"
  DESTINATION
    "${CMAKE_INSTALL_LIBDIR}/cmake/LunarPathPlannerV3")

if(LPP_V3_BUILD_TESTS)
  enable_testing()
  find_package(GTest 1.17.0 EXACT CONFIG REQUIRED)
  add_executable(lpp_v3_toolchain_tests tests/toolchain/smoke_test.cpp)
  target_link_libraries(
    lpp_v3_toolchain_tests
    PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json
            GTest::gtest_main lpp_v3_project_options lpp_v3_project_warnings)
  include(GoogleTest)
  gtest_discover_tests(
    lpp_v3_toolchain_tests
    TEST_PREFIX "lpp_v3_toolchain.")

  add_executable(
    lpp_v3_contracts_tests tests/contracts/planning_request_test.cpp)
  target_link_libraries(
    lpp_v3_contracts_tests
    PRIVATE lpp_v3_contracts GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_contracts_tests
    TEST_PREFIX "lpp_v3_contracts.")

  add_executable(
    lpp_v3_references_tests tests/contracts/reference_bundle_test.cpp)
  target_link_libraries(
    lpp_v3_references_tests
    PRIVATE lpp_v3_contracts GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_references_tests
    TEST_PREFIX "lpp_v3_references.")

  add_executable(
    lpp_v3_map_tests
    tests/map/immutable_snapshot_test.cpp
    tests/map/safe_projection_test.cpp)
  target_link_libraries(
    lpp_v3_map_tests
    PRIVATE lpp_v3_map GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_map_tests
    TEST_PREFIX "lpp_v3_map.")

  add_executable(
    lpp_v3_cost_tests tests/cost/learned_cost_snapshot_test.cpp)
  target_link_libraries(
    lpp_v3_cost_tests
    PRIVATE lpp_v3_cost GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_cost_tests
    TEST_PREFIX "lpp_v3_cost.")

  add_executable(
    lpp_v3_search_tests
    tests/search/ara_star_test.cpp
    tests/search/candidate_ranker_test.cpp)
  target_link_libraries(
    lpp_v3_search_tests
    PRIVATE lpp_v3_search GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_search_tests
    TEST_PREFIX "lpp_v3_search.")

  add_executable(
    lpp_v3_goal_tests
    tests/goal/terminal_resolver_test.cpp)
  target_link_libraries(
    lpp_v3_goal_tests
    PRIVATE lpp_v3_goal GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_goal_tests
    TEST_PREFIX "lpp_v3_goal.")

  add_executable(
    lpp_v3_corridor_tests
    tests/corridor/convex_corridor_test.cpp)
  target_link_libraries(
    lpp_v3_corridor_tests
    PRIVATE lpp_v3_corridor GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_corridor_tests
    TEST_PREFIX "lpp_v3_corridor.")

  add_executable(
    lpp_v3_optimization_tests
    tests/optimization/bounded_qp_solver_test.cpp)
  target_link_libraries(
    lpp_v3_optimization_tests
    PRIVATE lpp_v3_optimization GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_optimization_tests
    TEST_PREFIX "lpp_v3_optimization.")

  add_executable(
    lpp_v3_cache_tests
    tests/cache/deterministic_cache_test.cpp)
  target_link_libraries(
    lpp_v3_cache_tests
    PRIVATE lpp_v3_cache GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_cache_tests
    TEST_PREFIX "lpp_v3_cache.")

  if(LPP_V3_ENABLE_OSQP)
    add_executable(
      lpp_v3_osqp_tests
      tests/optimization/osqp_bounded_qp_solver_test.cpp)
    target_link_libraries(
      lpp_v3_osqp_tests
      PRIVATE lpp_v3_osqp_backend GTest::gtest_main
              lpp_v3_project_options lpp_v3_project_warnings)
    gtest_discover_tests(
      lpp_v3_osqp_tests
      TEST_PREFIX "lpp_v3_osqp.")
  endif()

  add_executable(
    lpp_v3_crypto_tests tests/crypto/sha256_test.cpp)
  target_link_libraries(
    lpp_v3_crypto_tests
    PRIVATE lpp_v3_wire_primitives GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_crypto_tests
    TEST_PREFIX "lpp_v3_crypto.")

  add_executable(
    lpp_v3_codec_tests
    tests/codec/jcs_canonicalizer_test.cpp
    tests/codec/json_codec_test.cpp
    tests/codec/semantic_validator_test.cpp)
  target_compile_definitions(
    lpp_v3_codec_tests
    PRIVATE
      LPP_V3_SCHEMA_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../schemas/v3")
  target_link_libraries(
    lpp_v3_codec_tests
    PRIVATE lpp_v3_codec GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  add_dependencies(lpp_v3_codec_tests lpp_v3_crypto_tests)
  gtest_discover_tests(
    lpp_v3_codec_tests
    TEST_PREFIX "lpp_v3_codec.")

  add_executable(
    lpp_v3_integration_tests
    tests/integration/shared_core_pipeline_test.cpp)
  target_link_libraries(
    lpp_v3_integration_tests
    PRIVATE lpp_v3_common GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_integration_tests
    TEST_PREFIX "lpp_v3_integration.")

  add_executable(
    lpp_v3_wheel_tests
    tests/unit/wheel/wheel_corridor_test.cpp
    tests/unit/wheel/wheel_lattice_test.cpp
    tests/unit/wheel/wheel_primitive_test.cpp
    tests/unit/wheel/wheel_spline_optimizer_test.cpp
    tests/unit/wheel/wheel_sweep_validator_test.cpp
    tests/unit/wheel/wheel_timing_test.cpp
    tests/integration/wheel/wheel_fault_matrix_test.cpp
    tests/integration/wheel/wheel_planner_test.cpp
    tests/integration/wheel/wheel_search_test.cpp
    tests/property/wheel/wheel_reference_property_test.cpp)
  target_include_directories(
    lpp_v3_wheel_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(
    lpp_v3_wheel_tests
    PRIVATE lpp_v3_wheel GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_wheel_tests
    TEST_PREFIX "lpp_v3_wheel.")

  add_executable(
    lpp_v3_legged_tests
    tests/unit/legged/body_motion_primitive_test.cpp
    tests/unit/legged/height_interval_test.cpp
    tests/unit/legged/legged_corridor_test.cpp
    tests/unit/legged/legged_lattice_test.cpp
    tests/unit/legged/legged_spline_optimizer_test.cpp
    tests/unit/legged/legged_terrain_test.cpp
    tests/unit/legged/legged_timing_test.cpp
    tests/integration/legged/legged_fault_matrix_test.cpp
    tests/integration/legged/legged_planner_test.cpp
    tests/integration/legged/legged_search_test.cpp
    tests/property/legged/legged_reference_property_test.cpp)
  target_include_directories(
    lpp_v3_legged_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(
    lpp_v3_legged_tests
    PRIVATE lpp_v3_legged GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_legged_tests
    TEST_PREFIX "lpp_v3_legged.")

  add_executable(
    lpp_v3_hopper_tests
    tests/hopper/aim_point_generator_test.cpp
    tests/hopper/attitude_certifier_test.cpp
    tests/hopper/ballistic_kinematics_test.cpp
    tests/hopper/ballistic_time_solver_test.cpp
    tests/hopper/flight_tube_certifier_test.cpp
    tests/hopper/hop_certifier_test.cpp
    tests/hopper/hopper_commitment_state_machine_test.cpp
    tests/hopper/hopper_planner_test.cpp
    tests/hopper/hopper_test_fixtures.cpp
    tests/hopper/landing_geometry_test.cpp
    tests/hopper/landing_graph_test.cpp
    tests/hopper/landing_region_test.cpp
    tests/hopper/landing_set_propagator_test.cpp
    tests/hopper/swept_footprint_test.cpp)
  target_include_directories(
    lpp_v3_hopper_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(
    lpp_v3_hopper_tests
    PRIVATE lpp_v3_hopper GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_hopper_tests
    TEST_PREFIX "lpp_v3_hopper.")

  add_executable(
    lpp_v3_api_tests
    tests/integration/api/planner_dispatch_test.cpp
    tests/unit/api/execution_arbitrator_test.cpp
    tests/unit/api/reference_bundle_builder_test.cpp)
  target_link_libraries(
    lpp_v3_api_tests
    PRIVATE lpp_v3_api GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_api_tests
    TEST_PREFIX "lpp_v3_api.")

  add_executable(
    lpp_v3_system_tests
    tests/fixtures/system/planner_v3_system_fixture.cpp
    tests/integration/system/planner_v3_fault_matrix_test.cpp
    tests/integration/system/planner_v3_system_test.cpp
    tests/property/system/planner_v3_determinism_property_test.cpp)
  target_include_directories(
    lpp_v3_system_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(
    lpp_v3_system_tests
    PRIVATE lpp_v3_api GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_system_tests
    TEST_PREFIX "lpp_v3_api.system.")

  add_executable(
    lpp_v3_schema_codec_tests
    tests/contract/schema_codec_conformance_test.cpp)
  target_compile_definitions(
    lpp_v3_schema_codec_tests
    PRIVATE
      LPP_V3_SCHEMA_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../schemas/v3")
  target_link_libraries(
    lpp_v3_schema_codec_tests
    PRIVATE lpp_v3_codec GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_schema_codec_tests
    TEST_PREFIX "lpp_v3_schema_codec.")

  add_executable(
    lpp_v3_binding_tests
    tests/unit/bindings/python_projection_test.cpp)
  target_link_libraries(
    lpp_v3_binding_tests
    PRIVATE lpp_v3_python_projection GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_binding_tests
    TEST_PREFIX "lpp_v3_python_projection.")

  add_executable(
    lpp_v3_benchmark_tests
    tests/unit/benchmark/benchmark_report_test.cpp
    tests/unit/benchmark/benchmark_runner_test.cpp)
  target_link_libraries(
    lpp_v3_benchmark_tests
    PRIVATE lpp_v3_benchmark_runner GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_benchmark_tests
    TEST_PREFIX "lpp_v3_benchmark.")

  add_executable(
    lpp_v3_multiscale_scenario_tests
    tests/fixtures/system/planner_v3_system_fixture.cpp
    tests/unit/benchmark/multiscale_scenario_test.cpp)
  target_include_directories(
    lpp_v3_multiscale_scenario_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(
    lpp_v3_multiscale_scenario_tests
    PRIVATE lpp_v3_api GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  set_target_properties(
    lpp_v3_multiscale_scenario_tests
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  gtest_discover_tests(
    lpp_v3_multiscale_scenario_tests
    TEST_PREFIX "lpp_v3_multiscale_scenario.")

  add_executable(
    lpp_v3_multiscale_experiment_tests
    tests/unit/benchmark/multiscale_experiment_test.cpp)
  target_include_directories(
    lpp_v3_multiscale_experiment_tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(
    lpp_v3_multiscale_experiment_tests
    PRIVATE lpp_v3_multiscale_experiment_lib GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  set_target_properties(
    lpp_v3_multiscale_experiment_tests
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  gtest_discover_tests(
    lpp_v3_multiscale_experiment_tests
    TEST_PREFIX "lpp_v3_multiscale_experiment.")
endif()

if(LPP_V3_BUILD_BENCHMARKS)
  find_package(benchmark 1.9.5 EXACT CONFIG REQUIRED)

  add_executable(
    lpp_v3_api_benchmark
    benchmarks/planner_api_benchmark.cpp
    tests/fixtures/system/planner_v3_system_fixture.cpp)
  target_include_directories(
    lpp_v3_api_benchmark
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(
    lpp_v3_api_benchmark
    PRIVATE lpp_v3_benchmark_runner
            lpp_v3_project_options lpp_v3_project_warnings)
  set_target_properties(
    lpp_v3_api_benchmark
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")

  add_executable(
    lpp_v3_multiscale_experiment
    benchmarks/multiscale_planner_experiment.cpp)
  target_link_libraries(
    lpp_v3_multiscale_experiment
    PRIVATE lpp_v3_multiscale_experiment_lib
            lpp_v3_project_options lpp_v3_project_warnings)

  add_executable(
    lpp_v3_shared_core_benchmark benchmarks/shared_core_benchmark.cpp)
  target_link_libraries(
    lpp_v3_shared_core_benchmark
    PRIVATE benchmark::benchmark lpp_v3_common
            lpp_v3_project_options)

  add_executable(
    lpp_v3_wheel_benchmark benchmarks/wheel_planner_benchmark.cpp)
  target_link_libraries(
    lpp_v3_wheel_benchmark
    PRIVATE benchmark::benchmark lpp_v3_wheel
            lpp_v3_project_options)

  add_executable(
    lpp_v3_legged_benchmark benchmarks/legged_planner_benchmark.cpp)
  target_link_libraries(
    lpp_v3_legged_benchmark
    PRIVATE benchmark::benchmark lpp_v3_legged
            lpp_v3_project_options)
endif()
