cmake_minimum_required(VERSION 3.13)
project(lunar_planner_core LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

option(LUNAR_BUILD_LEGACY_V3 "Build the private frozen planner v3 baseline" ON)
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

if(LUNAR_BUILD_LEGACY_V3)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  find_package(OpenSSL REQUIRED)
  find_package(nlohmann_json REQUIRED)
  find_package(double-conversion REQUIRED)
  find_package(fmt REQUIRED)

  set(LEGACY_V3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/migration/legacy_v3/upstream")
  set(LEGACY_V3_COMPAT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/migration/legacy_v3/compat")
  file(
    GLOB_RECURSE LEGACY_V3_SOURCES
    CONFIGURE_DEPENDS
    "${LEGACY_V3_ROOT}/src/*.cpp")
  list(
    FILTER LEGACY_V3_SOURCES
    EXCLUDE REGEX "/optimization/osqp_bounded_qp_solver\\.cpp$")

  add_library(lunar_planner_legacy_v3 STATIC ${LEGACY_V3_SOURCES})
  set_target_properties(lunar_planner_legacy_v3 PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_compile_features(lunar_planner_legacy_v3 PRIVATE cxx_std_20)
  target_include_directories(lunar_planner_legacy_v3 PRIVATE
    "${LEGACY_V3_COMPAT_ROOT}"
    "${LEGACY_V3_ROOT}/include")
  target_link_libraries(lunar_planner_legacy_v3 PRIVATE
    Eigen3::Eigen
    OpenSSL::Crypto
    double-conversion::double-conversion
    fmt::fmt
    nlohmann_json::nlohmann_json)

  add_executable(lunar_planner_legacy_summary
    src/migration/legacy_v3/legacy_summary_generator.cpp
    "${LEGACY_V3_ROOT}/tests/fixtures/system/planner_v3_system_fixture.cpp")
  target_compile_features(lunar_planner_legacy_summary PRIVATE cxx_std_20)
  target_include_directories(lunar_planner_legacy_summary PRIVATE
    "${LEGACY_V3_COMPAT_ROOT}"
    "${LEGACY_V3_ROOT}/include"
    "${LEGACY_V3_ROOT}/tests")
  target_link_libraries(lunar_planner_legacy_summary PRIVATE
    Eigen3::Eigen
    lunar_planner_legacy_v3
    nlohmann_json::nlohmann_json)

  target_sources(${PROJECT_NAME} PRIVATE
    src/migration/legacy_v3_adapter.cpp)
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    LUNAR_HAS_LEGACY_V3=1)
  target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${LEGACY_V3_COMPAT_ROOT}"
    "${LEGACY_V3_ROOT}/include")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    lunar_planner_legacy_v3)
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
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

  if(LUNAR_BUILD_LEGACY_V3)
    ament_add_gtest(lunar_planner_core_legacy_adapter_test
      test/legacy_adapter_test.cpp)
    target_include_directories(lunar_planner_core_legacy_adapter_test PRIVATE
      src test)
    target_link_libraries(lunar_planner_core_legacy_adapter_test ${PROJECT_NAME})
    lunar_enable_sanitizers(lunar_planner_core_legacy_adapter_test)

    add_executable(lunar_planner_facade_summary
      test/facade_summary.cpp)
    target_compile_features(lunar_planner_facade_summary PRIVATE cxx_std_20)
    target_include_directories(lunar_planner_facade_summary PRIVATE src test)
    target_link_libraries(lunar_planner_facade_summary PRIVATE
      ${PROJECT_NAME}
      nlohmann_json::nlohmann_json)
    lunar_enable_sanitizers(lunar_planner_facade_summary)

    add_test(
      NAME lunar_planner_core_legacy_adapter_v3_semantic_equivalence
      COMMAND "${CMAKE_COMMAND}" -E env
        "LUNAR_PLANNER_FACADE_SUMMARY=$<TARGET_FILE:lunar_planner_facade_summary>"
        "${Python3_EXECUTABLE}" -m pytest -q
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/differential/test_v3_semantic_equivalence.py")
    set_tests_properties(
      lunar_planner_core_legacy_adapter_v3_semantic_equivalence PROPERTIES
      LABELS "legacy_adapter;differential")
  endif()

  add_test(
    NAME lunar_planner_core_public_header_boundary
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/test/public_header_boundary_test.py"
      --include-root "${CMAKE_INSTALL_PREFIX}/include"
      --compiler "${CMAKE_CXX_COMPILER}")
endif()

ament_package()
