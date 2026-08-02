cmake_minimum_required(VERSION 3.8)
project(lunar_planner_core LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

option(LUNAR_BUILD_LEGACY_V3 "Build the private frozen planner v3 baseline" OFF)

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
endif()

ament_package()
