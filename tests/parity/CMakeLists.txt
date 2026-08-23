cmake_minimum_required(VERSION 3.13)
project(pure_planner_algorithm_parity_probe LANGUAGES CXX)

set(PLANNER_VARIANT "" CACHE STRING "Planner core variant: old or pure")
set_property(CACHE PLANNER_VARIANT PROPERTY STRINGS old pure)
set(PROBE_OUTPUT_DIR "" CACHE PATH "Directory for deterministic probe JSON")

if(NOT PLANNER_VARIANT STREQUAL "old" AND
   NOT PLANNER_VARIANT STREQUAL "pure")
  message(FATAL_ERROR "PLANNER_VARIANT must be old or pure")
endif()
if(PROBE_OUTPUT_DIR STREQUAL "")
  get_filename_component(
    PROBE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/../${PLANNER_VARIANT}" ABSOLUTE)
endif()

find_package(nlohmann_json REQUIRED)
if(PLANNER_VARIANT STREQUAL "old")
  foreach(prefix IN LISTS CMAKE_PREFIX_PATH)
    if(IS_DIRECTORY "${prefix}/lunar_planner_core")
      list(APPEND CMAKE_PREFIX_PATH "${prefix}/lunar_planner_core")
    endif()
  endforeach()
  find_package(lunar_planner_core REQUIRED)
  set(PLANNER_CORE_TARGET lunar_planner_core::lunar_planner_core)
  set(PLANNER_VARIANT_DEFINITION PLANNER_VARIANT_OLD=1)
else()
  foreach(prefix IN LISTS CMAKE_PREFIX_PATH)
    if(IS_DIRECTORY "${prefix}/lunar_pure_planner_core")
      list(APPEND CMAKE_PREFIX_PATH "${prefix}/lunar_pure_planner_core")
    endif()
  endforeach()
  find_package(lunar_pure_planner_core REQUIRED)
  set(PLANNER_CORE_TARGET lunar_pure_planner_core::lunar_pure_planner_core)
  set(PLANNER_VARIANT_DEFINITION PLANNER_VARIANT_PURE=1)
endif()

add_executable(algorithm_probe algorithm_probe.cpp)
target_compile_features(algorithm_probe PRIVATE cxx_std_20)
target_compile_definitions(algorithm_probe PRIVATE ${PLANNER_VARIANT_DEFINITION})
target_link_libraries(algorithm_probe PRIVATE
  ${PLANNER_CORE_TARGET}
  nlohmann_json::nlohmann_json)

enable_testing()
file(MAKE_DIRECTORY "${PROBE_OUTPUT_DIR}")
foreach(platform IN ITEMS wheel legged hopper)
  foreach(run RANGE 1 3)
    add_test(
      NAME probe_${platform}_run_${run}
      COMMAND algorithm_probe
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${platform}.yaml"
        "${PROBE_OUTPUT_DIR}/${platform}.run-${run}.json")
  endforeach()
endforeach()
