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



ament_package()
