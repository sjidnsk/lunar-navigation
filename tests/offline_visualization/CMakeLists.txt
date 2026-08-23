cmake_minimum_required(VERSION 3.13)
project(lunar_pure_planner_offline_visualization LANGUAGES CXX)

find_package(lunar_pure_planner_core REQUIRED)

set(LUNAR_CORE_SOURCE_DIR "" CACHE PATH "Source directory of lunar_pure_planner_core")
if(NOT IS_DIRECTORY "${LUNAR_CORE_SOURCE_DIR}/src")
  message(FATAL_ERROR "LUNAR_CORE_SOURCE_DIR must point to lunar_pure_planner_core")
endif()

add_executable(offline_wheel_demo offline_wheel_demo.cpp)
target_compile_features(offline_wheel_demo PRIVATE cxx_std_20)
target_include_directories(offline_wheel_demo PRIVATE
  "${LUNAR_CORE_SOURCE_DIR}/include"
  "${LUNAR_CORE_SOURCE_DIR}/src")
target_link_libraries(offline_wheel_demo PRIVATE
  lunar_pure_planner_core::lunar_pure_planner_core)
