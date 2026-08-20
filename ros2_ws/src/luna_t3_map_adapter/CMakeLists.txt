cmake_minimum_required(VERSION 3.13)
project(luna_t3_map_adapter LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(lunar_planner_core REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/conservative_aggregation.cpp
  src/roi_level.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(${PROJECT_NAME} lunar_planner_core)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(luna_t3_map_adapter_roi_level_test
    test/roi_level_test.cpp)
  target_compile_features(luna_t3_map_adapter_roi_level_test PRIVATE cxx_std_20)
  target_link_libraries(luna_t3_map_adapter_roi_level_test ${PROJECT_NAME})

  ament_add_gtest(luna_t3_map_adapter_conservative_aggregation_test
    test/conservative_aggregation_test.cpp)
  target_compile_features(
    luna_t3_map_adapter_conservative_aggregation_test PRIVATE cxx_std_20)
  target_link_libraries(
    luna_t3_map_adapter_conservative_aggregation_test ${PROJECT_NAME})
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(lunar_planner_core)
ament_export_include_directories(include)

ament_package()
