cmake_minimum_required(VERSION 3.16)
project(lunar_observed_map LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/sparse_observed_map.cpp
  src/obstacle_classifier.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

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

  ament_add_gtest(sparse_observed_map_test
    test/sparse_observed_map_test.cpp)
  target_link_libraries(sparse_observed_map_test ${PROJECT_NAME})

  ament_add_gtest(obstacle_classifier_test
    test/obstacle_classifier_test.cpp)
  target_link_libraries(obstacle_classifier_test ${PROJECT_NAME})
endif()

ament_package()
