cmake_minimum_required(VERSION 3.22)
project(lunar_unreal_tcp_bridge LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/metadata_validation.cpp
  src/outbound_queue.cpp
  src/protocol.cpp
  src/session_protocol.cpp
  src/stream_decoder.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(nlohmann_json)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(unreal_tcp_protocol_test test/protocol_test.cpp)
  target_link_libraries(unreal_tcp_protocol_test ${PROJECT_NAME})
  target_compile_definitions(unreal_tcp_protocol_test PRIVATE
    UNREAL_TCP_VECTOR_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/fixtures/unreal_tcp/protocol_v1_vectors.json")

  ament_add_gtest(unreal_tcp_metadata_validation_test
    test/metadata_validation_test.cpp)
  target_link_libraries(unreal_tcp_metadata_validation_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_stream_decoder_test test/stream_decoder_test.cpp)
  target_link_libraries(unreal_tcp_stream_decoder_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_outbound_queue_test test/outbound_queue_test.cpp)
  target_link_libraries(unreal_tcp_outbound_queue_test ${PROJECT_NAME})

  ament_add_gtest(unreal_tcp_session_protocol_test test/session_protocol_test.cpp)
  target_link_libraries(unreal_tcp_session_protocol_test ${PROJECT_NAME})
endif()

ament_package()
