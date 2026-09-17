cmake_minimum_required(VERSION 3.8)
project(lunar_car_ctrl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- 依赖 ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ---- 自定义消息 ----
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VehicleTelemetry.msg"
  "msg/WheelCommands.msg"
  DEPENDENCIES std_msgs
)

# ---- 可执行文件 (分层架构: 通信层 / 功能层 / 调度层) ----
add_executable(lunar_car_node
  src/main.cpp
  src/protocol_codec.cpp
  src/tcp_client.cpp
  src/kinematics.cpp
  src/telemetry_parser.cpp
)
target_include_directories(lunar_car_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(lunar_car_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
)

# 链接自定义消息
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(lunar_car_node "${cpp_typesupport_target}")

# ---- 安装 ----
install(TARGETS lunar_car_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
