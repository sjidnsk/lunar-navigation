cmake_minimum_required(VERSION 3.22)
project(lunar_pure_wheeled_controller)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(lunar_planning_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR python/${PROJECT_NAME})

install(PROGRAMS
  scripts/lunar_pure_wheeled_controller_node.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(${PROJECT_NAME}_reference_test test/test_reference.py)
  ament_add_pytest_test(${PROJECT_NAME}_tracking_test test/test_tracking.py)
  ament_add_pytest_test(${PROJECT_NAME}_node_test test/test_node.py)
endif()

ament_package()
