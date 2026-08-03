cmake_minimum_required(VERSION 3.13)
project(lunar_planner_training_bridge LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(lunar_planner_core REQUIRED)
find_package(pybind11_vendor REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)

pybind11_add_module(_lunar_planner_training_bridge MODULE NO_EXTRAS
  src/conversions.cpp
  src/python_bindings.cpp)
target_compile_features(_lunar_planner_training_bridge PRIVATE cxx_std_20)
target_include_directories(_lunar_planner_training_bridge PRIVATE include)
target_link_libraries(_lunar_planner_training_bridge PRIVATE
  lunar_planner_core::lunar_planner_core)

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR python/${PROJECT_NAME})
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(TARGETS _lunar_planner_training_bridge
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(${PROJECT_NAME}_test_bridge
    test/test_bridge.py
    TIMEOUT 180)
endif()

ament_package()
