cmake_minimum_required(VERSION 3.8)
project(scan_merger)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(scan_merger_node
  src/scan_merger_node.cpp
)

target_include_directories(scan_merger_node SYSTEM PUBLIC
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(scan_merger_node
  ${PCL_LIBRARIES}
)

ament_target_dependencies(scan_merger_node
  nav_msgs
  pcl_conversions
  rclcpp
  sensor_msgs
)

target_link_libraries(scan_merger_node
  Eigen3::Eigen
)

install(TARGETS scan_merger_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
