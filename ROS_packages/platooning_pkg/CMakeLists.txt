cmake_minimum_required(VERSION 3.0.2)
project(platooning_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  visualization_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(lidar_clustering_cpp src/lidar_clustering_cpp.cpp)

target_link_libraries(lidar_clustering_cpp
  ${catkin_LIBRARIES}
)
