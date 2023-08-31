cmake_minimum_required(VERSION 2.8.3)
project(pointcloud)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set variable CATKIN_DEPENDENCIES
set (CATKIN_DEPENDENCIES
  cv_bridge
  message_filters
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)

#find catkin packages set above
#Find a package and load its package-specific details
find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPENDENCIES})

#find non catkin packages
#Find a package and load its package-specific details
find_package(PCL REQUIRED)

find_package(OpenCV 3.4 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${CATKIN_DEPENDENCIES}
  DEPENDS PCL OpenCV
)

#Add the given directories to those the compiler uses to search for include files
include_directories(
	include
	${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#Set source files to compile
# set(POINTCLOUD_SRC_FILES
# )

#below we will see a few of these.

#add_library
#add library target to be built from the source files
#if no type is specified, resorts to checking BUILD_SHARED_LIBS (default static)

#add_executable
#specify executables that must be build

#add_dependencies:
#Makes a top-level <target> depend on other top-level targets to ensure that 
#they build before <target> does. A top-level target is one created by one 
#of the add_executable(), add_library()

#target_link_libraries
#Specify libraries or flags to use when linking a given target and/or its dependents

#optional: install
#After build time, targets are placed into the devel space of the catkin workspace. 
#However, often we want to install targets to the system 
#so that they can be used by others or to a local folder to test a 
#system-level installation. In other words, if you want to be able to do a "make install" 
#of your code, you need to specify where targets should end up.

# add_library(${PROJECT_NAME}
#     ${POINTCLOUD_SRC_FILES}
# )

# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(${PROJECT_NAME}_node src/pointcloud_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
