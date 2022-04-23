cmake_minimum_required(VERSION 3.12)
project(Keypoint_evaluation)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc features2d xfeatures2d)

include_directories(${OpenCV_INCLUDE_DIRS} ./)

add_executable(main main.cpp)

target_link_libraries(main ${OpenCV_LIBS})
