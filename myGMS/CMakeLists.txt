cmake_minimum_required(VERSION 2.8)
project(myGMS)

# OpenCV
find_package(OpenCV REQUIRED)

#
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(myGMS test.cpp)
target_link_libraries(myGMS ${OpenCV_LIBS})
