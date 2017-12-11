cmake_minimum_required(VERSION 2.8)

# Define project name
project(detectChessboard)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Declare the executable target built from your sources
add_executable(detectsquare detectsquare.cpp)
# Link your application with OpenCV libraries
target_link_libraries(detectsquare ${OpenCV_LIBS})

add_executable(drawContours drawContours.cpp)
target_link_libraries(drawContours ${OpenCV_LIBS})

add_executable(test test.cpp threshold.cpp)
target_link_libraries(test ${OpenCV_LIBS})

add_executable(rotatedRect rotatedRect.cpp convexHull.cpp rotatingCalipers.cpp threshold.cpp findContours.cpp)
target_link_libraries(rotatedRect ${OpenCV_LIBS})

add_executable(learnCvSeq learnCvSeq.cpp)
target_link_libraries(learnCvSeq ${OpenCV_LIBS})

#add_executable(convexHull convexHull.cpp)
#target_link_libraries(convexHull ${OpenCV_LIBS})

add_executable(stackHull stackHull.cpp)
target_link_libraries(stackHull ${OpenCV_LIBS})

# add_executable(monotoneChain monotoneChain.cpp)
# target_link_libraries(monotoneChain ${OpenCV_LIBS})
