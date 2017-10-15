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

# Declare the executable target built from your sources
add_executable(detectsquare detectsquare.cpp)
# Link your application with OpenCV libraries
target_link_libraries(detectsquare ${OpenCV_LIBS})

add_executable(drawContours drawContours.cpp)
target_link_libraries(drawContours ${OpenCV_LIBS})

add_executable(threshold threshold.cpp)
target_link_libraries(threshold ${OpenCV_LIBS})

add_executable(rotatedRect rotatedRect.cpp)
target_link_libraries(rotatedRect ${OpenCV_LIBS})

add_executable(minarea minarea.cpp)
target_link_libraries(minarea ${OpenCV_LIBS})

add_executable(convexhull convexhull.cpp)
target_link_libraries(convexhull ${OpenCV_LIBS})    
