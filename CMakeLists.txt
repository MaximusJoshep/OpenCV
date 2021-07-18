cmake_minimum_required(VERSION 3.5)

project(ImageFilter LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ImageFilter main.cpp image.h rgb.h kernel.h)

target_link_libraries( ImageFilter ${OpenCV_LIBS} )
