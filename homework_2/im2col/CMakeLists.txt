cmake_minimum_required(VERSION 3.12...3.18)

project(im2col_Ustinov LANGUAGES CXX)
if(UNIX)
    set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -fopenmp")
endif()
add_executable(im2col_Ustinov src/main.cpp)
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
