cmake_minimum_required(VERSION 3.18)

project(mat)

set(CMAKE_CXX_STANDARD 20)

add_definitions(-fopenmp)
add_definitions(-mavx2)
add_definitions(-O3)

include_directories(./inc)

aux_source_directory(./src DIR_SRCS )

add_executable(mat ${DIR_SRCS})