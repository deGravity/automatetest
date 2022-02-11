cmake_minimum_required(VERSION 3.8)
project(pspy)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

add_subdirectory(parasolid)

add_library(pspy   "disjointset.cpp" "disjointset.h" "lsh.h" "lsh.cpp" "eclass.h" "eclass.cpp" "types.h" "face.h" "face.cpp" "loop.h" "loop.cpp" "edge.h" "edge.cpp" "body.h" "body.cpp" "topology.h" "vertex.h" "vertex.cpp" "part.h" "part.cpp")

target_link_libraries(pspy PUBLIC parasolid)

target_include_directories(pspy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

#add_definitions(-DTEST_PART="${CMAKE_CURRENT_SOURCE_DIR}/brick.x_t")
add_definitions(-DTEST_PART="/projects/grail/benjones/cadlab/data/models/b6c979cf4e6660ff81e5a9c2/67d2f776871477086fbc37c5/1520c99d5029db1cafd9d6e5/default/JHD.xt")

add_executable(debug "main.cpp")
target_link_libraries(debug PUBLIC pspy)