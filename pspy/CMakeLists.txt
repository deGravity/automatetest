cmake_minimum_required(VERSION 3.8)
project(pspy)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCASCADE REQUIRED)
find_package(VTK REQUIRED CommonCore)
find_package(GLEW REQUIRED)
find_package(Freetype REQUIRED)

include_directories(${OpenCASCADE_INCLUDE_DIR})
link_directories(${OpenCASCADE_LIBRARY_DIR})

add_subdirectory(parasolid)

pybind11_add_module(pspy_cpp   "pspy.cpp" "disjointset.h" "disjointset.cpp" "lsh.h" "lsh.cpp" "eclass.h" "eclass.cpp" "types.h" "face.h" "psface.cpp" "occtface.cpp" "loop.h" "psloop.cpp" "occtloop.cpp" "edge.h" "psedge.cpp" "occtedge.cpp" "body.h" "body.cpp" "psbody.cpp" "occtbody.cpp" "topology.h" "vertex.h" "psvertex.cpp" "occtvertex.cpp" "part.h" "part.cpp" "neoflann.h" "implicit_part.cpp" "implicit_part.h")

target_link_libraries(pspy_cpp PUBLIC parasolid Eigen3::Eigen ${OpenCASCADE_LIBRARIES})

target_include_directories(pspy_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})