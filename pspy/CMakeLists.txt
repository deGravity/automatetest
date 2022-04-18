cmake_minimum_required(VERSION 3.8)
project(pspy)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCASCADE REQUIRED)

add_subdirectory(parasolid)

pybind11_add_module(pspy_cpp   
	"pspy.cpp" 
	"disjointset.h" "disjointset.cpp" 
	"lsh.h" "lsh.cpp" 
	"eclass.h" "eclass.cpp" 
	"types.h" 
	"face.h" "psface.cpp" "occtface.cpp" 
	"loop.h" "psloop.cpp" "occtloop.cpp" 
	"edge.h" "psedge.cpp" "occtedge.cpp" 
	"body.h" "body.cpp" "psbody.cpp" "occtbody.cpp" 
	"topology.h" 
	"vertex.h" "psvertex.cpp" "occtvertex.cpp" 
	"part.h" "part.cpp" 
	"neoflann.h"  
	"implicit_part.h" "implicit_part.cpp"
	)

target_link_libraries(pspy_cpp PUBLIC parasolid Eigen3::Eigen TKernel TKXSBase TKSTEP TKBRep TKG3d TKMath TKMesh TKTopAlgo TKShHealing)
target_include_directories(pspy_cpp PUBLIC ${OpenCASCADE_INCLUDE_DIR})
target_link_directories(pspy_cpp PUBLIC ${OpenCASCADE_LIBRARY_DIR})

add_definitions(-DTEST_X_T="${CMAKE_CURRENT_SOURCE_DIR}/brick.x_t")
add_definitions(-DTEST_STEP="${CMAKE_CURRENT_SOURCE_DIR}/brick.step")

add_executable(debug
	"main.cpp"
	"disjointset.h" "disjointset.cpp" 
	"lsh.h" "lsh.cpp" 
	"eclass.h" "eclass.cpp" 
	"types.h" 
	"face.h" "psface.cpp" "occtface.cpp" 
	"loop.h" "psloop.cpp" "occtloop.cpp" 
	"edge.h" "psedge.cpp" "occtedge.cpp" 
	"body.h" "body.cpp" "psbody.cpp" "occtbody.cpp" 
	"topology.h" 
	"vertex.h" "psvertex.cpp" "occtvertex.cpp" 
	"part.h" "part.cpp" 
	"neoflann.h"  
	"implicit_part.h" "implicit_part.cpp"
)

target_link_libraries(debug PUBLIC parasolid Eigen3::Eigen TKernel TKXSBase TKSTEP TKBRep TKG3d TKMath TKMesh TKTopAlgo TKShHealing)
target_include_directories(debug PUBLIC ${OpenCASCADE_INCLUDE_DIR})
target_link_directories(debug PUBLIC ${OpenCASCADE_LIBRARY_DIR})