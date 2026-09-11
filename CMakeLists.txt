cmake_minimum_required(VERSION 3.16)
project(RayTracer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to a sane build type if none given (helps when just running
# `cmake ..` directly without -DCMAKE_BUILD_TYPE=...).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Library target -------------------------------------------------------
# Everything here is header-only for now (BVH.hpp, ScatObject.hpp,
# BruteForce.hpp, ScatObjects/Triangle.hpp), so this is an INTERFACE
# target: no .cpp to compile into it, it just carries the include path
# and compile requirements to anything that links against it.
add_library(raytracer INTERFACE)
target_include_directories(raytracer INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(raytracer INTERFACE cxx_std_17)

# --- Tests ------------------------------------------------------------------
enable_testing()

add_executable(triangle_intersect_test tests/Triangle_Intersect.cpp)
target_link_libraries(triangle_intersect_test PRIVATE raytracer)

add_test(NAME TriangleIntersect COMMAND triangle_intersect_test)

# --- trace_scene: SurfacePatch-driven ray tracing example -------------------
# SurfacePatch.hpp (via Utils/PatchInterpolation.hpp) needs MKL for
# cblas_dgemm, mirroring IFGF-RP's own global.h. Only build this target when
# an MKL install can actually be found, so the Triangle-only library and its
# test above keep configuring on machines without oneAPI installed.
if(DEFINED ENV{MKLROOT})
    add_executable(trace_scene src/main.cpp)
    target_link_libraries(trace_scene PRIVATE raytracer)

    target_include_directories(trace_scene PRIVATE $ENV{MKLROOT}/include)
    target_link_directories(trace_scene PRIVATE $ENV{MKLROOT}/lib/intel64)
    target_link_libraries(trace_scene PRIVATE
        mkl_intel_lp64
        mkl_sequential
        mkl_core
    )

    # mpi.h is included (unused) by Utils/PatchInterpolation.hpp; only its
    # headers are needed here since nothing in this project calls into MPI.
    if(DEFINED ENV{I_MPI_ROOT})
        target_include_directories(trace_scene PRIVATE $ENV{I_MPI_ROOT}/include)
    endif()

    # RayTracingDriver.hpp uses std::filesystem; GCC < 9 needs an explicit
    # -lstdc++fs. Harmless no-op on every other toolchain.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(trace_scene PRIVATE stdc++fs)
    endif()
else()
    message(STATUS "MKLROOT not set — skipping trace_scene target (SurfacePatch needs MKL for Lagrange interpolation)")
endif()