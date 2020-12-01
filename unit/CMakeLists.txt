cmake_minimum_required(VERSION 3.0)
project(yakl_unit)
enable_testing()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

# YAKL library
add_subdirectory(.. ./yakl)

# Unit test directories
add_subdirectory(CArray)
add_subdirectory(FArray)
add_subdirectory(Gator)
add_subdirectory(Random)
add_subdirectory(FFT)
add_subdirectory(Reductions)
add_subdirectory(Atomics)
add_subdirectory(Lambda)
