cmake_minimum_required(VERSION 3.0)
project(yakl_unit)
enable_testing()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
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
add_subdirectory(Pentadiagonal)
add_subdirectory(Tridiagonal)
add_subdirectory(Lambda)
add_subdirectory(Fortran_Link)
add_subdirectory(Fortran_Gator)
add_subdirectory(OpenMP_Regions)
add_subdirectory(Intrinsics)
add_subdirectory(ParForC)
add_subdirectory(ParForFortran)

if (DEFINED NETCDF_LINK_FLAGS)
  add_subdirectory(NetCDF)
endif()

