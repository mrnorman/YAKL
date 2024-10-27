cmake_minimum_required(VERSION 3.5)
project(yakl_unit)
enable_testing()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)

set(CMAKE_CXX_STANDARD 17)

set(Kokkos_HOME /ccs/home/imn/kokkos)
set(Kokkos_BIN  ${CMAKE_CURRENT_BINARY_DIR}/kokkos)
add_subdirectory(${Kokkos_HOME} ${Kokkos_BIN})
include_directories(${Kokkos_INCLUDE_DIRS_RET})

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
include_directories(${YAKL_BIN})

# Unit test directories
add_subdirectory(CArray)
add_subdirectory(FArray)
add_subdirectory(Random)
add_subdirectory(Reductions)
add_subdirectory(Atomics)
add_subdirectory(Lambda)
add_subdirectory(Fortran_Link)
add_subdirectory(Fortran_Gator)
add_subdirectory(OpenMP_Regions)
add_subdirectory(Intrinsics)
add_subdirectory(ParForC)
add_subdirectory(ParForFortran)
add_subdirectory(performance)
if (YAKL_TEST_NETCDF)
  add_subdirectory(NetCDF)
endif()

