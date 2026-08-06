cmake_minimum_required(VERSION 3.5)
project(yakl_unit)
enable_testing()
option(YAKL_UNIT_LARGE_MEMORY "Build tests requiring more than 4 GB of device memory" OFF)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)

set(CMAKE_CXX_STANDARD 20)

set(Kokkos_HOME $ENV{HOME}/kokkos)
set(Kokkos_BIN  ${CMAKE_CURRENT_BINARY_DIR}/kokkos)
add_subdirectory(${Kokkos_HOME} ${Kokkos_BIN})

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})

# Unit test directories
add_subdirectory(Atomics)
add_subdirectory(CArray)
add_subdirectory(FArray)
add_subdirectory(Fortran_Gator)
add_subdirectory(Fortran_Link)
add_subdirectory(Intrinsics)
add_subdirectory(Kokkos_DeviceSpace)
add_subdirectory(Lambda)
if (YAKL_TEST_NETCDF)
  add_subdirectory(NetCDF)
endif()
if (YAKL_TEST_PNETCDF)
  add_subdirectory(PNetCDF)
endif()
add_subdirectory(OpenMP_Regions)
add_subdirectory(ParForC)
add_subdirectory(ParForFortran)
add_subdirectory(performance EXCLUDE_FROM_ALL)
add_subdirectory(Random)
add_subdirectory(Reductions)
add_subdirectory(Validation)
if (YAKL_UNIT_LARGE_MEMORY)
  add_subdirectory(LargeMemory)
endif()

if (YAKL_ENABLE_COVERAGE)
  find_program(YAKL_GCOV_EXECUTABLE gcov)
  find_program(YAKL_GCOVR_EXECUTABLE gcovr)
  if (NOT YAKL_GCOV_EXECUTABLE)
    message(FATAL_ERROR "YAKL_ENABLE_COVERAGE requires gcov")
  endif()
  if (NOT YAKL_GCOVR_EXECUTABLE)
    message(FATAL_ERROR "YAKL_ENABLE_COVERAGE requires gcovr")
  endif()

  set(YAKL_UNIT_TESTS Atomics_test
                      CArray_test
                      FArray_test
                      Fortran_Gator_test
                      Fortran_Gator_bytes_positive
                      Fortran_Gator_bytes_zero
                      Fortran_Gator_bytes_negative
                      Fortran_Gator_bytes_overflow
                      Fortran_Link_test
                      Intrinsics_test
                      Kokkos_DeviceSpace_test
                      Lambda_test
                      OpenMP_Regions_test
                      ParForC_test
                      ParForFortran_test
                      Random_test
                      Reductions_test
                      Validation_positive)
  if (YAKL_TEST_NETCDF)
    list(APPEND YAKL_UNIT_TESTS NetCDF_test)
  endif()
  if (YAKL_TEST_PNETCDF)
    list(APPEND YAKL_UNIT_TESTS PNetCDF_test)
  endif()
  if (YAKL_UNIT_LARGE_MEMORY)
    list(APPEND YAKL_UNIT_TESTS LargeMemory_test)
  endif()

  add_test(NAME zz_coverage_summary
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/coverage_summary.sh
                        ${YAKL_HOME}
                        ${CMAKE_CURRENT_BINARY_DIR}
                        ${YAKL_GCOVR_EXECUTABLE}
                        ${YAKL_GCOV_EXECUTABLE})
  set_tests_properties(zz_coverage_summary PROPERTIES DEPENDS "${YAKL_UNIT_TESTS}")
endif()
