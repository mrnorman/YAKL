cmake_minimum_required(VERSION 3.5)

project(YAKL
        LANGUAGES C CXX Fortran
        HOMEPAGE_URL https://github.com/mrnorman/YAKL/
        VERSION 0.1
        DESCRIPTION "A Simple C++ Framework for Performance Portability and Fortran Code Porting")

set(YAKL_F90_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/YAKL_gator_mod.F90)
set(YAKL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/YAKL_fortran_interfaces.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/YAKL.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/YAKL_gator_mod.F90)

# Ensure the Fortran file uses the desired flags
set_source_files_properties(${YAKL_F90_SOURCE} PROPERTIES COMPILE_OPTIONS "${YAKL_F90_FLAGS}")

# Add the header-only YAKL C++ library target as well as the YAKL Fortran interface library target
add_library(yakl STATIC ${YAKL_SOURCE})
target_link_libraries(yakl PUBLIC Kokkos::kokkos)
target_include_directories(yakl PUBLIC src external src/extensions ${CMAKE_CURRENT_BINARY_DIR})

option(YAKL_ENABLE_COVERAGE "Instrument YAKL and its consumers for gcov coverage" OFF)
if (YAKL_ENABLE_COVERAGE)
  if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "YAKL_ENABLE_COVERAGE requires GNU C++ and Fortran compilers")
  endif()

  # Most YAKL implementation code is header-only, so these options must propagate to consumers.
  target_compile_options(yakl PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--coverage>
                                     $<$<COMPILE_LANGUAGE:Fortran>:--coverage>)
  target_link_libraries(yakl PUBLIC --coverage)
endif()

# Special treatment for MPI
if (YAKL_HAVE_MPI)
  find_package(MPI COMPONENTS C REQUIRED)
  if(MPI_FOUND)
    target_link_libraries(yakl INTERFACE MPI::MPI_C)
    target_link_libraries(${YAKL_FORTRAN_INTERFACE_TARGET} MPI::MPI_C)
  endif()
endif()
