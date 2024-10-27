cmake_minimum_required(VERSION 3.0)

project(YAKL
        LANGUAGES C CXX Fortran
        HOMEPAGE_URL https://github.com/mrnorman/YAKL/
        VERSION 0.1
        DESCRIPTION "A Simple C++ Framework for Performance Portability and Fortran Code Porting")

set(YAKL_F90_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/YAKL_gator_mod.F90)
set(YAKL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/YAKL_fortran_interfaces.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/YAKL_gator_mod.F90)

# Ensure the Fortran file uses the desired flags
set_source_files_properties(${YAKL_F90_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_F90_FLAGS}")

# Add the header-only YAKL C++ library target as well as the YAKL Fortran interface library target
add_library(yakl         INTERFACE               )
add_library(yakl_fortran STATIC    ${YAKL_SOURCE})
target_link_libraries(yakl         INTERFACE kokkos)
target_link_libraries(yakl_fortran PUBLIC    kokkos)
target_include_directories(yakl         INTERFACE src external src/extensions ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(yakl_fortran PUBLIC    src external src/extensions ${CMAKE_CURRENT_BINARY_DIR})

# Special treatment for MPI
if (YAKL_HAVE_MPI)
  find_package(MPI COMPONENTS C REQUIRED)
  if(MPI_FOUND)
    target_link_libraries(yakl INTERFACE MPI::MPI_C)
    target_link_libraries(${YAKL_FORTRAN_INTERFACE_TARGET} MPI::MPI_C)
  endif()
endif()

