cmake_minimum_required(VERSION 3.0)

project(YAKL
        LANGUAGES C CXX Fortran
        HOMEPAGE_URL https://github.com/mrnorman/YAKL/
        DESCRIPTION "A Simple C++ Framework for Performance Portability and Fortran Code Porting")

# gnu compatibility,
# see https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html
include(GNUInstallDirs)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

option(YAKL_VERBOSE "Turn on verbose log messages (default OFF)." OFF)
option(YAKL_VERBOSE_FILE "Turn on verbose log messages dump in file (default OFF)." OFF)
option(YAKL_DEBUG "Turn on error message throwing (default OFF)." OFF)
option(YAKL_HAVE_MPI "Turn on MPI (default OFF)." OFF)
option(YAKL_ENABLE_STREAMS "Turn on streams (default OFF)." OFF)
option(YAKL_AUTO_PROFILE "Turn on (nvtx-based) profiling (default OFF)." OFF)
option(YAKL_AUTO_FENCE "Turn on auto fence (default OFF)." OFF)
option(YAKL_B4B "Force serial computation (default OFF)." OFF)
option(YAKL_MANAGED_MEMORY "Turn on managed memory (default OFF)." OFF)
option(YAKL_MEMORY_DEBUG "Turn on memory allocation debugging (default OFF)." OFF)


set(YAKL_CXX_SOURCE src/YAKL_fortran_interfaces.cpp)
set(YAKL_F90_SOURCE src/YAKL_gator_mod.F90)
set(YAKL_SOURCE ${YAKL_CXX_SOURCE} ${YAKL_F90_SOURCE})

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  message(STATUS "** Building YAKL for a CUDA backend **")
elseif ("${YAKL_ARCH}" STREQUAL "HIP")
  message(STATUS "** Building YAKL for a HIP backend **")
elseif ("${YAKL_ARCH}" STREQUAL "OPENMP")
  message(STATUS "** Building YAKL for a CPU OPENMP threading backend **")
elseif ("${YAKL_ARCH}" STREQUAL "SYCL")
  message(STATUS "** Building YAKL for an SYCL offload backend **")
elseif ("${YAKL_ARCH}" STREQUAL "")
  message(STATUS "** YAKL_ARCH not set. Building YAKL for a serial CPU backend **")
else()
  message(FATAL_ERROR "** YAKL_ARCH set to an invalid backend target: ${YAKL_ARCH} **")
endif()

include(yakl_utils.cmake)

yakl_process_cxx_source_files("${YAKL_CXX_SOURCE}")
message(STATUS "** YAKL is using the following C++ flags: ${YAKL_COMPILER_FLAGS} **")

set_source_files_properties(${YAKL_F90_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_F90_FLAGS}")

add_library(yakl STATIC ${YAKL_SOURCE})

if (YAKL_DEBUG)
  target_compile_definitions(yakl PUBLIC -DYAKL_DEBUG)
endif()
if (YAKL_VERBOSE)
  target_compile_definitions(yakl PUBLIC -DYAKL_VERBOSE)
endif()
if (YAKL_VERBOSE_FILE)
  target_compile_definitions(yakl PUBLIC -DYAKL_VERBOSE_FILE)
endif()
if (YAKL_HAVE_MPI)
  target_compile_definitions(yakl PUBLIC -DHAVE_MPI)
endif()
if (YAKL_ENABLE_STREAMS)
  target_compile_definitions(yakl PUBLIC -DYAKL_ENABLE_STREAMS)
endif()
if (YAKL_AUTO_PROFILE)
  target_compile_definitions(yakl PUBLIC -DYAKL_AUTO_PROFILE)
endif()
if (YAKL_AUTO_FENCE)
  target_compile_definitions(yakl PUBLIC -DYAKL_AUTO_FENCE)
endif()
if (YAKL_B4B)
  target_compile_definitions(yakl PUBLIC -DYAKL_B4B)
endif()
if (YAKL_MANAGED_MEMORY)
  target_compile_definitions(yakl PUBLIC -DYAKL_MANAGED_MEMORY)
endif()
if (YAKL_MEMORY_DEBUG)
  target_compile_definitions(yakl PUBLIC -DYAKL_MEMORY_DEBUG)
endif()

set_property(TARGET yakl PROPERTY CXX_STANDARD 17)

target_include_directories(yakl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/extensions>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/extensions>)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  set_target_properties(yakl PROPERTIES LINKER_LANGUAGE CUDA CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(yakl cufft)
  if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
    set_property(TARGET yakl PROPERTY CUDA_ARCHITECTURES OFF)
  endif()

endif()

if ("${YAKL_ARCH}" STREQUAL "HIP")
  target_link_libraries(yakl roctx64 rocfft)
endif()

if ("${YAKL_ARCH}" STREQUAL "SYCL")
  target_link_libraries(yakl "${YAKL_COMPILER_FLAGS}")
endif()

if ("${YAKL_ARCH}" STREQUAL "OPENMP")
  target_link_libraries(yakl "${YAKL_OPENMP_FLAGS}")
endif()


