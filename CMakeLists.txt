cmake_minimum_required(VERSION 3.0)

project(YAKL
        LANGUAGES C CXX Fortran
        HOMEPAGE_URL https://github.com/mrnorman/YAKL/
        VERSION 0.1
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

set(YAKL_SYCL_USE_BBFFT              OFF    CACHE BOOL    "Use Double-Batched FFT")
set(YAKL_SYCL_BBFFT_USE_AOT          OFF    CACHE BOOL    "Use Double-Batched FFT AOT")
set(YAKL_SYCL_BBFFT_HOME             ""     CACHE STRING  "Double-Batched FFT Home directory")
set(YAKL_SYCL_BBFFT_AOT_LEGACY_UMD   OFF    CACHE BOOL    "Use older PVC user mode driver")

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
  $<INSTALL_INTERFACE:include>)

target_include_directories(yakl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/extensions>
  $<INSTALL_INTERFACE:include/extensions>)

set(target_list "")

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
  target_compile_options(yakl PUBLIC "-fsycl")
  if (YAKL_SYCL_USE_BBFFT)
    set(CMAKE_PREFIX_PATH ${YAKL_SYCL_BBFFT_HOME})
    set(BBFFT_AUX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/bbfft)
    find_package(bbfft-sycl REQUIRED)
    target_include_directories(yakl PUBLIC "${BBFFT_AUX_PATH}")
    target_link_libraries(yakl bbfft::bbfft-sycl "${YAKL_COMPILER_FLAGS}")
  else()
    message("Building with MKL")
    if(NOT MKL_ROOT)
      set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "Folder contains MKL")
    endif(NOT MKL_ROOT)
    message("-- ** Looking for MKL installation, ${MKL_ROOT}")

    add_compile_options(-fsycl-default-sub-group-size=16)
    add_compile_definitions(MKL_ILP64)

    find_path(MKL_INCLUDE mkl.h HINTS $ENV{MKLROOT} ${MKL_ROOT} PATH_SUFFIXES include)
    target_include_directories(yakl PUBLIC "${MKL_INCLUDE}")

    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(MKL_SEQU NAMES mkl_sequential PATHS ${MKL_ROOT}/lib/intel64 REQUIRED)
    find_library(MKL_CORE NAMES mkl_core PATHS ${MKL_ROOT}/lib/intel64 REQUIRED)
    find_library(MKL_SYCL NAMES mkl_sycl PATHS ${MKL_ROOT}/lib/intel64 REQUIRED)
    find_library(MKL_IPL64 NAMES mkl_intel_ilp64 PATHS ${MKL_ROOT}/lib/intel64 REQUIRED)
    target_link_libraries(yakl "${YAKL_COMPILER_FLAGS} ${MKL_SYCL} -Wl,-export-dynamic -Wl,--start-group ${MKL_SEQU} ${MKL_IPL64} ${MKL_CORE} -Wl,--end-group -lsycl -lOpenCL -lpthread -lm -ldl")
  endif()
endif()

if ("${YAKL_ARCH}" STREQUAL "OPENMP")
  target_link_libraries(yakl "${YAKL_OPENMP_FLAGS}")
endif()

# Generate configs files
include(CMakePackageConfigHelpers)

# setup some variables
set(version_config ${PROJECT_BINARY_DIR}/yakl-config-version.cmake)
set(project_config_src ${PROJECT_SOURCE_DIR}/src/yakl-config.cmake.in)
set(project_config_dst ${PROJECT_BINARY_DIR}/src/yakl-config.cmake)
set(targets_export_name yakl-targets)

set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE STRING
  "Installation directory for libraries, relative to ${CMAKE_INSTALL_PREFIX}.")

set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE STRING
  "Installation directory for include files, relative to ${CMAKE_INSTALL_PREFIX}.")

set(INSTALL_PKGCONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/pkgconfig CACHE PATH
  "Installation directory for pkgconfig (.pc) files, relative to ${CMAKE_INSTALL_PREFIX}.")

set(INSTALL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake CACHE STRING
  "Installation directory for cmake files, relative to ${CMAKE_INSTALL_PREFIX}.")

# Generate the version, config and target files into the build directory.
write_basic_package_version_file(
  ${version_config}
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

# generate cmake my_package-config.cmake file
configure_package_config_file(
  ${project_config_src}
  ${project_config_dst}
  INSTALL_DESTINATION ${INSTALL_CMAKE_DIR})

# setup the list of targets to install
set (INSTALL_TARGETS yakl ${target_list})

# Use a namespace because CMake provides better diagnostics
# for namespaced imported targets.
export(
  TARGETS ${INSTALL_TARGETS} NAMESPACE yakl::
  FILE ${PROJECT_BINARY_DIR}/${targets_export_name}.cmake)

# Install cmake config and target files.
install(
  FILES ${project_config_dst} ${version_config}
  DESTINATION ${INSTALL_CMAKE_DIR})

install(
  EXPORT ${targets_export_name}
  DESTINATION ${INSTALL_CMAKE_DIR}
  NAMESPACE yakl::)

# install libyakl
install(
  TARGETS yakl
  EXPORT ${targets_export_name}
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY src/ external/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gator_mod.mod DESTINATION include)
install(CODE "file(WRITE ${CMAKE_INSTALL_PREFIX}/yakl_variables.cmake \"set(YAKL_COMPILER_FLAGS \\\"${YAKL_COMPILER_FLAGS}\\\")\nset(YAKL_ARCH \\\"${YAKL_ARCH}\\\")\n\")" )
install(CODE "file(WRITE ${CMAKE_INSTALL_PREFIX}/yakl_variables.make \"YAKL_COMPILER_FLAGS := ${YAKL_COMPILER_FLAGS}\nYAKL_ARCH := ${YAKL_ARCH}\n\")" )
