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

# CMake Options that turn on CPP macro defines to control YAKL's behavior
option(YAKL_VERBOSE "Turn on verbose log messages (default OFF)." OFF)
option(YAKL_VERBOSE_FILE "Turn on verbose log messages dump in file (default OFF)." OFF)
option(YAKL_DEBUG "Turn on error message throwing (default OFF)." OFF)
option(YAKL_HAVE_MPI "Turn on MPI (default OFF)." OFF)
option(YAKL_ENABLE_STREAMS "Turn on streams (default OFF)." OFF)
option(YAKL_AUTO_PROFILE "Turn on (nvtx-based) profiling (default OFF)." OFF)
option(YAKL_PROFILE "Turn on (nvtx-based) profiling (default OFF)." OFF)
option(YAKL_AUTO_FENCE "Turn on auto fence (default OFF)." OFF)
option(YAKL_B4B "Force serial computation (default OFF)." OFF)
option(YAKL_MANAGED_MEMORY "Turn on managed memory (default OFF)." OFF)
option(YAKL_MEMORY_DEBUG "Turn on memory allocation debugging (default OFF)." OFF)


set(YAKL_SYCL_BBFFT                  OFF                  CACHE BOOL    "Use Double-Batched FFT")
set(YAKL_SYCL_BBFFT_AOT              OFF                  CACHE BOOL    "Use Double-Batched FFT AOT")
set(YAKL_SYCL_BBFFT_HOME             $ENV{BBFFT_HOME}     CACHE PATH    "Double-Batched FFT Home directory")


# Inform the user of the backend being used
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

# Set YAKL_NAME_APPEND to be used to change file and target names
if ("${YAKL_TARGET_SUFFIX}" STREQUAL "")
  set(YAKL_NAME_APPEND "" PARENT_SCOPE)
  set(YAKL_NAME_APPEND "")
else()
  set(YAKL_NAME_APPEND "_${YAKL_TARGET_SUFFIX}" PARENT_SCOPE)
  set(YAKL_NAME_APPEND "_${YAKL_TARGET_SUFFIX}")
endif()

# Set YAKL_TARGET and YAKL_FORTRAN_INTERFACE_TARGET in this and parent scopes
# These are the names for the YAKL targets in this build
set(YAKL_TARGET yakl${YAKL_NAME_APPEND} PARENT_SCOPE)
set(YAKL_TARGET yakl${YAKL_NAME_APPEND})
set(YAKL_FORTRAN_INTERFACE_TARGET yakl_fortran_interface${YAKL_NAME_APPEND} PARENT_SCOPE)
set(YAKL_FORTRAN_INTERFACE_TARGET yakl_fortran_interface${YAKL_NAME_APPEND})

# Configure fortran interface files to append "_${YAKL_TARGET_SIFFIX}" if YAKL_TARGET_SUFFIX is defined
# This ensures Fortran can control the correct YAKL build
configure_file(src/YAKL_gator_mod.F90.in          src/YAKL_gator_mod${YAKL_NAME_APPEND}.F90         )
configure_file(src/YAKL_fortran_interfaces.cpp.in src/YAKL_fortran_interfaces${YAKL_NAME_APPEND}.cpp)
set(YAKL_F90_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/src/YAKL_gator_mod${YAKL_NAME_APPEND}.F90)
set(YAKL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/src/YAKL_fortran_interfaces${YAKL_NAME_APPEND}.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/src/YAKL_gator_mod${YAKL_NAME_APPEND}.F90)

# Ensure the Fortran file uses the desired flags
set_source_files_properties(${YAKL_F90_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_F90_FLAGS}")

# Add the header-only YAKL C++ library target as well as the YAKL Fortran interface library target
add_library(${YAKL_TARGET} INTERFACE)
add_library(${YAKL_FORTRAN_INTERFACE_TARGET} STATIC ${YAKL_SOURCE})

# Process the YAKL Fortran interface library target appropriately
include(yakl_utils.cmake)
yakl_process_target(${YAKL_FORTRAN_INTERFACE_TARGET})

# Let the user know what compile flags are being used
message(STATUS "** YAKL is using the following compiler flags: ${YAKL_COMPILER_FLAGS} **")

# Include directories for the YAKL target
target_include_directories(${YAKL_TARGET} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/extensions>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/extensions>)

# Special treatment for CUDA
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.22.0 )
    message("Using CUDAToolkit macros")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${YAKL_TARGET} INTERFACE CUDA::cufft)
  else()
    target_link_libraries(${YAKL_TARGET} INTERFACE cufft)
  endif()
 
  if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
    set_property(TARGET ${YAKL_TARGET} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
endif()

# Special treatment for HIP
if ("${YAKL_ARCH}" STREQUAL "HIP")
  target_link_libraries(${YAKL_TARGET} INTERFACE roctx64 rocfft)
endif()

# Special treatment for SYCL
if ("${YAKL_ARCH}" STREQUAL "SYCL")
  if (YAKL_SYCL_BBFFT)
    message("-- Building with BBFFT")
    set(CMAKE_PREFIX_PATH ${YAKL_SYCL_BBFFT_HOME})
    find_package(bbfft-sycl REQUIRED)
    target_link_libraries(${YAKL_TARGET} INTERFACE bbfft::bbfft-sycl "${YAKL_COMPILER_FLAGS}")
  else()
    if(NOT MKL_ROOT)
      set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "Folder contains MKL")
    endif(NOT MKL_ROOT)
    message("-- ** Looking for MKL installation, ${MKL_ROOT}")
    add_compile_options(-fsycl-default-sub-group-size=16)
    add_compile_definitions(MKL_ILP64)
    find_path(MKL_INCLUDE mkl.h HINTS $ENV{MKLROOT} ${MKL_ROOT} PATH_SUFFIXES include)
    target_include_directories(${YAKL_TARGET} INTERFACE "${MKL_INCLUDE}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(MKL_SEQU NAMES mkl_sequential PATHS ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib REQUIRED)
    find_library(MKL_CORE NAMES mkl_core PATHS ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib REQUIRED)
    find_library(MKL_SYCL NAMES mkl_sycl PATHS ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib REQUIRED)
    find_library(MKL_IPL64 NAMES mkl_intel_ilp64 PATHS ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib REQUIRED)
    target_link_libraries(${YAKL_TARGET} INTERFACE "${YAKL_COMPILER_FLAGS} ${MKL_SYCL} -Wl,-export-dynamic -Wl,--start-group ${MKL_SEQU} ${MKL_IPL64} ${MKL_CORE} -Wl,--end-group -lsycl -lOpenCL -lpthread -lm -ldl")
  endif()
endif()

# Special treatment for SYCL
if ("${YAKL_ARCH}" STREQUAL "OPENMP")
  target_link_libraries(${YAKL_TARGET} INTERFACE "${YAKL_OPENMP_FLAGS}")
endif()

# Special treatment for MPI
if (YAKL_HAVE_MPI)
  find_package(MPI COMPONENTS C REQUIRED)
  if(MPI_FOUND)
    target_link_libraries(${YAKL_TARGET} INTERFACE MPI::MPI_C)
    target_link_libraries(${YAKL_FORTRAN_INTERFACE_TARGET} MPI::MPI_C)
  endif()
endif()

######################################################################
## Everything below is installation stuff
######################################################################
# Generate configs files
include(CMakePackageConfigHelpers)

# setup some variables
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
write_basic_package_version_file(${PROJECT_BINARY_DIR}/yakl-config-version.cmake
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

# generate cmake my_package-config.cmake file
configure_package_config_file(${PROJECT_SOURCE_DIR}/src/yakl-config.cmake.in
                              ${PROJECT_BINARY_DIR}/src/yakl-config.cmake
                              INSTALL_DESTINATION ${INSTALL_CMAKE_DIR})

configure_package_config_file(${PROJECT_SOURCE_DIR}/src/yakl-variables.cmake.in
                              ${PROJECT_BINARY_DIR}/src/yakl-variables.cmake
                              INSTALL_DESTINATION ${INSTALL_CMAKE_DIR})

# setup the list of targets to install
set(INSTALL_TARGETS ${YAKL_TARGET} ${YAKL_FORTRAN_INTERFACE_TARGET})

# Use a namespace because CMake provides better diagnostics for namespaced imported targets.
export(TARGETS ${INSTALL_TARGETS} NAMESPACE ${YAKL_TARGET}:: FILE ${PROJECT_BINARY_DIR}/${targets_export_name}.cmake)

# Install cmake config and target files.
install(FILES ${PROJECT_BINARY_DIR}/src/yakl-config.cmake
              ${PROJECT_BINARY_DIR}/src/yakl-variables.cmake
              ${PROJECT_BINARY_DIR}/yakl-config-version.cmake
        DESTINATION ${INSTALL_CMAKE_DIR})

install(EXPORT ${targets_export_name} DESTINATION ${INSTALL_CMAKE_DIR} NAMESPACE ${YAKL_TARGET}::)

# install libyakl
install(TARGETS ${INSTALL_TARGETS}
        EXPORT ${targets_export_name}
        LIBRARY DESTINATION ${INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY src/ external/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gator_mod${YAKL_NAME_APPEND}.mod DESTINATION include)
install(FILES ${PROJECT_BINARY_DIR}/src/yakl-variables.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(CODE "file(WRITE ${CMAKE_INSTALL_PREFIX}/yakl-variables.make \"YAKL_COMPILER_FLAGS := ${YAKL_COMPILER_FLAGS}\nYAKL_ARCH := ${YAKL_ARCH}\n\")" )
