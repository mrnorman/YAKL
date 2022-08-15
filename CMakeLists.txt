project(YAKL C CXX Fortran)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

set(YAKL_CXX_SOURCE src/YAKL.cpp)
set(YAKL_F90_SOURCE src/YAKL_gator_mod.F90)
set(YAKL_SOURCE ${YAKL_CXX_SOURCE} ${YAKL_F90_SOURCE})

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  message(STATUS "** Building YAKL for a CUDA backend **")
elseif ("${YAKL_ARCH}" STREQUAL "HIP")
  message(STATUS "** Building YAKL for a HIP backend **")
elseif ("${YAKL_ARCH}" STREQUAL "OPENMP")
  message(STATUS "** Building YAKL for a CPU OPENMP threading backend **")
elseif ("${YAKL_ARCH}" STREQUAL "OPENMP45")
  message(STATUS "** Building YAKL for an OPENMP offload backend **")
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
set_property(TARGET yakl PROPERTY CXX_STANDARD 17)

target_include_directories(yakl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(yakl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/extensions)
target_include_directories(yakl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  set_target_properties(yakl PROPERTIES LINKER_LANGUAGE CUDA CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(yakl cufft)
endif()

if ("${YAKL_ARCH}" STREQUAL "HIP")
  target_link_libraries(yakl roctx64 rocfft)
endif()

if ("${YAKL_ARCH}" STREQUAL "SYCL")
  target_link_libraries(yakl "${YAKL_COMPILER_FLAGS}")
endif()

if ("${YAKL_ARCH}" STREQUAL "OPENMP45")
  target_link_libraries(yakl "${YAKL_OPENMP45_FLAGS}")
endif()

if ("${YAKL_ARCH}" STREQUAL "OPENMP")
  target_link_libraries(yakl "${YAKL_OPENMP_FLAGS}")
endif()


