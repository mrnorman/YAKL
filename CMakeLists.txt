project(YAKL C CXX Fortran)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()


set(YAKL_C_SOURCE src/gptl/GPTLget_memusage.c
                  src/gptl/GPTLprint_memusage.c
                  src/gptl/GPTLutil.c
                  src/gptl/gptl.c
                  src/gptl/gptl_papi.c)
set(YAKL_CXX_SOURCE src/YAKL.cpp)
set(YAKL_F90_SOURCE src/YAKL_gator_mod.F90)
set(YAKL_SOURCE ${YAKL_C_SOURCE} ${YAKL_CXX_SOURCE} ${YAKL_F90_SOURCE})

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

# Have to ensure HIP configure has happened
if ("${YAKL_ARCH}" STREQUAL "HIP")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/rocPRIM/rocprim/include/rocprim/rocprim_version.hpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocPRIM/rocprim/include/rocprim/rocprim_version.hpp"
    @ONLY
  )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipCUB/hipcub/include/hipcub/hipcub_version.hpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipCUB/hipcub/include/hipcub/hipcub_version.hpp"
    @ONLY
  )
endif()

include(yakl_utils.cmake)

yakl_process_cxx_source_files("${YAKL_CXX_SOURCE}")
message(STATUS "** YAKL is using the following C++ flags: ${YAKL_COMPILER_FLAGS} **")

set_source_files_properties(${YAKL_C_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_C_FLAGS}")
set_source_files_properties(${YAKL_F90_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_F90_FLAGS}")

add_library(yakl STATIC ${YAKL_SOURCE})
if ("${YAKL_ARCH}" STREQUAL "SYCL")
  target_compile_features(yakl PUBLIC cxx_std_17)
else()
  target_compile_features(yakl PUBLIC cxx_std_14)
endif()

target_include_directories(yakl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(yakl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gptl)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  set_target_properties(yakl PROPERTIES LINKER_LANGUAGE CUDA CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_include_directories(yakl PUBLIC cub)
endif()

if ("${YAKL_ARCH}" STREQUAL "HIP")
  target_include_directories(yakl PUBLIC hipCUB/hipcub/include)
  target_include_directories(yakl PUBLIC rocPRIM/rocprim/include)
endif()

if ("${YAKL_ARCH}" STREQUAL "SYCL")
  target_link_libraries(yakl "${YAKL_SYCL_FLAGS}")
endif()

if ("${YAKL_ARCH}" STREQUAL "OPENMP45")
  target_link_libraries(yakl "${YAKL_OPENMP45_FLAGS}")
endif()

if ("${YAKL_ARCH}" STREQUAL "OPENMP")
  target_link_libraries(yakl "${YAKL_OPENMP_FLAGS}")
endif()


