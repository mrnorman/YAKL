
set(YAKL_C_SOURCE gptl/GPTLget_memusage.c gptl/GPTLprint_memusage.c gptl/GPTLutil.c gptl/gptl.c gptl/gptl_papi.c)
set(YAKL_CXX_SOURCE YAKL.cpp)
set(YAKL_F90_SOURCE YAKL_gator_mod.F90)
set(YAKL_SOURCE ${YAKL_C_SOURCE} ${YAKL_CXX_SOURCE} ${YAKL_F90_SOURCE})

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

process_cxx_source_files("${YAKL_CXX_SOURCE}")

set_source_files_properties(${YAKL_C_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_C_FLAGS}")
set_source_files_properties(${YAKL_F90_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_F90_FLAGS}")

add_library(yakl STATIC ${YAKL_SOURCE})
target_compile_features(yakl PUBLIC cxx_std_14)

target_include_directories(yakl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(yakl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gptl)

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


