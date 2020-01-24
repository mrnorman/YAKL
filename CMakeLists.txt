
set(YAKL_SOURCE BuddyAllocator.cpp YAKL.cpp gator_mod.F90 PARENT_SCOPE)
set(YAKL_CXX_SOURCE BuddyAllocator.cpp YAKL.cpp PARENT_SCOPE)
set(YAKL_F90_SOURCE gator_mod.F90 PARENT_SCOPE)

if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES LANGUAGE CUDA)
  set(YAKL_CXX_FLAGS "${YAKL_CXX_FLAGS} -D__USE_CUDA__ -std=c++11 --expt-extended-lambda ${CUDA_FLAGS}" PARENT_SCOPE)
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
  if ("${YAKL_CUB_HOME}" STREQUAL "")
    message(FATAL_ERROR "The YAKL_CUB_HOME CMake variable is not defined. Please pass -DYAKL_CUB_HOME=/path/to/cub to the cmake configuration. Clone it at https://github.com/NVlabs/cub")
  endif()
  include_directories(${YAKL_CUB_HOME})
endif()

if ("${ARCH}" STREQUAL "HIP")
  set(YAKL_CXX_FLAGS "${YAKL_CXX_FLAGS} -D__USE_HIP__ -std=c++11 ${HIP_FLAGS}" PARENT_SCOPE)
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
  if ("${YAKL_HIPCUB_HOME}" STREQUAL "")
    message(FATAL_ERROR "The YAKL_HIPCUB_HOME CMake variable is not defined. Please pass -DYAKL_HIPCUB_HOME=/path/to/hipCUB to the cmake configuration. Clone it at https://github.com/ROCmSoftwarePlatform/hipCUB")
  endif()
  if ("${YAKL_ROCPRIM_HOME}" STREQUAL "")
    message(FATAL_ERROR "The YAKL_ROCPRIM_HOME CMake variable is not defined. Please pass -DYAKL_ROCPRIM_HOME=/path/to/rocPRIM to the cmake configuration. Clone it at https://github.com/ROCmSoftwarePlatform/rocPRIM")
  endif()
  include_directories(${YAKL_HIPCUB_HOME})
  include_directories(${YAKL_ROCPRIM_HOME})
endif()

add_library(yakl STATIC ${YAKL_SOURCE})


