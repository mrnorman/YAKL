
set(YAKL_SOURCE BuddyAllocator.cpp YAKL.cpp gator_mod.F90 PARENT_SCOPE)
set(YAKL_CXX_SOURCE BuddyAllocator.cpp YAKL.cpp PARENT_SCOPE)
set(YAKL_F90_SOURCE gator_mod.F90 PARENT_SCOPE)

if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES LANGUAGE CUDA)
  set(YAKL_CXX_FLAGS "${YAKL_CXX_FLAGS} -D__USE_CUDA__ -std=c++11 --expt-extended-lambda" PARENT_SCOPE)
  if (NOT "${CUDA_FLAGS}" STREQUAL "")
    set(YAKL_CXX_FLAGS "${YAKL_CXX_FLAGS} ${CUDA_FLAGS}" PARENT_SCOPE)
  endif()
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cub)
endif()

if ("${ARCH}" STREQUAL "HIP")
  set(YAKL_CXX_FLAGS "${YAKL_CXX_FLAGS} -D__USE_HIP__ -std=c++11" PARENT_SCOPE)
  if (NOT "${HIP_FLAGS}" STREQUAL "")
    set(YAKL_CXX_FLAGS "${YAKL_CXX_FLAGS} ${HIP_FLAGS}" PARENT_SCOPE)
  endif()
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../hipCUB)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../rocPRIM)
endif()

add_library(yakl STATIC ${YAKL_SOURCE})


