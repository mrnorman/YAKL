
set(SRC BuddyAllocator.cpp YAKL.cpp gator_mod.F90)

if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES COMPILE_FLAGS "-D__USE_CUDA__ -std=c++11 --expt-extended-lambda")
  if (NOT "${CUDA_FLAGS}" STREQUAL "")
    set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES COMPILE_FLAGS "${CUDA_FLAGS}")
  endif()
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cub)
endif()

if ("${ARCH}" STREQUAL "HIP")
  set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES COMPILE_FLAGS "-D__USE_HIP__ -std=c++11")
  if (NOT "${HIP_FLAGS}" STREQUAL "")
    set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES COMPILE_FLAGS "${HIP_FLAGS}")
  endif()
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../hipCUB)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../rocPRIM)
endif()

add_library(yakl STATIC ${SRC})


