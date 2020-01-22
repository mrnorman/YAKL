
set(SRC BuddyAllocator.cpp YAKL.cpp gator_mod.F90)

if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES COMPILE_FLAGS "-D__USE_CUDA__ -D__MANAGED__ -std=c++11 --expt-extended-lambda")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cub)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../hipCUB)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../rocPRIM)
endif()

add_library(yakl STATIC ${SRC})


