

set(MYSRC Atomics.cpp)

add_executable(Atomics ${MYSRC})
target_compile_features(Atomics PUBLIC cxx_std_14)
include_directories(../..)
target_link_libraries(Atomics yakl)

set_source_files_properties(${MYSRC} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(${MYSRC} PROPERTIES LANGUAGE CUDA)
  include_directories(../../cub)
endif()
if ("${ARCH}" STREQUAL "HIP")
  include_directories(../../hipCUB)
  include_directories(../../rocPRIM)
endif()

add_test(NAME Atomics_test COMMAND ./Atomics)

