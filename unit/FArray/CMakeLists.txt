

set(MYSRC FArray.cpp)

add_executable(FArray ${MYSRC})
target_compile_features(FArray PUBLIC cxx_std_14)
include_directories(../..)
target_link_libraries(FArray yakl)

set_source_files_properties(${MYSRC} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(${MYSRC} PROPERTIES LANGUAGE CUDA)
  include_directories(../../cub)
endif()
if ("${ARCH}" STREQUAL "HIP")
  include_directories(../../hipCUB)
  include_directories(../../rocPRIM)
endif()

add_test(NAME FArray_test COMMAND ./FArray)

