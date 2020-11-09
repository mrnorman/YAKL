

set(MYSRC FFT.cpp)

add_executable(FFT ${MYSRC})
target_compile_features(FFT PUBLIC cxx_std_14)
include_directories(../..)
target_link_libraries(FFT yakl)

set_source_files_properties(${MYSRC} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(${MYSRC} PROPERTIES LANGUAGE CUDA)
  include_directories(../../cub)
endif()
if ("${ARCH}" STREQUAL "HIP")
  include_directories(../../hipCUB)
  include_directories(../../rocPRIM)
endif()

add_test(NAME FFT_test COMMAND ./FFT)

