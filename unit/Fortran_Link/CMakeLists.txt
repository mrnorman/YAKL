

set(CXX_SRC simple.cpp)
set(F90_SRC wrapper.F90)

add_executable(Fortran_Link ${CXX_SRC} ${F90_SRC})
target_compile_features(Fortran_Link PUBLIC cxx_std_14)
include_directories(../..)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../yakl)
target_link_libraries(Fortran_Link yakl)
set_target_properties(Fortran_Link PROPERTIES LINKER_LANGUAGE Fortran)

set_source_files_properties(${CXX_SRC} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(${CXX_SRC} PROPERTIES LANGUAGE CUDA)
  include_directories(../../cub)
endif()
if ("${ARCH}" STREQUAL "HIP")
  include_directories(../../hipCUB/hipcub/include)
  include_directories(../../rocPRIM/rocprim/include)
endif()

add_test(NAME Fortran_Link_test COMMAND ./Fortran_Link)

