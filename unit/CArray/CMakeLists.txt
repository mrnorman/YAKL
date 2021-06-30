

set(MYSRC CArray.cpp)

add_executable(CArray ${MYSRC})
target_compile_features(CArray PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(CArray yakl)

add_test(NAME CArray_test COMMAND ./CArray)

