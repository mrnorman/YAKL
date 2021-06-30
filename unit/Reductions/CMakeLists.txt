

set(MYSRC Reductions.cpp)

add_executable(Reductions ${MYSRC})
target_compile_features(Reductions PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(Reductions yakl)

add_test(NAME Reductions_test COMMAND ./Reductions)

