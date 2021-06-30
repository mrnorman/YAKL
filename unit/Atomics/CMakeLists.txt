

set(MYSRC Atomics.cpp)

add_executable(Atomics ${MYSRC})
target_compile_features(Atomics PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(Atomics yakl)

add_test(NAME Atomics_test COMMAND ./Atomics)

