

set(MYSRC Atomics.cpp)

add_executable(Atomics ${MYSRC})
target_compile_features(Atomics PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Atomics)

add_test(NAME Atomics_test COMMAND ./Atomics)

