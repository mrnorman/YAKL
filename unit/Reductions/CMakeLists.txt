

set(MYSRC Reductions.cpp)

add_executable(Reductions ${MYSRC})
target_compile_features(Reductions PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Reductions)

add_test(NAME Reductions_test COMMAND ./Reductions)

