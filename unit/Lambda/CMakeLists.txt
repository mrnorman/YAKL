

set(MYSRC Lambda.cpp)

add_executable(Lambda ${MYSRC})
target_compile_features(Lambda PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Lambda)

add_test(NAME Lambda_test COMMAND ./Lambda)

