

set(MYSRC Tridiagonal.cpp)

add_executable(Tridiagonal ${MYSRC})
target_compile_features(Tridiagonal PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Tridiagonal)

add_test(NAME Tridiagonal_test COMMAND ./Tridiagonal)

