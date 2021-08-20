

set(MYSRC Pentadiagonal.cpp)

add_executable(Pentadiagonal ${MYSRC})
target_compile_features(Pentadiagonal PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Pentadiagonal)

add_test(NAME Pentadiagonal_test COMMAND ./Pentadiagonal)

