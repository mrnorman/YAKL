

set(MYSRC FFT.cpp)

add_executable(FFT ${MYSRC})
target_compile_features(FFT PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(FFT)

add_test(NAME FFT_test COMMAND ./FFT)

