

set(MYSRC FFT.cpp)

add_executable(FFT ${MYSRC})
target_compile_features(FFT PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(FFT yakl)

add_test(NAME FFT_test COMMAND ./FFT)

