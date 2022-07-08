

set(MYSRC performance.cpp)

add_executable(performance ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(performance)

