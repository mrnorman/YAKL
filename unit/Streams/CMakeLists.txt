

set(MYSRC Streams.cpp)

add_executable(Streams ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Streams)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DYAKL_ENABLE_STREAMS")

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Streams_test COMMAND ${REAL_MPI_COMMAND} ./Streams)

