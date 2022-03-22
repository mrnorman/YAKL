

set(MYSRC NetCDF.cpp)

add_executable(NetCDF ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(NetCDF)
target_link_libraries(NetCDF "${NETCDF_LINK_FLAGS}")

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME NetCDF_test COMMAND ${REAL_MPI_COMMAND} ./NetCDF)

