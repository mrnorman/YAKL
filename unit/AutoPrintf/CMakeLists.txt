set(AUTO_PRINTF_TARGETS AutoPrintf_disabled AutoPrintf_enabled)

foreach(TARGET_NAME IN LISTS AUTO_PRINTF_TARGETS)
  add_executable(${TARGET_NAME} AutoPrintf.cpp)
  target_link_libraries(${TARGET_NAME} yakl)
  target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
  if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
    target_link_libraries(${TARGET_NAME} "${YAKL_UNIT_CXX_LINK_FLAGS}")
  endif()
endforeach()

target_compile_definitions(AutoPrintf_enabled PRIVATE YAKL_AUTO_PRINTF)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME AutoPrintf_disabled_test COMMAND ${REAL_MPI_COMMAND} ./AutoPrintf_disabled)
add_test(NAME AutoPrintf_enabled_test  COMMAND ${REAL_MPI_COMMAND} ./AutoPrintf_enabled)
set_tests_properties(AutoPrintf_disabled_test AutoPrintf_enabled_test PROPERTIES RUN_SERIAL TRUE)
