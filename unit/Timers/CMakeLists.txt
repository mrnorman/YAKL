set(TIMER_TARGETS Timers_disabled Timers_profile Timers_auto_profile)

foreach(TARGET_NAME IN LISTS TIMER_TARGETS)
  add_executable(${TARGET_NAME} Timers.cpp)
  target_link_libraries(${TARGET_NAME} yakl)
  target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
  if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
    target_link_libraries(${TARGET_NAME} "${YAKL_UNIT_CXX_LINK_FLAGS}")
  endif()
endforeach()

target_compile_definitions(Timers_profile PRIVATE YAKL_PROFILE)
target_compile_definitions(Timers_auto_profile PRIVATE YAKL_AUTO_PROFILE)

add_test(NAME Timers_disabled_test COMMAND ./Timers_disabled)
add_test(NAME Timers_profile_test COMMAND ./Timers_profile)
add_test(NAME Timers_auto_profile_test COMMAND ./Timers_auto_profile)
