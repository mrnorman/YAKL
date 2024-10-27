

set(MYSRC performance.cpp)

add_executable(performance ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(performance)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(performance "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

