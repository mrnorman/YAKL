
set(YAKL_SOURCE YAKL.cpp PARENT_SCOPE)
set(YAKL_SOURCE YAKL.cpp             )
set(YAKL_CXX_SOURCE YAKL.cpp PARENT_SCOPE)
set(YAKL_CXX_SOURCE YAKL.cpp             )

if ("${ARCH}" STREQUAL "SYCL")
  set(YAKL_CXX_FLAGS "-D__USE_SYCL__ ${SYCL_FLAGS} ${YAKL_CXX_FLAGS}" PARENT_SCOPE)
  set(YAKL_CXX_FLAGS "-D__USE_SYCL__ ${SYCL_FLAGS} ${YAKL_CXX_FLAGS}"             )
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
endif()

add_library(yakl STATIC ${YAKL_SOURCE})
target_compile_features(yakl PUBLIC cxx_std_14)

if ("${ARCH}" STREQUAL "SYCL")
  target_link_libraries(yakl "--intel -fsycl")
endif()


