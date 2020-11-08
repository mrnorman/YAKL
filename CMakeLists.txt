
set(YAKL_SOURCE YAKL.cpp gator_mod.F90 PARENT_SCOPE)
set(YAKL_SOURCE YAKL.cpp gator_mod.F90             )
set(YAKL_CXX_SOURCE YAKL.cpp PARENT_SCOPE)
set(YAKL_CXX_SOURCE YAKL.cpp             )
set(YAKL_F90_SOURCE gator_mod.F90 PARENT_SCOPE)
set(YAKL_F90_SOURCE gator_mod.F90             )

if ("${ARCH}" STREQUAL "CUDA")
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES LANGUAGE CUDA)
  set(YAKL_CXX_FLAGS "-D__USE_CUDA__ --expt-extended-lambda --expt-relaxed-constexpr ${CUDA_FLAGS} ${YAKL_CXX_FLAGS}" PARENT_SCOPE)
  set(YAKL_CXX_FLAGS "-D__USE_CUDA__ --expt-extended-lambda --expt-relaxed-constexpr ${CUDA_FLAGS} ${YAKL_CXX_FLAGS}"             )
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
  include_directories(cub)
elseif ("${ARCH}" STREQUAL "HIP")
  set(YAKL_CXX_FLAGS "-D__USE_HIP__ ${HIP_FLAGS} ${YAKL_CXX_FLAGS}" PARENT_SCOPE)
  set(YAKL_CXX_FLAGS "-D__USE_HIP__ ${HIP_FLAGS} ${YAKL_CXX_FLAGS}"             )
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
  include_directories(hipCUB)
  include_directories(rocPRIM)
else()
  set_source_files_properties(${YAKL_CXX_SOURCE} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
endif()

add_library(yakl STATIC ${YAKL_SOURCE})
target_compile_features(yakl PUBLIC cxx_std_14)

if ("${ARCH}" STREQUAL "CUDA")
  set_target_properties(yakl PROPERTIES LINKER_LANGUAGE CUDA CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()


