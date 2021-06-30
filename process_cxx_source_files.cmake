
macro(process_cxx_source_files files)

  if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_source_files_properties(${files} PROPERTIES LANGUAGE CUDA)
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_CUDA --expt-extended-lambda --expt-relaxed-constexpr ${YAKL_CUDA_FLAGS}")
  elseif ("${YAKL_ARCH}" STREQUAL "HIP")
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_HIP ${YAKL_HIP_FLAGS}")
  elseif ("${YAKL_ARCH}" STREQUAL "SYCL")
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_SYCL ${YAKL_SYCL_FLAGS}")
  elseif ("${YAKL_ARCH}" STREQUAL "OPENMP45")
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_OPENMP45 ${YAKL_OPENMP45_FLAGS}")
  else()
    set(YAKL_COMPILER_FLAGS "${YAKL_CXX_FLAGS}")
  endif()

  set_source_files_properties(${files} PROPERTIES COMPILE_FLAGS "${YAKL_COMPILER_FLAGS}")

endmacro(process_cxx_source_files)


