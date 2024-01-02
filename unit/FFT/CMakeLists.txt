set(MYSRC FFT.cpp)

add_executable(FFT ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(FFT)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(FFT "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME FFT_test COMMAND ${REAL_MPI_COMMAND} ./FFT)

if (YAKL_SYCL_BBFFT)
  if (YAKL_SYCL_BBFFT_AOT)
    message("-- Building with BBFFT AOT")
    find_package(bbfft-aot-generate REQUIRED)
    # Adding AOT compilation for:
    # * double precision (d), real-to-complex (r), forward (f), in-place (i), FFT size of 7, maximum batch size of 16384
    # * double precision (d), real-to-complex (r), forward (b), in-place (i), FFT size of 7, maximum batch size of 16384
    # * double precision (d), real-to-complex (r), forward (f), in-place (i), FFT size of 100, maximum batch size of 16384
    set(descriptor_list drfi7*16384 drbi7*16384 drfi100*16384)
    add_aot_kernels_to_target(NO_LINK_SCOPE TARGET FFT PREFIX kernels DEVICE pvc LIST ${descriptor_list})
  else()
    message("-- Building with BBFFT JIT")
  endif()
endif()
