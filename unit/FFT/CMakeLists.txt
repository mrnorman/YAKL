

set(MYSRC FFT.cpp)

add_executable(FFT ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(FFT)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME FFT_test COMMAND ${REAL_MPI_COMMAND} ./FFT)

if (YAKL_SYCL_BBFFT)
  #  list(APPEND CMAKE_PREFIX_PATH ${YAKL_SYCL_BBFFT_HOME})

  if (YAKL_SYCL_BBFFT_AOT)
    message("Building with BBFFT AOT")
    find_package(bbfft-sycl REQUIRED)
    find_program(OCLOC ocloc REQUIRED)
    add_executable(yakl-aot-generate generate.cpp configurations.cpp)
    target_compile_features(yakl-aot-generate PRIVATE cxx_std_17)
    target_link_libraries(yakl-aot-generate PRIVATE bbfft::bbfft-base)

    set(SRC_FILE "kernels.cl")
    set(NAMES_FILE "aot_compiled_kernels.cpp")
    if (YAKL_SYCL_BBFFT_AOT_LEGACY_UMD)
      message("Building with legacy UMD")
      # older UMD - agama-ci-devel <= 543
      set(BIN_FILE "kernels_XE_HPC_COREpvc.bin")
    else()
      message("Building with standard UMD")
      # newer UMD - ci-neo-master >= 025611
      set(BIN_FILE "kernels_pvc.bin")
    endif()
    set(OBJ_FILE "kernels.o")
    add_custom_command(
      OUTPUT ${OBJ_FILE}
      BYPRODUCTS ${SRC_FILE} ${NAMES_FILE} ${BIN_FILE}
      COMMAND yakl-aot-generate ${SRC_FILE} ${NAMES_FILE}
      COMMAND ${OCLOC} compile -file ${SRC_FILE} -device pvc -internal_options "-cl-ext=+cl_khr_fp64"
      COMMAND ${CMAKE_LINKER} -r -b binary -o ${OBJ_FILE} ${BIN_FILE}
      DEPENDS yakl-aot-generate
      COMMENT "Ahead-of-time compilation of FFT kernels"
    )
    add_library(yakl-kernels OBJECT IMPORTED)
    set_property(TARGET yakl-kernels PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_FILE}")
    target_sources(FFT PRIVATE ${NAMES_FILE})

    target_link_libraries(FFT yakl-kernels)
  else()
    message("Building with BBFFT JIT")
  endif()
endif()
