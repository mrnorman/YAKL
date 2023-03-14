

function(yakl_get_lang_from_list files lang langfiles)
  set(lang_files_loc "")
  foreach(file ${files})
    get_source_file_property(lang_loc ${file} LANGUAGE)
    if ("${lang_loc}" STREQUAL "${lang}")
      list(APPEND lang_files_loc "${file}")
    endif()
  endforeach()
  set(${langfiles} "${lang_files_loc}" PARENT_SCOPE)
endfunction()



macro(yakl_set_definitions)
  set(YAKL_DEFINTIONS "")
  if ("${YAKL_TARGET_SUFFIX}" STREQUAL "")
  else()
    string(UUID YAKL_UNIQUE_NAMESPACE_LABEL_IN NAMESPACE 0599ab2c-bc4c-11ed-afa1-0242ac120002 NAME "yakl_${YAKL_TARGET_SUFFIX}" TYPE MD5 UPPER)
    string(REPLACE "-" "_" YAKL_UNIQUE_NAMESPACE_LABEL ${YAKL_UNIQUE_NAMESPACE_LABEL_IN})
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_NAMESPACE_WRAPPER_LABEL=YAKL_NS_${YAKL_UNIQUE_NAMESPACE_LABEL}")
  endif()

  if (YAKL_DEBUG)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_DEBUG")
  endif()
  if (YAKL_VERBOSE)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_VERBOSE")
  endif()
  if (YAKL_VERBOSE_FILE)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_VERBOSE_FILE")
  endif()
  if (YAKL_HAVE_MPI)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DHAVE_MPI")
  endif()
  if (YAKL_ENABLE_STREAMS)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_ENABLE_STREAMS")
  endif()
  if (YAKL_AUTO_PROFILE)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_AUTO_PROFILE")
  endif()
  if (YAKL_AUTO_FENCE)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_AUTO_FENCE")
  endif()
  if (YAKL_B4B)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_B4B")
  endif()
  if (YAKL_MANAGED_MEMORY)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_MANAGED_MEMORY")
  endif()
  if (YAKL_MEMORY_DEBUG)
    set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_MEMORY_DEBUG")
  endif()
endmacro(yakl_set_definitions)



macro(yakl_process_cxx_source_files files)
  yakl_set_definitions()
  if ("${YAKL_ARCH}" STREQUAL "CUDA")
    set_source_files_properties(${files} PROPERTIES LANGUAGE CUDA)
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_CUDA --expt-extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -std=c++17 ${YAKL_CUDA_FLAGS} ${YAKL_DEFINITIONS}")
  elseif ("${YAKL_ARCH}" STREQUAL "HIP")
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_HIP ${YAKL_HIP_FLAGS} ${YAKL_DEFINITIONS}")
  elseif ("${YAKL_ARCH}" STREQUAL "SYCL")
    if (YAKL_SYCL_BBFFT)
      set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_SYCL_BBFFT")
      if (YAKL_SYCL_BBFFT_AOT)
        set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_SYCL_BBFFT_AOT")
        if (YAKL_SYCL_BBFFT_AOT_LEGACY_UMD)
          # older UMD - agama-ci-devel <= 543
          set(YAKL_DEFINITIONS "${YAKL_DEFINITIONS} -DYAKL_SYCL_BBFFT_AOT_LEGACY_UMD")
        endif()
      endif()
    endif()
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_SYCL ${YAKL_SYCL_FLAGS} ${YAKL_DEFINITIONS}")
  elseif ("${YAKL_ARCH}" STREQUAL "OPENMP")
    set(YAKL_COMPILER_FLAGS "-DYAKL_ARCH_OPENMP ${YAKL_OPENMP_FLAGS} ${YAKL_DEFINITIONS}")
  else()
    set(YAKL_COMPILER_FLAGS "${YAKL_CXX_FLAGS} ${YAKL_DEFINITIONS}")
  endif()

  set_source_files_properties(${files} PROPERTIES COMPILE_FLAGS "${YAKL_COMPILER_FLAGS}")
  set_source_files_properties(${files} PROPERTIES CXX_STANDARD 17)
endmacro(yakl_process_cxx_source_files)




macro(yakl_process_target tname)
  get_target_property(files ${tname} SOURCES)
  yakl_get_lang_from_list("${files}" "CXX" cxxfiles)

  yakl_process_cxx_source_files("${cxxfiles}")

  set_property(TARGET ${tname} PROPERTY CXX_STANDARD 17)

  if ("${YAKL_TARGET_SUFFIX}" STREQUAL "")
    set(YAKL_TARGET yakl)
  else()
    set(YAKL_TARGET yakl_${YAKL_TARGET_SUFFIX})
  endif()

  target_link_libraries(${tname} ${YAKL_TARGET})
  if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_target_properties(${tname}  PROPERTIES CUDA_ARCHITECTURES OFF)
  endif()
endmacro(yakl_process_target)



