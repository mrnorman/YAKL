# YAKL: YAKL is A Kokkos Layer
## A Simple Kokkos-based C++ Framework for Performance Portability and Fortran Code Porting

**IMPORTANT**: YAKL is now built entirely on Kokkos, and Streams, FFTs, and hierarchical parallelism have been removed. Please note the following modifications to YAKL documentation:
* Streams and hierarchical parallelism are now gone. Now that YAKL is Kokkos, just use Kokkos if you want those things.
* Array classes are still in play as well as the pool allocator and the Fortran-style Array classes.
* The pool allocator no longer has multiple pools. This was meant to be a benefit, but it only seemed to cause confusion for users. So just one pool now. If it runs out of memory, simply make it bigger. It can be changed with `GATOR_DISABLE=1` and `GATOR_INITIAL_MB=...` just like before; as well as with parameters to `yakl::init()`;
* YAKL FFTs, Tridiagonal, and Pentadiagonal solves are now gone. They require hardware specific logic, which YAKL no longer has.
* All previous YAKL code is in the YAKL/deprecated directory now.
* YAKL is now a header-only layer that extends Kokkos, meaning the build system will use typical Kokkos linking in CMake, and you can set YAKL flags the same way you set Kokkos flags (likely with generator expressions).
* All `parallel_for` routines use Kokkos `parallel_for` and all memory allocations, copies, and frees use Kokkos supported API routines.
* All reductions use Kokkos reductions, nothing backend specific anymore like cub, hipcub, or MKL.
* Stack array / static array classes are still in here: `CSArray` (aka, `SArray`) and `FSArray`
* Intrinsics are still in here.
* `yakl::fence()` should now be `Kokkos::fence()`
* `yakl::yakl_throw()` should now be `Kokkos::abort()`
* `YAKL_INLINE` should now be `KOKKOS_INLINE_FUNCTION`
* `YAKL_LAMBDA` should now be `KOKKOS_LAMBDA`
* `YAKL_EXECUTE_ON_HOST_ONLY(...)` should now be `KOKKOS_IF_ON_HOST(...)`
* `YAKL_EXECUTE_ON_DEVICE_ONLY(...)` should now be `KOKKOS_IF_ON_DEVICE(...)`
* `yakl::atomicAdd(var,rhs)` should now be `Kokkos::atomic_add(&var,rhs)`
   * Don't forget the ampersand. Kokkos accepts a pointer where YAKL used to accept a reference

## Example compilation approach
```cmake
add_subdirectory(${KOKKOS_HOME} ${KOKKOS_BIN})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
add_executable(my_target_name ${MY_SOURCE_FILES})
target_compile_options(my_target_name PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${ADDED_CXX_FLAGS}>)
target_link_libraries(my_target_name yakl kokkos [${ADDED_LINK_FLAGS}])
```
```bash
# OLCF Frontier Example
export MPICH_GPU_SUPPORT_ENABLED=1
export MY_BACKEND="Kokkos_ENABLE_HIP"
export MH_ARCH="Kokkos_ARCH_AMD_GFX90A"
export ADDED_CXX_FLAGS="-DUSE_GPU_AWARE_MPI;-munsafe-fp-atomics;-O3;-ffast-math;-I${ROCM_PATH}/include;-D__HIP_ROCclr__;-D__HIP_ARCH_GFX90A__=1;--rocm-path=${ROCM_PATH};--offload-arch=gfx90a;-Wno-unused-result;-Wno-macro-redefined"
export ADDED_LINK_FLAGS="--rocm-path=${ROCM_PATH};-L${ROCM_PATH}/lib;-lamdhip64"
# Create the CMake command
CMAKE_COMMAND=(cmake)
CMAKE_COMMAND+=(-DADDED_CXX_FLAGS="$ADDED_CXX_FLAGS")
CMAKE_COMMAND+=(-DADDED_LINK_FLAGS="$ADDED_LINK_FLAGS")
[[ ! "$MY_BACKEND" == "" ]] && CMAKE_COMMAND+=(-D${MY_BACKEND}=ON)
[[ ! "$MY_ARCH"    == "" ]] && CMAKE_COMMAND+=(-D${MY_ARCH}=ON)
[[ "$MY_BACKEND" == "Kokkos_ENABLE_CUDA" ]] && CMAKE_COMMAND+=(-DKokkos_ENABLE_CUDA_CONSTEXPR=ON)
CMAKE_COMMAND+=($CMAKE_DIRECTORY_LOC)
# Run the CMake command
"${CMAKE_COMMAND[@]}"
```

## Documentation: https://github.com/mrnorman/YAKL/wiki

## API Documentation: https://mrnorman.github.io/yakl_api/html/index.html

## Cite YAKL: https://link.springer.com/article/10.1007/s10766-022-00739-0

Primary Developer: Matt Norman (Oak Ridge National Laboratory) - mrnorman.github.io

Contributors: https://github.com/mrnorman/YAKL/wiki#contributors

## Example YAKL Usage
For a self-contained example of how to use YAKL, please checkout the `cpp/` folder of the miniWeather repo
* https://github.com/mrnorman/miniWeather

