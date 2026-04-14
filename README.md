# YAKL: YAKL is A Kokkos Layer
## A Simple Kokkos-based C++ Framework for Performance Portability and Fortran Code Porting


## April, 2026 Changes: `yakl::Array` now derived from `Kokkos:View`, `yakl::DeviceSpace` is now a `Kokkos::MemorySpace`, and YAKL now uses C++20
* YAKL in its current form works with Kokkos versions 4.3 - 5.1 and will continue to work with any version > 5.1 as long as the Kokkos `KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION`, `KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_SPECIALIZATION`, `KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION`, `KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION` macro functions are defined and behave as they have from 2024-2026.
* Most importantly, `yakl::Array` (formally `yakl::CArray`) and `yakl::Array_F` (formally `yakl::FArray`) are now derived from `Kokkos::View` and, by default, use `View` member functions, data, constructors, and destructors. Further, `yakl::DeviceSpace` is now formally a `Kokkos::MemorySpace` and can be used as such for `yakl::Array`, `yakl::Array_F`, and `Kokkos::View` class objects.
   * There are now just two `yakl::Array` and `yakl::Array_F` template parameters: (1) The Kokkos-like `data_type`, e.g., `float **` that replaces the previous templates of value_type, `T`, and rank integer, `N`; and (2) the memory space, e.g. `yakl::DeviceSpace` or `Kokkos::HostSpce`. All YAKL objects are assumed to take on one of these two memory spaces, meaning it is host or device resident just like before.
      * You can construct a Kokkos View `data_type` (e.g., `int ***`) using the `template <class T, int N> yakl::ViewType` struct. E.g., `using my_view_type = typename yakl::ViewType<float,4>::type`, which set `float ****` as `my_view_type`.
      * `template <class KT, class MemSpace = yakl::DeviceSpace> yakl::Array[_F]`: The `yakl::Array` class assumes `Kokkos::LayoutRight` (row-major indexing where the last index varies the fastest), and the `yakl::Array_F` class assumes `Kokkos::LayoutLeft` (column-major indexing where the first index varies the fastest).
      * All ctors, copy ctors, move ctors, copy and move assignment operators for `yakl::Array` are derived directly from `Kokkos::View`. Because `yakl::Array` inherits ctors, dtors, assignment `operator=` and `operator()` directly from the `Kokkos::View`, it should for all intents and purposes behave just like a `View` in the Kokkos ecosystem with the traditional YAKL bells and whistles like: `Array::deep_copy_to`, `operator=(T rhs) requires std::is_arithmetic_v<T>`, `slice`, `subset_slowest_dimension`, `reshape`, `collapse`, `createDeviceObject`, `createHostObject`, `createDeviceCopy`, `createHostCopy`, `as`, `extents`, `ubounds`, `lbounds`, `begin`, `end`, and `operator<<`. The same is not guaranteed to be true for `Array_F` because it was prudent to *not* include the `View` ctors and copy and move assignment operators to avoid accidentally creating an `Array_F` object without the lower bounds properly specified.
      * For `yakl::Array_F` (the Fortran-style `Array` class) you need to use the constructors: `Array_F(std::string label, {lower1,upper1}[, ...])` and `Array_F( value_type * ptr, {lower1,upper1}[, ...])`, where the initializer lists populate `yakl::Array_F::AB` struct objects that contain a lower (inclusive) and upper bound (inclusive) for each dimension of the array.
   * `yakl::Array` and `yakl::Array_F` objects now use Kokkos's `SharedAllocationRecord` internals, which means behavior changes a bit now
      1. You can no longer name or debug-by-name an non-owning `Array` or `Array_F` class.
      2. When you call `slice`, `reshape`, `collapse` or `subset_slowest_dimension` on the host, it is no longer reference counted with the creating object like it was before but is rather a non-owning `View` that is purely at the mercy of the user making sure the owning `Array` object that created it does not deallocate its memory by falling out of scope before the non-owning object is finished being used.
   * They yakl `Style` template parameter for `yakl::Array` objects no longer exists. Rather the user needs to explicitly create an `Array` or `Array_F` object for a C-style or Fortran-style `Array` object, respectively.
* In general, a `_F` suffix appended to a class or function name is now used to indicate that it is Fortran-style, and the ommision of that suffix indicates that it is C-style. This is true for `Array[_F]`, `SArray[_F]`, `parallel_for[_F]`, `Bounds[_F]`, and `SimpleBounds[_F]`. The `Style` template parameter is removed in lieu of this clearer syntax.
* YAKL now defines its own `yakl::DeviceSpace`, which satisfies the concepts of a `Kokkos::MemorySpace`. It is assumed that if compiling for a GPU device target that the host space is not accessible. This may be relaxed in the future, but since paging memory automatically to and from device memory with CUDA / HIP Managed Memory or Linux kernel memory paging is so much less performant than user-managed memory, it is being disabled for now so that users can more easily see when they are accidentally accessing host memory on the device or device memory on the host. The core `yakl::DeviceSpace` class is extremely simple and merely calls YAKL's `alloc_device` and `free_device` routines that automatically use the YAKL memory pool when it is enabled. You can `deep_copy` between `yakl::DeviceSpace` and `KokkosHostSpace` as well as between `yakl::DeviceSpace` and `Kokkos::DefaultExecutionSpace::memory_space`. In fact, `yakl::DeviceSpace` uses `Kokkos::DefaultExecutionSpace::memory_space` behind the scenes.
* The `yakl::SArray` class (formerly aliased from `yakl::CSArray`) for small, local stack-based arrays in the memory space of the execution space being used in a `parallel_for` call has changed its template parameters. **You no longer specify the rank explicitly but rather just specify the type and the dimensions directly**. Therefore, a 2-D `SArray` object will now be specified as `template <class T, std::integral auto dimensions...> SArray`, meaning the user will now declare something like, `yakl::SArray<float,nx,ny> my_local_stack_array;`, and YAKL can infer from the constructor template parameters what the rank of the array is. The same is true for the `yakl::SArray_F` class (previously `yakl::FSArray`), except that the user needs to use the `yakl::Bnds` class to specify the lower (inclusive) and upper (inclusive) bounds of the array. E.g., `using yakl::Bnds; yakl::SArray_F<double,Bnds{1,nx},Bnds{1,nz}> my_fortran_stack_array;`. I trid to get rid of the need for specifying `Bnds` in that syntax, but not all compiler play nicely with Non-Type Template Parameters (NTTPs) resulting from the initializer list syntax. So, we're stuck with it. You can obtain the rank of an `SArray` or `SArray_F` object with the `static constexpr int rank` class data member.
   * The `SArray` and `SArray_F` classes both define the `Kokkos::View`'s `value_type`, `const_value_type`, `non_const_value_type` types. They also define `is_SArray=true`, `rank`, `is_cstyle`, and `is_fstyle` as `static constexpr` class data members. As before, they both define `operator=(T rhs) requires std:is_arithmetic_v<T>`, `operator() const`, `data()`, `begin()`, `end()`, `size()`, `span_is_contiguous() {return true;}`, `is_allocated() {return true;}`, `extent(int i)`, `extent<I>()`, `operator<<`, `extents()`, `lbounds()`, `ubounds()`.
* The recommended way to pass a YAKL `Array`, `Array_F`, `SArray`, or `SArray_F` object to a function is to declare a generic template parameter, `template <class ArrayLike>` and add a `requires` clause to constrain the properties of the object such as `requires yakl::is_Array<ArrayLike> && (ArrayLike::rank()==3) && ArrayLike::is_cstyle`  or  `requires yakl::is_SArray<ArrayLike> && (ArrayLike::rank==2) && ArrayLike::is_fstyle && std::is_integral_v<ArrayLike::value_type>`. 
* The `parallel_for` launcher is largely the same as before with a slightly more performant implementation under the hood, which still ultimately uses `Kokkos::parallel_for` for all launches but still avoids using `MDPolicyRange` due to some performance issues. The biggest changes is that if you want Fortran-style `parallel_for` and `Bounds`, you need to append the `_F` suffix to the functions to declare that rather than relying on a `Style` parameter. 
* The `yakl::componentwise` namespace now has functions for any combination of `[arithmetic_type || SArray[_F]]` or `[arithmetic_type || Array[_F]]` parameter for binary operators: `operator[+-/*<>]`, `operator<=`, `operator>=`, `operator==`, `operator!=`, `operator&&`, `operator||`. It also has functions for `Array[_F]` or `SArray[_F]` inputs to unary operators: `operator[!+-]`, `abs,sqrt,cbrt,pow(arr,arithmetic),sin,cos,tan,asin,acos,atan,exp,log,log10,log2,floor,ceil,round,isnan,isinf`. Each of these will launch a kernel to run the operation componentwise on the input(s) and return the result in a new `Array` or `SArray` object of the correction operation result type depending on the input(s). These are intended mainly to help users write debug and unit testing code more quickly, since each function/operator launches a separate kernel.
* The `yakl::intrinsics` namespace is the same as before except that `pack` has been removed because it was never running on the device anyway, and `matinv_ge` has been renamed to `matinv` and now includes partial pivoting.
* `Toney` the timer is still in there for quick and easy timer values that are queryable and given to you via `std::cout` upon `yakl::finalize()`. It's been simplified, and you now have access to the following control/lookup functions in the `yakl::` namesapce based on timer's `std::string label`: `timer_start(label)`, `timer_stop(label)`, `timer_get_last_duration(label)`, `timer_get_accumulated_duration(label)`, `timer_get_min_duration(label)`, `timer_get_max_duration(label)`, and `timer_get_count(label)`. You can also call the `timer_print()` function at any point to send the user the current timer count,min,max,accumulated values. All timers must still be perfectly nested so the class can identify parent and child timers and format the output appropriately.
* YALK's `Array[_F]` classes now accept any integral type for ctors and `operator()`, and the `Array_F` class accepts `ptrdiff_t` types for the dimension bounds. The `parallel_for[_F]` and `[Simple]Bounds[_F]` classes use `size_t` and `ptrdiff_t` types. So indexing arrays larger than 2B indices and looping over more than 2B indices is technically allowed now. However, be aware that individual GPU backends often limit these sizes to `unsigned int` types, so I still do not advise this.
* To summarize the changes a bit more succinctly:
   * `int yakl::memHost` --> `class Kokkos::HostSpace`
   * `int yakl::memDevice` --> `class yakl::DeviceSpace`
   * `yakl::c::[parallel_for|Bounds|SimpleBounds]` --> `yakl::[parallel_for|Bounds|SimpleBounds]`
   * `yakl::fortran::[parallel_for|Bounds|SimpleBounds]` --> `yakl::[parallel_for_F|Bounds_F|SimpleBounds_F]`
   * `yakl::SArray<float,3,nx,ny,nz>` --> `yakl::SArray<float,nx,ny,nz>`
   * `yakl::FSArray<double,2,SB<1,nx>,SB<1,ny>>` --> `yakl::SArray_F<double,Bnds{1,nx},Bnds{1,nz}>`
   * `template <int memSpace = yakl::memHost` --> `template <class MemSpace = Kokkos::HostSpace>`
   * `yakl::Array<float,2,yakl::memHost,yakl::styleC>` --> `yakl::Array<float **,Kokkos::HostSpace>`
   * `yakl::Array<double const,3,yakl::memDevice,yakl::styleFortran>` --> `yakl::Array_F<double const ***,yakl::DeviceSpace>`
   * `yakl::Array<float,3,yakl::memHost> arr("label",other_arr.data(),nz,ny,nx)` --> `yakl::Array<float ***,Kokkos::HostSpace> arr(other_arr.data(),nz,ny,nx)`
   * `template <class T, int N> requires std::is_arithmetic_v<T> func(yakl::Array<T,N,memDevice,styleC> const &arr)` --> `template <class ViewLike> requires is_Array<ViewLike> && std::is_arithmetic_v<ViewLike::value_type> && ViewLike::is_cstyle && ViewLike::on_device func(ViewLike const &arr)`
   * `yakl::intrinsic::matinv_ge` --> `yakl::intrinsics::matinv` (Only for `SArray` objects)
   * Is the `Array[_F]` object allocated / initialized?  `arr.is_allocated()`
   * Total number of elements in the `[S]Array[_F]` object:  `arr.size()`

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

