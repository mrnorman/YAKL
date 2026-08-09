
#include "YAKL.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

// template <typename T> struct ViewArrayAnalysis {
//   using base_type  = T;
//   using value_type = T;
//   static constexpr int rank = 0;
// };
// template <typename T> struct ViewArrayAnalysis<T*> {
//   using base_type  = typename ViewArrayAnalysis<T>::base_type;
//   using value_type = T*;
//   static constexpr int rank = ViewArrayAnalysis<T>::rank + 1;
// };

using yakl::Array;
using yakl::Array_F;
using yakl::SArray;
using yakl::SArray_F;
using yakl::Bnds;
using yakl::Bounds;
using yakl::Bounds_F;
using yakl::parallel_for;
using yakl::parallel_for_F;
using yakl::intrinsics::sum;
using yakl::COLON;


using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

#ifdef YAKL_INTERNAL_DEVICE_SPACE_IS_HOST_ACCESSIBLE
static_assert(std::is_same_v<DefaultMemorySpace,Kokkos::HostSpace>);
static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,yakl::DeviceSpace>::assignable);
static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,yakl::DeviceSpace>::accessible);
static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,yakl::DeviceSpace>::deepcopy);
static_assert(Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,Kokkos::HostSpace>::assignable);
static_assert(Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,Kokkos::HostSpace>::accessible);
static_assert(Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,Kokkos::HostSpace>::deepcopy);
#else
static_assert(! std::is_same_v<DefaultMemorySpace,Kokkos::HostSpace>);
static_assert(! Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,yakl::DeviceSpace>::assignable);
static_assert(! Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,yakl::DeviceSpace>::accessible);
static_assert(  Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,yakl::DeviceSpace>::deepcopy);
static_assert(! Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,Kokkos::HostSpace>::assignable);
static_assert(! Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,Kokkos::HostSpace>::accessible);
static_assert(  Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,Kokkos::HostSpace>::deepcopy);
static_assert(Kokkos::Impl::MemorySpaceAccess<DefaultMemorySpace,yakl::DeviceSpace>::assignable);
static_assert(Kokkos::Impl::MemorySpaceAccess<DefaultMemorySpace,yakl::DeviceSpace>::accessible);
static_assert(Kokkos::Impl::MemorySpaceAccess<DefaultMemorySpace,yakl::DeviceSpace>::deepcopy);
static_assert(Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,DefaultMemorySpace>::assignable);
static_assert(Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,DefaultMemorySpace>::accessible);
static_assert(Kokkos::Impl::MemorySpaceAccess<yakl::DeviceSpace,DefaultMemorySpace>::deepcopy);
#endif


void die(std::string const &msg) {
  Kokkos::abort(msg.c_str());
}


int main(int argc, char **argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    Array<float *,yakl::DeviceSpace> arr("arr",10);
    parallel_for( "mykernel" , 10 , KOKKOS_LAMBDA (int i) {
      arr(i) = i+1;
    });
    auto arr_h = arr.createHostObject();
    arr.deep_copy_to(arr_h);
    auto arr_d = arr_h.createDeviceCopy();
    if (sum(arr) != 55 || sum(arr_h) != 55 || sum(arr_d) != 55) { die("ERROR: host/device array copy changed values"); }
    if (arr.rank() != 1 || arr.size() != 10 || arr.data() == nullptr) { die("ERROR: incorrect dynamic array metadata"); }
    if (arr.begin() != arr.data() || arr.end() != arr.data()+10) { die("ERROR: incorrect dynamic array iterator bounds"); }
    if (! arr.span_is_contiguous() || ! arr.is_allocated()) { die("ERROR: dynamic array allocation metadata is incorrect"); }
    if (arr.label() != "arr") { die("ERROR: dynamic array label is incorrect"); }
    std::cout << sum(arr  ) << std::endl;
    std::cout << sum(arr_h) << std::endl;
    std::cout << sum(arr_d) << std::endl;
    Array<float[10],yakl::DeviceSpace> arr_s("arr_s");
    arr_s = 1;
    if (sum(arr_s) != 10) { die("ERROR: static device array assignment failed"); }
    std::cout << sum(arr_s) << std::endl;
    std::cout << "arr Rank:   " << arr.rank              () << std::endl;
    std::cout << "arr Size:   " << arr.size              () << std::endl;
    std::cout << "arr ptr:    " << arr.data              () << std::endl;
    std::cout << "arr begin:  " << arr.begin             () << std::endl;
    std::cout << "arr end:    " << arr.end               () << std::endl;
    std::cout << "arr contig: " << arr.span_is_contiguous() << std::endl;
    std::cout << "arr alloc:  " << arr.is_allocated      () << std::endl;
    std::cout << "arr alloc:  " << arr.label             () << std::endl;
    std::cout << "arr use ct: " << arr.use_count         () << std::endl;
    std::cout << arr;
    std::cout << arr.reshape(5,2);
    std::cout << arr.reshape(5,2).collapse();
    std::cout << arr.reshape(5,2).slice<1>(1,COLON);
    auto reshaped = arr.reshape(5,2);
    auto reshaped_h = reshaped.createHostCopy();
    if (reshaped_h.extent(0) != 5 || reshaped_h.extent(1) != 2) { die("ERROR: reshape returned incorrect extents"); }
    if (reshaped_h(4,1) != 10 || reshaped_h.collapse()(9) != 10) { die("ERROR: reshape or collapse changed indexing"); }
    auto sliced_h = reshaped.slice<1>(1,COLON).createHostCopy();
    if (sliced_h.extent(0) != 2 || sliced_h(0) != 3 || sliced_h(1) != 4) { die("ERROR: slice returned incorrect values"); }
    std::cout << "reshp use ct: " << arr.reshape(5,2).use_count() << std::endl;
    std::cout << arr.as<double>();
    std::cout << "arr.reshape(2,5) extents: " << arr.reshape(2,5).extents();
    std::cout << "arr.reshape(2,5) lbounds: " << arr.reshape(2,5).lbounds();
    std::cout << "arr.reshape(2,5) ubounds: " << arr.reshape(2,5).ubounds();
    SArray<float,3,2> csarray;
    csarray = 2;
    csarray(2,1) = 1;
    if (sum(csarray) != 11 || csarray.extent(0) != 3 || csarray.extent(1) != 2) {
      die("ERROR: C-style SArray metadata or values are incorrect");
    }
    std::cout << csarray;
    std::cout << csarray.extents();
    std::cout << csarray.lbounds();
    std::cout << csarray.ubounds();
    std::cout << sum(csarray) << std::endl;
    SArray_F<float,Bnds{1,3},Bnds{1,2}> fsarray;
    fsarray = 2;
    fsarray(3,2) = 1;
    if (sum(fsarray) != 11 || fsarray.lbounds()(1) != 1 || fsarray.ubounds()(2) != 2) {
      die("ERROR: Fortran-style SArray metadata or values are incorrect");
    }
    std::cout << fsarray;
    std::cout << fsarray.extents();
    std::cout << fsarray.lbounds();
    std::cout << fsarray.ubounds();
    std::cout << sum(fsarray) << std::endl;
    Array_F<float ***,yakl::DeviceSpace> farr("arr",{1,3},{1,3},{1,3});
    farr = 3;
    parallel_for_F( YAKL_AUTO_LABEL() , 1 , KOKKOS_LAMBDA (int i) { farr(1,1,1) = 1; });
    std::cout << farr.lbounds();
    std::cout << farr.ubounds();
    auto farr_re = farr.reshape(27);
    if (sum(farr) != 79 || farr_re.lbounds()(1) != 1 || farr_re.ubounds()(1) != 27) {
      die("ERROR: Fortran-style reshape metadata or values are incorrect");
    }
    std::cout << farr_re.is_fstyle << " , " << farr_re.rank << std::endl;
    std::cout << farr_re.lbounds();
    std::cout << farr_re.ubounds();
    std::cout << sum(farr) << std::endl;
    parallel_for_F( YAKL_AUTO_LABEL() , 1 , KOKKOS_LAMBDA (int i) { farr_re(27) = 1; });
    Kokkos::fence();
    if (sum(farr_re) != 77) { die("ERROR: reshaped Fortran-style array did not alias its source"); }
    std::cout << farr_re;
    std::cout << sum(farr_re) << std::endl;
    Array_F<float *,yakl::DeviceSpace> farr2("farr2",{1,10});
    parallel_for_F( YAKL_AUTO_LABEL() , 10 , KOKKOS_LAMBDA (int i) {
      farr2(i) = i;
    });
    std::cout << farr2;
    std::cout << farr2.reshape(5,2).slice<1>(COLON,2);
    Bounds_F<3> bnds({1,3},{-1,5},{2,7,2});

    // Exercise allocator rounding, exact capacity, all three hole-search paths,
    // pointer ownership, and move-only ownership without involving the global pool.
    {
      int zeroCalls = 0;
      int freeCalls = 0;
      size_t constexpr alignment = yakl::LinearAllocator::requiredAlignment;
      yakl::LinearAllocator allocator(
        4*alignment-1,
        alignment,
        [] (size_t bytes) { return std::malloc(bytes); },
        [&] (void *ptr) { freeCalls++; std::free(ptr); },
        [&] (void *ptr, size_t bytes) { zeroCalls++; std::memset(ptr,0,bytes); },
        "unit allocator"
      );
      if (! allocator.initialized() || allocator.poolSize() != 4*alignment || zeroCalls != 1) {
        die("ERROR: LinearAllocator initialization or pool rounding failed");
      }
      if (allocator.allocate(0) != nullptr || allocator.numAllocs() != 0 || ! allocator.iGotRoom(0)) {
        die("ERROR: LinearAllocator zero-byte behavior is incorrect");
      }

      void *first = allocator.allocate(1,"first");
      void *middle = allocator.allocate(alignment,"middle");
      void *last = allocator.allocate(alignment+1,"last");
      if (allocator.numAllocs() != 3 || allocator.iGotRoom(1)) { die("ERROR: LinearAllocator exact-capacity check failed"); }
      if (reinterpret_cast<uintptr_t>(first)%yakl::LinearAllocator::requiredAlignment != 0 ||
          reinterpret_cast<uintptr_t>(middle)%yakl::LinearAllocator::requiredAlignment != 0 ||
          reinterpret_cast<uintptr_t>(last)%yakl::LinearAllocator::requiredAlignment != 0) {
        die("ERROR: LinearAllocator returned a pointer that does not satisfy Kokkos memory alignment");
      }
      void *pastEnd = static_cast<void *>(static_cast<char *>(allocator.pool)+allocator.poolSize());
      if (! allocator.thisIsMyPointer(first) || allocator.thisIsMyPointer(pastEnd)) {
        die("ERROR: LinearAllocator pointer ownership bounds are incorrect");
      }

      if (allocator.free(middle,"middle") != alignment || ! allocator.iGotRoom(alignment)) {
        die("ERROR: LinearAllocator failed to expose an interior hole");
      }
      if (allocator.allocate(alignment,"middle replacement") != middle) {
        die("ERROR: LinearAllocator failed to reuse an interior hole");
      }
      if (allocator.free(first,"first") != alignment || allocator.allocate(8,"first replacement") != first) {
        die("ERROR: LinearAllocator failed to reuse a leading hole");
      }

      allocator.free(first,"first replacement");
      allocator.free(middle,"middle replacement");
      allocator.free(last,"last");
      void *movedPointer = allocator.allocate(4*alignment,"move ownership");
      yakl::LinearAllocator moved(std::move(allocator));
      if (allocator.initialized() || ! moved.initialized() || ! moved.thisIsMyPointer(movedPointer)) {
        die("ERROR: LinearAllocator move construction failed");
      }
      moved = std::move(moved);
      yakl::LinearAllocator moveAssigned;
      moveAssigned = std::move(moved);
      if (moved.initialized() || ! moveAssigned.initialized()) { die("ERROR: LinearAllocator move assignment failed"); }
      if (moveAssigned.free(movedPointer,"move ownership") != 4*alignment) {
        die("ERROR: moved LinearAllocator lost allocation state");
      }
      moveAssigned.finalize();
      if (freeCalls != 1 || moveAssigned.initialized()) { die("ERROR: LinearAllocator finalization failed"); }
    }

    // Exercise the overflow-safe room query independently of an allocation failure.
    {
      size_t constexpr alignment = yakl::LinearAllocator::requiredAlignment;
      yakl::LinearAllocator allocator(4*alignment,alignment);
      void *direct = allocator.allocate(1,"direct alignment");
      if (reinterpret_cast<uintptr_t>(direct)%alignment != 0 ||
          allocator.iGotRoom(std::numeric_limits<size_t>::max())) {
        die("ERROR: LinearAllocator alignment or overflow room query failed");
      }
      allocator.free(direct,"direct alignment");
    }

    // Repeated simultaneous allocation/free cycles exercise synchronization around the allocator's vector metadata.
    {
      size_t constexpr alignment = yakl::LinearAllocator::requiredAlignment;
      yakl::LinearAllocator allocator(
        64*alignment,
        alignment,
        [] (size_t bytes) { return std::malloc(bytes); }
      );
      int constexpr numThreads = 8;
      int constexpr numCycles  = 2000;
      std::atomic<int> ready(0);
      std::atomic<bool> go(false);
      std::atomic<bool> badQuery(false);
      std::vector<std::thread> threads;
      for (int thread=0; thread < numThreads; thread++) {
        threads.emplace_back([&,thread] {
          ready.fetch_add(1,std::memory_order_release);
          while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
          for (int cycle=0; cycle < numCycles; cycle++) {
            void *ptr = allocator.allocate(static_cast<size_t>((cycle+thread)%alignment)+1,"concurrent allocation");
            if (! allocator.initialized() || allocator.poolSize() != 64*alignment ||
                ! allocator.thisIsMyPointer(ptr) || allocator.numAllocs() == 0) {
              badQuery.store(true,std::memory_order_relaxed);
            }
            allocator.iGotRoom(1);
            if ((cycle+thread)%7 == 0) std::this_thread::yield();
            allocator.free(ptr,"concurrent allocation");
          }
        });
      }
      while (ready.load(std::memory_order_acquire) != numThreads) std::this_thread::yield();
      go.store(true,std::memory_order_release);
      for (auto &thread : threads) thread.join();
      if (badQuery.load() || allocator.numAllocs() != 0) {
        die("ERROR: concurrent LinearAllocator operations or read-only queries corrupted allocator state");
      }
    }

    // Exercise the same synchronization through YAKL's global pool and live-allocation accounting.
    if (yakl::get_yakl_instance().use_pool()) {
      size_t constexpr alignment = yakl::LinearAllocator::requiredAlignment;
      if (reinterpret_cast<uintptr_t>(yakl::get_yakl_instance().pool.pool)%alignment != 0) {
        die("ERROR: global LinearAllocator pool does not satisfy Kokkos memory alignment");
      }
      Array<int *,yakl::DeviceSpace> alignedArray("Kokkos-aligned Array",1);
      if (reinterpret_cast<uintptr_t>(alignedArray.data())%alignment != 0) {
        die("ERROR: Array data does not satisfy Kokkos memory alignment");
      }
      void *direct = yakl::alloc_device(1,"Kokkos-aligned direct allocation");
      if (reinterpret_cast<uintptr_t>(direct)%alignment != 0) {
        die("ERROR: direct global pool allocation does not satisfy Kokkos memory alignment");
      }
      yakl::free_device(direct,"Kokkos-aligned direct allocation");
      size_t const initialAllocs = yakl::get_yakl_instance().num_device_allocations.load();
      int constexpr numThreads = 8;
      int constexpr numCycles  = 500;
      std::atomic<int> ready(0);
      std::atomic<bool> go(false);
      std::atomic<bool> badAlignment(false);
      std::vector<std::thread> threads;
      for (int thread=0; thread < numThreads; thread++) {
        threads.emplace_back([&,thread] {
          ready.fetch_add(1,std::memory_order_release);
          while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
          for (int cycle=0; cycle < numCycles; cycle++) {
            void *ptr = yakl::alloc_device(static_cast<size_t>((cycle+thread)%513)+1,"concurrent global allocation");
            if (reinterpret_cast<uintptr_t>(ptr)%alignment != 0) badAlignment.store(true,std::memory_order_relaxed);
            if ((cycle+thread)%11 == 0) std::this_thread::yield();
            yakl::free_device(ptr,"concurrent global allocation");
          }
        });
      }
      while (ready.load(std::memory_order_acquire) != numThreads) std::this_thread::yield();
      go.store(true,std::memory_order_release);
      for (auto &thread : threads) thread.join();
      if (badAlignment.load() || yakl::get_yakl_instance().num_device_allocations.load() != initialAllocs) {
        die("ERROR: concurrent global pool operations corrupted alignment or allocation accounting");
      }
    }

    {
      size_t constexpr alignment = yakl::LinearAllocator::requiredAlignment;
      yakl::InitConfig config;
      if (config.get_pool_setting() != yakl::InitConfig::PoolSetting::Default || config.get_pool_enabled() ||
          config.get_pool_size_mb() != 0 ||
          config.get_pool_block_bytes() != 4096) {
        die("ERROR: InitConfig defaults are incorrect");
      }
      config = config.set_pool_enabled(true).set_pool_size_mb(37).set_pool_block_bytes(4*alignment);
      if (config.get_pool_setting() != yakl::InitConfig::PoolSetting::Enabled || ! config.get_pool_enabled() ||
          config.get_pool_size_mb() != 37 ||
          config.get_pool_block_bytes() != 4*alignment) {
        die("ERROR: InitConfig setters are incorrect");
      }
      config = config.set_pool_enabled(false);
      if (config.get_pool_setting() != yakl::InitConfig::PoolSetting::Disabled || config.get_pool_enabled()) {
        die("ERROR: InitConfig did not retain an explicit disabled setting");
      }
    }

    // Force two timer records to share a hash and verify that labels and parent/child links remain distinct.
    {
      yakl::Toney collisionTimer;
      collisionTimer.start("first label");
      collisionTimer.stop ("first label");
      collisionTimer.timers[0].label_hash = collisionTimer.hasher("second label");
      collisionTimer.start("first label");
      collisionTimer.start("second label");
      collisionTimer.stop ("second label");
      collisionTimer.stop ("first label");
      int const first  = collisionTimer.get_timer_id("first label");
      int const second = collisionTimer.get_timer_id("second label");
      if (first == second || collisionTimer.timers.size() != 2 || collisionTimer.get_count("first label") != 2 ||
          collisionTimer.get_count("second label") != 1 || collisionTimer.timers[first].child_indices.size() != 1 ||
          collisionTimer.timers[first].child_indices[0] != second) {
        die("ERROR: timer labels or nesting were merged by a hash collision");
      }
    }
    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif
  return 0;
}
