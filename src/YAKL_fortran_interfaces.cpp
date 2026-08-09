
#include "YAKL.h"
#include <unordered_set>

// Fortran-facing routines

namespace {
  // Init and finalize change the lifetime of every wrapper resource, while ordinary calls only need that lifetime to remain
  // stable. Allocation ownership is independent metadata and is locked only while the set itself is accessed.
  std::shared_mutex         gator_lifecycle_mutex;
  std::mutex                gator_allocations_mutex;
  std::unordered_set<void*> gator_allocations;
  bool                      gator_initialized = false;
  bool                      gator_owns_kokkos  = false;
  bool                      gator_owns_yakl    = false;
}

/** @brief Fortran YAKL initialization call */
extern "C" void gatorInit() {
  std::unique_lock<std::shared_mutex> lock(gator_lifecycle_mutex);
  if (gator_initialized) Kokkos::abort("ERROR: gatorInit called more than once without gatorFinalize");
  gator_owns_kokkos = ! Kokkos::is_initialized();
  if (gator_owns_kokkos) Kokkos::initialize();
  gator_owns_yakl = ! yakl::get_yakl_instance().is_initialized();
  if (gator_owns_yakl) yakl::init();
  gator_initialized = true;
}

/** @brief Fortran YAKL finalization call */
extern "C" void gatorFinalize() {
  std::unique_lock<std::shared_mutex> lock(gator_lifecycle_mutex);
  if (!gator_initialized) Kokkos::abort("ERROR: gatorFinalize called without a matching gatorInit");
  if (!gator_allocations.empty()) Kokkos::abort("ERROR: gatorFinalize called with live Fortran allocations");
  if (gator_owns_yakl) yakl::finalize();
  if (gator_owns_kokkos) Kokkos::finalize();
  gator_initialized = false;
  gator_owns_kokkos  = false;
  gator_owns_yakl    = false;
}

/** @brief Fortran YAKL device allocation call */
extern "C" void* gatorAllocate( size_t bytes ) {
  if (bytes == 0) Kokkos::abort("ERROR: gatorAllocate received a zero byte count");
  std::shared_lock<std::shared_mutex> lifecycleLock(gator_lifecycle_mutex);
  if (!Kokkos::is_initialized() || !yakl::get_yakl_instance().is_initialized()) {
    Kokkos::abort("ERROR: gatorAllocate called before initialization");
  }
  void *ptr = yakl::alloc_device( bytes , "gatorAllocate");
  {
    std::lock_guard<std::mutex> allocationsLock(gator_allocations_mutex);
    if (!gator_allocations.insert(ptr).second) Kokkos::abort("ERROR: gatorAllocate returned a duplicate pointer");
  }
  return ptr;
}

/** @brief Fortran YAKL device free call */
extern "C" void gatorDeallocate( void *ptr ) {
  if (ptr == nullptr) Kokkos::abort("ERROR: gatorDeallocate received a null pointer");
  std::shared_lock<std::shared_mutex> lifecycleLock(gator_lifecycle_mutex);
  if (!Kokkos::is_initialized() || !yakl::get_yakl_instance().is_initialized()) {
    Kokkos::abort("ERROR: gatorDeallocate called before initialization");
  }
  {
    std::lock_guard<std::mutex> allocationsLock(gator_allocations_mutex);
    auto const allocation = gator_allocations.find(ptr);
    if (allocation == gator_allocations.end()) {
      Kokkos::abort("ERROR: gatorDeallocate requires the base pointer returned by gatorAllocate");
    }
    gator_allocations.erase(allocation);
  }
  yakl::free_device( ptr , "gatorDeallocate");
}
