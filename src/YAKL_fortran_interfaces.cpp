
#include "YAKL.h"

// Fortran-facing routines

/** @brief Fortran YAKL initialization call */
extern "C" void gatorInit() {
  if (Kokkos::is_initialized()) Kokkos::abort("ERROR: gatorInit called after Kokkos was already initialized");
  Kokkos::initialize();
  yakl::init();
}

/** @brief Fortran YAKL finalization call */
extern "C" void gatorFinalize() {
  if (!Kokkos::is_initialized()) Kokkos::abort("ERROR: gatorFinalize called before Kokkos initialization");
  if (!yakl::get_yakl_instance().is_initialized()) Kokkos::abort("ERROR: gatorFinalize called before YAKL initialization");
  yakl::finalize();
  Kokkos::finalize();
}

/** @brief Fortran YAKL device allocation call */
extern "C" void* gatorAllocate( size_t bytes ) {
  if (bytes == 0) Kokkos::abort("ERROR: gatorAllocate received a zero byte count");
  if (!Kokkos::is_initialized() || !yakl::get_yakl_instance().is_initialized()) {
    Kokkos::abort("ERROR: gatorAllocate called before initialization");
  }
  return yakl::alloc_device( bytes , "gatorAllocate");
}

/** @brief Fortran YAKL device free call */
extern "C" void gatorDeallocate( void *ptr ) {
  if (ptr == nullptr) Kokkos::abort("ERROR: gatorDeallocate received a null pointer");
  if (!Kokkos::is_initialized() || !yakl::get_yakl_instance().is_initialized()) {
    Kokkos::abort("ERROR: gatorDeallocate called before initialization");
  }
  yakl::free_device( ptr , "gatorDeallocate");
}

