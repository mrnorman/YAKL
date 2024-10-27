
#include "YAKL.h"

// Fortran-facing routines

/** @brief Fortran YAKL initialization call */
extern "C" void gatorInit() {
  Kokkos::initialize();
  yakl::init();
}

/** @brief Fortran YAKL finalization call */
extern "C" void gatorFinalize() {
  yakl::finalize();
  Kokkos::finalize();
}

/** @brief Fortran YAKL device allocation call */
extern "C" void* gatorAllocate( size_t bytes ) {
  return yakl::alloc_device( bytes , "gatorAllocate");
}

/** @brief Fortran YAKL device free call */
extern "C" void gatorDeallocate( void *ptr ) {
  yakl::free_device( ptr , "gatorDeallocate");
}


