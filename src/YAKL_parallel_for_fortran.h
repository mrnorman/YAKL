
#pragma once

namespace fortran {

  #include "YAKL_Bounds_fortran.h"

  #include "YAKL_parallel_for_common.h"

  template <class F> inline void parallel_for_cpu_serial( int ubnd , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for
    #endif
    for (int i0 = 1; i0 < ubnd; i0++) {
      f( i0 );
    }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<1,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
      f( i0 );
    }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<2,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd collapse(2)
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(2)
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
    for (int i1 = 1; i1 <= bounds.dims[1]; i1++) {
      f( i0 , i1 );
    } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<3,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd collapse(3)
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(3)
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
    for (int i1 = 1; i1 <= bounds.dims[1]; i1++) {
    for (int i2 = 1; i2 <= bounds.dims[2]; i2++) {
      f( i0 , i1 , i2 );
    } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<4,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd collapse(4)
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(4)
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
    for (int i1 = 1; i1 <= bounds.dims[1]; i1++) {
    for (int i2 = 1; i2 <= bounds.dims[2]; i2++) {
    for (int i3 = 1; i3 <= bounds.dims[3]; i3++) {
      f( i0 , i1 , i2 , i3 );
    } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<5,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd collapse(5)
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(5)
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
    for (int i1 = 1; i1 <= bounds.dims[1]; i1++) {
    for (int i2 = 1; i2 <= bounds.dims[2]; i2++) {
    for (int i3 = 1; i3 <= bounds.dims[3]; i3++) {
    for (int i4 = 1; i4 <= bounds.dims[4]; i4++) {
      f( i0 , i1 , i2 , i3 , i4 );
    } } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<6,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd collapse(6)
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(6)
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
    for (int i1 = 1; i1 <= bounds.dims[1]; i1++) {
    for (int i2 = 1; i2 <= bounds.dims[2]; i2++) {
    for (int i3 = 1; i3 <= bounds.dims[3]; i3++) {
    for (int i4 = 1; i4 <= bounds.dims[4]; i4++) {
    for (int i5 = 1; i5 <= bounds.dims[5]; i5++) {
      f( i0 , i1 , i2 , i3 , i4 , i5 );
    } } } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<7,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd collapse(7)
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(7)
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
    for (int i1 = 1; i1 <= bounds.dims[1]; i1++) {
    for (int i2 = 1; i2 <= bounds.dims[2]; i2++) {
    for (int i3 = 1; i3 <= bounds.dims[3]; i3++) {
    for (int i4 = 1; i4 <= bounds.dims[4]; i4++) {
    for (int i5 = 1; i5 <= bounds.dims[5]; i5++) {
    for (int i6 = 1; i6 <= bounds.dims[6]; i6++) {
      f( i0 , i1 , i2 , i3 , i4 , i5 , i6 );
    } } } } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<8,true> const &bounds , F const &f ) {
    #ifdef YAKL_ARCH_OPENMP45
      #pragma omp target teams distribute parallel for simd collapse(8)
    #endif
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(8)
    #endif
    for (int i0 = 1; i0 <= bounds.dims[0]; i0++) {
    for (int i1 = 1; i1 <= bounds.dims[1]; i1++) {
    for (int i2 = 1; i2 <= bounds.dims[2]; i2++) {
    for (int i3 = 1; i3 <= bounds.dims[3]; i3++) {
    for (int i4 = 1; i4 <= bounds.dims[4]; i4++) {
    for (int i5 = 1; i5 <= bounds.dims[5]; i5++) {
    for (int i6 = 1; i6 <= bounds.dims[6]; i6++) {
    for (int i7 = 1; i7 <= bounds.dims[7]; i7++) {
      f( i0 , i1 , i2 , i3 , i4 , i5 , i6 , i7 );
    } } } } } } } }
  }



  ////////////////////////////////////////////////
  // MAIN USER-LEVEL FUNCTIONS
  ////////////////////////////////////////////////
  // Single bound or integer, no label
  // Since "bnd" is accepted by value, integers will be accepted as well
  template <class F> inline void parallel_for( LBnd bnd , F const &f , int vectorSize = 128 ) {
    if (bnd.l == 1 && bnd.s == 1) {
      parallel_for( Bounds<1,true>(bnd.to_scalar()) , f , vectorSize );
    } else {
      parallel_for( Bounds<1,false>(bnd) , f , vectorSize );
    }
  }

  // Single bound or integer, label
  // Since "bnd" is accepted by value, integers will be accepted as well
  template <class F> inline void parallel_for( char const * str , LBnd bnd , F const &f , int vectorSize = 128 ) {
    #ifdef YAKL_ARCH_CUDA
      nvtxRangePushA(str);
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_start(str);
    #endif

    if (bnd.l == 1 && bnd.s == 1) {
      parallel_for( Bounds<1,true>(bnd.to_scalar()) , f , vectorSize );
    } else {
      parallel_for( Bounds<1,false>(bnd) , f , vectorSize );
    }

    #ifdef YAKL_AUTO_PROFILE
      timer_stop(str);
    #endif
    #ifdef YAKL_ARCH_CUDA
      nvtxRangePop();
    #endif
  }


}

