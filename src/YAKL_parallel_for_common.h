
// #pragma once is purposefully omitted here because it needs to be included twice: once in each namespace: c and fortran
// Included by YAKL_parallel_for_c.h and YAKL_parallel_for_fortran.h
// Inside the yakl::c and yakl::fortran namespaces

//////////////////////////////////////////////////////////////////////////////////////////////
// Convenience functions to handle the indexing
// Reduces code for the rest of the parallel_for implementations
// Calls the functor for the specified global index ID "i" using the specified loop bounds
//////////////////////////////////////////////////////////////////////////////////////////////
template <class F, bool simple, int N>
KOKKOS_INLINE_FUNCTION void callFunctor(F const &f , Bounds<N,simple> const &bnd , int64_t const i ) {
  int ind[N];
  bnd.unpackIndices( i , ind );
  if constexpr (N == 1 && simple) f(i);
  if constexpr (N == 1 && !simple) f(ind[0]);
  if constexpr (N == 2) f(ind[0],ind[1]);
  if constexpr (N == 3) f(ind[0],ind[1],ind[2]);
  if constexpr (N == 4) f(ind[0],ind[1],ind[2],ind[3]);
  if constexpr (N == 5) f(ind[0],ind[1],ind[2],ind[3],ind[4]);
  if constexpr (N == 6) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5]);
  if constexpr (N == 7) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6]);
  if constexpr (N == 8) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6],ind[7]);
}


#ifdef YAKL_EXPERIMENTAL_HIP_LAUNCHER
  int constexpr VecLen = 256;
  template <class F, int N, bool simple> __global__ __launch_bounds__(VecLen)
  void hipKernel( Bounds<N,simple> bounds , F f) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bounds.nIter) { callFunctor( f , bounds , i ); }
  }

  template<class F, int N, bool simple>
  inline void parallel_for_hip( Bounds<N,simple> const &bounds , F const &f ) {
    hipKernel <<< (unsigned int) ((bounds.nIter-1)/VecLen+1) , VecLen , 0 , 0 >>> ( bounds , f );
  }
#endif


////////////////////////////////////////////////////////////////////////////////////
// parallel_for
////////////////////////////////////////////////////////////////////////////////////
template <class F, int N, bool simple>
inline void parallel_for( std::string str , Bounds<N,simple> const &bounds , F const &f ) {
  // exit early if there is no work to do
  if (bounds.nIter == 0) return;

  // Automatically time (if requested) and add nvtx ranges for easier nvprof / nsight profiling
  #ifdef YAKL_AUTO_PROFILE
    timer_start(str);
  #endif
  #ifdef YAKL_EXPERIMENTAL_HIP_LAUNCHER
    parallel_for_hip( bounds , f );
  #else
    Kokkos::parallel_for( str , bounds.nIter , KOKKOS_LAMBDA (int64_t i) { callFunctor(f,bounds,i); });
  #endif
  #if defined(YAKL_AUTO_FENCE)
    Kokkos::fence();
  #endif
  #ifdef YAKL_AUTO_PROFILE
    timer_stop(str);
  #endif
}

template <class F, int N, bool simple>
inline void parallel_for( Bounds<N,simple> const &bounds , F const &f ) {
  parallel_for( YAKL_AUTO_LABEL() , bounds , f );
}

template <class F>
inline void parallel_for( LBnd bnd , F const &f ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_for( YAKL_AUTO_LABEL() , Bounds<1,true>(bnd.to_scalar()) , f );
  } else {
    parallel_for( YAKL_AUTO_LABEL() , Bounds<1,false>(bnd) , f );
  }
}

template <class F>
inline void parallel_for( std::string str , LBnd bnd , F const &f ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_for( str , Bounds<1,true>(bnd.to_scalar()) , f );
  } else {
    parallel_for( str , Bounds<1,false>(bnd) , f );
  }
}


