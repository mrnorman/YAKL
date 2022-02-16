
// #pragma once is purposefully omitted here because it needs to be included twice: once in each namespace: c and fortran

////////////////////////////////////////////////
// Convenience functions to handle the indexing
////////////////////////////////////////////////
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<1,simple> const &bnd , int const i ) {
  int ind[1];
  bnd.unpackIndices( i , ind );
  f(ind[0]);
}
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<2,simple> const &bnd , int const i ) {
  int ind[2];
  bnd.unpackIndices( i , ind );
  f(ind[0],ind[1]);
}
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<3,simple> const &bnd , int const i ) {
  int ind[3];
  bnd.unpackIndices( i , ind );
  f(ind[0],ind[1],ind[2]);
}
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<4,simple> const &bnd , int const i ) {
  int ind[4];
  bnd.unpackIndices( i , ind );
  f(ind[0],ind[1],ind[2],ind[3]);
}
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<5,simple> const &bnd , int const i ) {
  int ind[5];
  bnd.unpackIndices( i , ind );
  f(ind[0],ind[1],ind[2],ind[3],ind[4]);
}
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<6,simple> const &bnd , int const i ) {
  int ind[6];
  bnd.unpackIndices( i , ind );
  f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5]);
}
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<7,simple> const &bnd , int const i ) {
  int ind[7];
  bnd.unpackIndices( i , ind );
  f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6]);
}
template <class F, bool simple> YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<8,simple> const &bnd , int const i ) {
  int ind[8];
  bnd.unpackIndices( i , ind );
  f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6],ind[7]);
}



////////////////////////////////////////////////
// HARDWARE BACKENDS FOR KERNEL LAUNCHING
////////////////////////////////////////////////
#ifdef YAKL_ARCH_CUDA
  template <class F, int N, bool simple, int VecLen> __global__ __launch_bounds__(VecLen)
  void cudaKernelVal( Bounds<N,simple> bounds , F f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bounds.nIter) {
      callFunctor( f , bounds , i );
    }
  }

  template <class F, int N, bool simple, int VecLen> __global__ __launch_bounds__(VecLen)
  void cudaKernelRef( Bounds<N,simple> bounds , F const &f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bounds.nIter) {
      callFunctor( f , bounds , i );
    }
  }

  template<class F , int N , bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN ,
           typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0>
  void parallel_for_cuda( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
    cudaKernelVal <<< (unsigned int) (bounds.nIter-1)/VecLen+1 , VecLen >>> ( bounds , f , config );
    check_last_error();
  }

  template<class F , int N , bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN ,
           typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0>
  void parallel_for_cuda( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
    F *fp = (F *) functorBuffer;
    cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
    check_last_error();
    cudaKernelRef <<< (unsigned int) (bounds.nIter-1)/VecLen+1 , VecLen >>> ( bounds , *fp , config );
    check_last_error();
  }
#endif



#ifdef YAKL_ARCH_HIP
  template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN> __global__ __launch_bounds__(VecLen)
  void hipKernel( Bounds<N,simple> bounds , F f , LaunchConfig<VecLen> config = LaunchConfig<>()) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bounds.nIter) {
      callFunctor( f , bounds , i );
    }
  }

  template<class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN>
  void parallel_for_hip( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
    hipLaunchKernelGGL( hipKernel , dim3((bounds.nIter-1)/VecLen+1) , dim3(VecLen) ,
                        (std::uint32_t) 0 , (hipStream_t) 0 , bounds , f , config );
    check_last_error();
  }
#endif



#ifdef YAKL_ARCH_SYCL
  template<class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN>
  void parallel_for_sycl( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
    if constexpr (sycl::is_device_copyable<F>::value) {
      sycl_default_stream().parallel_for( sycl::range<1>(bounds.nIter) , [=] (sycl::id<1> i) {
        callFunctor( f , bounds , i );
      });
      sycl_default_stream().wait();
    } else {
      F *fp = (F *) functorBuffer;
      sycl_default_stream().memcpy(fp, &f, sizeof(F));
      sycl_default_stream().parallel_for( sycl::range<1>(bounds.nIter) , [=] (sycl::id<1> i) {
        callFunctor( *fp , bounds , i );
      });
      sycl_default_stream().wait();
    }

    check_last_error();
  }
#endif



template <class F> inline void parallel_for_cpu_serial( LBnd &bnd , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for
  #endif
  for (int i0 = bnd.l; i0 < (int) (bnd.l+(bnd.u-bnd.l+1)); i0+=bnd.s) {
    f( i0 );
  }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<1,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
    f( i0 );
  }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<2,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd collapse(2) 
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for collapse(2) 
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
  for (int i1 = bounds.lbounds[1]; i1 < (int) (bounds.lbounds[1]+bounds.dims[1]*bounds.strides[1]); i1+=bounds.strides[1]) {
    f( i0 , i1 );
  } }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<3,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd collapse(3)
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for collapse(3)
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
  for (int i1 = bounds.lbounds[1]; i1 < (int) (bounds.lbounds[1]+bounds.dims[1]*bounds.strides[1]); i1+=bounds.strides[1]) {
  for (int i2 = bounds.lbounds[2]; i2 < (int) (bounds.lbounds[2]+bounds.dims[2]*bounds.strides[2]); i2+=bounds.strides[2]) {
    f( i0 , i1 , i2 );
  } } }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<4,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd collapse(4)
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for collapse(4)
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
  for (int i1 = bounds.lbounds[1]; i1 < (int) (bounds.lbounds[1]+bounds.dims[1]*bounds.strides[1]); i1+=bounds.strides[1]) {
  for (int i2 = bounds.lbounds[2]; i2 < (int) (bounds.lbounds[2]+bounds.dims[2]*bounds.strides[2]); i2+=bounds.strides[2]) {
  for (int i3 = bounds.lbounds[3]; i3 < (int) (bounds.lbounds[3]+bounds.dims[3]*bounds.strides[3]); i3+=bounds.strides[3]) {
    f( i0 , i1 , i2 , i3 );
  } } } }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<5,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd collapse(5)
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for collapse(5)
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
  for (int i1 = bounds.lbounds[1]; i1 < (int) (bounds.lbounds[1]+bounds.dims[1]*bounds.strides[1]); i1+=bounds.strides[1]) {
  for (int i2 = bounds.lbounds[2]; i2 < (int) (bounds.lbounds[2]+bounds.dims[2]*bounds.strides[2]); i2+=bounds.strides[2]) {
  for (int i3 = bounds.lbounds[3]; i3 < (int) (bounds.lbounds[3]+bounds.dims[3]*bounds.strides[3]); i3+=bounds.strides[3]) {
  for (int i4 = bounds.lbounds[4]; i4 < (int) (bounds.lbounds[4]+bounds.dims[4]*bounds.strides[4]); i4+=bounds.strides[4]) {
    f( i0 , i1 , i2 , i3 , i4 );
  } } } } }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<6,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd collapse(6)
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for collapse(6)
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
  for (int i1 = bounds.lbounds[1]; i1 < (int) (bounds.lbounds[1]+bounds.dims[1]*bounds.strides[1]); i1+=bounds.strides[1]) {
  for (int i2 = bounds.lbounds[2]; i2 < (int) (bounds.lbounds[2]+bounds.dims[2]*bounds.strides[2]); i2+=bounds.strides[2]) {
  for (int i3 = bounds.lbounds[3]; i3 < (int) (bounds.lbounds[3]+bounds.dims[3]*bounds.strides[3]); i3+=bounds.strides[3]) {
  for (int i4 = bounds.lbounds[4]; i4 < (int) (bounds.lbounds[4]+bounds.dims[4]*bounds.strides[4]); i4+=bounds.strides[4]) {
  for (int i5 = bounds.lbounds[5]; i5 < (int) (bounds.lbounds[5]+bounds.dims[5]*bounds.strides[5]); i5+=bounds.strides[5]) {
    f( i0 , i1 , i2 , i3 , i4 , i5 );
  } } } } } }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<7,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd collapse(7)
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for collapse(7)
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
  for (int i1 = bounds.lbounds[1]; i1 < (int) (bounds.lbounds[1]+bounds.dims[1]*bounds.strides[1]); i1+=bounds.strides[1]) {
  for (int i2 = bounds.lbounds[2]; i2 < (int) (bounds.lbounds[2]+bounds.dims[2]*bounds.strides[2]); i2+=bounds.strides[2]) {
  for (int i3 = bounds.lbounds[3]; i3 < (int) (bounds.lbounds[3]+bounds.dims[3]*bounds.strides[3]); i3+=bounds.strides[3]) {
  for (int i4 = bounds.lbounds[4]; i4 < (int) (bounds.lbounds[4]+bounds.dims[4]*bounds.strides[4]); i4+=bounds.strides[4]) {
  for (int i5 = bounds.lbounds[5]; i5 < (int) (bounds.lbounds[5]+bounds.dims[5]*bounds.strides[5]); i5+=bounds.strides[5]) {
  for (int i6 = bounds.lbounds[6]; i6 < (int) (bounds.lbounds[6]+bounds.dims[6]*bounds.strides[6]); i6+=bounds.strides[6]) {
    f( i0 , i1 , i2 , i3 , i4 , i5 , i6 );
  } } } } } } }
}
template <class F, bool simple> inline void parallel_for_cpu_serial( Bounds<8,simple> const &bounds , F const &f ) {
  #ifdef YAKL_ARCH_OPENMP45
    #pragma omp target teams distribute parallel for simd collapse(8)
  #endif
  #ifdef YAKL_ARCH_OPENMP
    #pragma omp parallel for collapse(8)
  #endif
  for (int i0 = bounds.lbounds[0]; i0 < (int) (bounds.lbounds[0]+bounds.dims[0]*bounds.strides[0]); i0+=bounds.strides[0]) {
  for (int i1 = bounds.lbounds[1]; i1 < (int) (bounds.lbounds[1]+bounds.dims[1]*bounds.strides[1]); i1+=bounds.strides[1]) {
  for (int i2 = bounds.lbounds[2]; i2 < (int) (bounds.lbounds[2]+bounds.dims[2]*bounds.strides[2]); i2+=bounds.strides[2]) {
  for (int i3 = bounds.lbounds[3]; i3 < (int) (bounds.lbounds[3]+bounds.dims[3]*bounds.strides[3]); i3+=bounds.strides[3]) {
  for (int i4 = bounds.lbounds[4]; i4 < (int) (bounds.lbounds[4]+bounds.dims[4]*bounds.strides[4]); i4+=bounds.strides[4]) {
  for (int i5 = bounds.lbounds[5]; i5 < (int) (bounds.lbounds[5]+bounds.dims[5]*bounds.strides[5]); i5+=bounds.strides[5]) {
  for (int i6 = bounds.lbounds[6]; i6 < (int) (bounds.lbounds[6]+bounds.dims[6]*bounds.strides[6]); i6+=bounds.strides[6]) {
  for (int i7 = bounds.lbounds[7]; i7 < (int) (bounds.lbounds[7]+bounds.dims[7]*bounds.strides[7]); i7+=bounds.strides[7]) {
    f( i0 , i1 , i2 , i3 , i4 , i5 , i6 , i7 );
  } } } } } } } }
}



// Bounds class, No label
// This serves as the template, which all other user-level functions route into
template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN>
inline void parallel_for( Bounds<N,simple> const &bounds , F const &f ,
                          LaunchConfig<VecLen> config = LaunchConfig<>() ) {
  #ifdef YAKL_ARCH_CUDA
    parallel_for_cuda( bounds , f , config );
  #elif defined(YAKL_ARCH_HIP)
    parallel_for_hip ( bounds , f , config );
  #elif defined(YAKL_ARCH_SYCL)
    parallel_for_sycl( bounds , f , config );
  #else
    parallel_for_cpu_serial( bounds , f );
  #endif

  #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
    fence();
  #endif
}

// Bounds class, Label
template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN>
inline void parallel_for( char const * str , Bounds<N,simple> const &bounds , F const &f ,
                          LaunchConfig<VecLen> config = LaunchConfig<>() ) {
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePushA(str);
  #endif
  #ifdef YAKL_AUTO_PROFILE
    timer_start(str);
  #endif

  parallel_for( bounds , f , config );

  #ifdef YAKL_AUTO_PROFILE
    timer_stop(str);
  #endif
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePop();
  #endif
}

// Single bound or integer, no label
// Since "bnd" is accepted by value, integers will be accepted as well
template <class F, int VecLen=YAKL_DEFAULT_VECTOR_LEN>
inline void parallel_for( LBnd bnd , F const &f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_for( Bounds<1,true>(bnd.to_scalar()) , f , config );
  } else {
    parallel_for( Bounds<1,false>(bnd) , f , config );
  }
}

// Single bound or integer, label
// Since "bnd" is accepted by value, integers will be accepted as well
template <class F, int VecLen=YAKL_DEFAULT_VECTOR_LEN>
inline void parallel_for( char const * str , LBnd bnd , F const &f , LaunchConfig<VecLen> config = LaunchConfig<>() ) {
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePushA(str);
  #endif
  #ifdef YAKL_AUTO_PROFILE
    timer_start(str);
  #endif

  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_for( Bounds<1,true>(bnd.to_scalar()) , f , config );
  } else {
    parallel_for( Bounds<1,false>(bnd) , f , config );
  }

  #ifdef YAKL_AUTO_PROFILE
    timer_stop(str);
  #endif
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePop();
  #endif
}

