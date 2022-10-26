
// #pragma once is purposefully omitted here because it needs to be included twice: once in each namespace: c and fortran
// Included by YAKL_parallel_for_c.h and YAKL_parallel_for_fortran.h
// Inside the yakl::c and yakl::fortran namespaces

//////////////////////////////////////////////////////////////////////////////////////////////
// Convenience functions to handle the indexing
// Reduces code for the rest of the parallel_for implementations
// Calls the functor for the specified global index ID "i" using the specified loop bounds
//////////////////////////////////////////////////////////////////////////////////////////////
template <class F, bool simple, int N>
YAKL_DEVICE_INLINE void callFunctor(F const &f , Bounds<N,simple> const &bnd , int const i ) {
  int ind[N];
  bnd.unpackIndices( i , ind );
  if constexpr (N == 1) f(ind[0]);
  if constexpr (N == 2) f(ind[0],ind[1]);
  if constexpr (N == 3) f(ind[0],ind[1],ind[2]);
  if constexpr (N == 4) f(ind[0],ind[1],ind[2],ind[3]);
  if constexpr (N == 5) f(ind[0],ind[1],ind[2],ind[3],ind[4]);
  if constexpr (N == 6) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5]);
  if constexpr (N == 7) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6]);
  if constexpr (N == 8) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6],ind[7]);
}


template <class F, bool simple, int N>
YAKL_DEVICE_INLINE void callFunctorOuter(F const &f , Bounds<N,simple> const &bnd , int const i ,
                                         InnerHandler handler ) {
  int ind[N];
  bnd.unpackIndices( i , ind );
  if constexpr (N == 1) f(ind[0],handler);
  if constexpr (N == 2) f(ind[0],ind[1],handler);
  if constexpr (N == 3) f(ind[0],ind[1],ind[2],handler);
  if constexpr (N == 4) f(ind[0],ind[1],ind[2],ind[3],handler);
  if constexpr (N == 5) f(ind[0],ind[1],ind[2],ind[3],ind[4],handler);
  if constexpr (N == 6) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],handler);
  if constexpr (N == 7) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6],handler);
  if constexpr (N == 8) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6],ind[7],handler);
}


////////////////////////////////////////////////
// HARDWARE BACKENDS FOR KERNEL LAUNCHING
////////////////////////////////////////////////
#ifdef YAKL_ARCH_CUDA
  // CUDA has a limit on the parameter space for a kernel launch. When it's exceeded, then the kernel
  // needs to be launched by reference from device memory (i.e., "cudaKernelRef").
  // otherwise, it needs to be launched by value (i.e., "cudaKernelVal")
  // A kernel launch will get the global ID and then use the callFunctor intermediate to tranform that
  // ID into a set of indices for multiple loops and then call the functor with those indices
  // The __launch_bounds__ matches the number of threads used to launch the kernels so that the compiler
  // does not compile for more threads per block than the user plans to use. This is for optimal register
  // usage and reduced register spilling.
  template <class F, int N, bool simple, int VecLen, bool B4B> __global__ __launch_bounds__(VecLen)
  void cudaKernelVal( Bounds<N,simple> bounds , F f , LaunchConfig<VecLen,B4B> config ) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bounds.nIter) {
      callFunctor( f , bounds , i );
    }
  }

  template <class F, int N, bool simple, int VecLen, bool B4B> __global__ __launch_bounds__(VecLen)
  void cudaKernelRef( Bounds<N,simple> bounds , F const &f , LaunchConfig<VecLen,B4B> config ) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bounds.nIter) {
      callFunctor( f , bounds , i );
    }
  }

  // If the functor is small enough, then launch it like normal
  template<class F , int N , bool simple, int VecLen, bool B4B>
  void parallel_for_cuda( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen,B4B> config ) {
    auto stream = config.get_stream();
    if constexpr (sizeof(F) <= 3800) {
      cudaKernelVal <<< (unsigned int) (bounds.nIter-1)/VecLen+1 , VecLen , 0 , stream.get_real_stream() >>> ( bounds , f , config );
      check_last_error();
    } else {
      F *fp = (F *) alloc_device(sizeof(F),"functor_buffer");
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice,stream.get_real_stream());
      check_last_error();
      cudaKernelRef <<< (unsigned int) (bounds.nIter-1)/VecLen+1 , VecLen , 0 , stream.get_real_stream() >>> ( bounds , *fp , config );
      check_last_error();
      #ifdef YAKL_ENABLE_STREAMS
        if (use_pool() && device_allocators_are_default) {
          pool.free_with_event_dependencies( fp , {record_event(stream)} , "functor_buffer" );
        } else          {
          free_device( fp , "functor_buffer" );
        }
      #else
        free_device( fp , "functor_buffer" );
      #endif
      check_last_error();
    }
  }



  template <class F, int N, bool simple, int VecLen, bool B4B> __global__ __launch_bounds__(VecLen)
  void cudaKernelOuterVal( Bounds<N,simple> bounds , F f , LaunchConfig<VecLen,B4B> config ,
                           InnerHandler handler ) {
    callFunctorOuter( f , bounds , blockIdx.x , handler );
  }

  template <class F, int N, bool simple, int VecLen, bool B4B> __global__ __launch_bounds__(VecLen)
  void cudaKernelOuterRef( Bounds<N,simple> bounds , F const &f , LaunchConfig<VecLen,B4B> config ,
                           InnerHandler handler ) {
    callFunctorOuter( f , bounds , blockIdx.x , handler );
  }

  template<class F , int N , bool simple, int VecLen, bool B4B>
  void parallel_outer_cuda( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen,B4B> config ) {
    auto stream = config.get_stream();
    if constexpr (sizeof(F) <= 3800) {
      cudaKernelOuterVal <<< (unsigned int) bounds.nIter , config.inner_size , 0 , stream.get_real_stream() >>> ( bounds , f , config , InnerHandler() );
      check_last_error();
    } else {
      F *fp = (F *) alloc_device(sizeof(F),"functor_buffer");
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice,stream.get_real_stream());
      check_last_error();
      cudaKernelOuterRef <<< (unsigned int) bounds.nIter , config.inner_size , 0 , stream.get_real_stream() >>> ( bounds , *fp , config , InnerHandler() );
      check_last_error();
      #ifdef YAKL_ENABLE_STREAMS
        if (use_pool() && device_allocators_are_default) {
          pool.free_with_event_dependencies( fp , {record_event(stream)} , "functor_buffer" );
        } else {
          free_device( fp , "functor_buffer" );
        }
      #else
        free_device( fp , "functor_buffer" );
      #endif
      check_last_error();
    }
  }



  template <class F, int N, bool simple>
  YAKL_INLINE void parallel_inner_cuda( Bounds<N,simple> bounds , F const &f ) {
    #if YAKL_CURRENTLY_ON_DEVICE()
      if (threadIdx.x < bounds.nIter) callFunctor( f , bounds , threadIdx.x );
    #else
      // Avoid not used warning
      (void) f;
    #endif
  }



  template <class F>
  YAKL_INLINE void single_inner_cuda( F const &f ) {
    #if YAKL_CURRENTLY_ON_DEVICE()
      if (threadIdx.x == 0) f();
    #else
      // Avoid not used warning
      (void) f;
    #endif
  }
#endif



#ifdef YAKL_ARCH_HIP
  // A kernel launch will get the global ID and then use the callFunctor intermediate to tranform that
  // ID into a set of indices for multiple loops and then call the functor with those indices
  // The __launch_bounds__ matches the number of threads used to launch the kernels so that the compiler
  // does not compile for more threads per block than the user plans to use. This is for optimal register
  // usage and reduced register spilling.
  template <class F, int N, bool simple, int VecLen, bool B4B> __global__ __launch_bounds__(VecLen)
  void hipKernel( Bounds<N,simple> bounds , F f , LaunchConfig<VecLen,B4B> config) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bounds.nIter) {
      callFunctor( f , bounds , i );
    }
  }

  template<class F, int N, bool simple, int VecLen, bool B4B>
  void parallel_for_hip( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen,B4B> config ) {
    hipKernel <<< (unsigned int) (bounds.nIter-1)/VecLen+1 , VecLen , 0 , config.get_stream().get_real_stream() >>> ( bounds , f , config );
    check_last_error();
  }



  template <class F, int N, bool simple, int VecLen, bool B4B> __global__ __launch_bounds__(VecLen)
  void hipOuterKernel( Bounds<N,simple> bounds , F f , LaunchConfig<VecLen,B4B> config , InnerHandler handler) {
    callFunctorOuter( f , bounds , blockIdx.x , handler );
  }

  template<class F, int N, bool simple, int VecLen, bool B4B>
  void parallel_outer_hip( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen,B4B> config ) {
    hipOuterKernel <<< (unsigned int) bounds.nIter , config.inner_size , 0 , config.get_stream().get_real_stream() >>> ( bounds , f , config , InnerHandler() );
    check_last_error();
  }



  template <class F, int N, bool simple>
  YAKL_INLINE void parallel_inner_hip( Bounds<N,simple> bounds , F const &f ) {
    #if YAKL_CURRENTLY_ON_DEVICE()
      if (threadIdx.x < bounds.nIter) callFunctor( f , bounds , threadIdx.x );
    #else
      // Avoid not used warning
      (void) f;
    #endif
  }



  template <class F>
  YAKL_INLINE void single_inner_hip( F const &f ) {
    #if YAKL_CURRENTLY_ON_DEVICE()
      if (threadIdx.x == 0) f();
    #else
      // Avoid not used warning
      (void) f;
    #endif
  }
#endif



#ifdef YAKL_ARCH_SYCL
  // Kernels are launched with the SYCL parallel_for routine. 
  // Currently, SYCL must copy this to the device manually and then run from the device
  // Also, there is a violation of dependence wherein the stream must synchronized with a wait()
  // call after launch
  template<class F, int N, bool simple, int VecLen, bool B4B>
  void parallel_for_sycl( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen,B4B> config ) {
    #ifdef SYCL_DEVICE_COPYABLE
      if constexpr (sizeof(F) < 1700) {
        SYCL_Functor_Wrapper sycl_functor_wrapper(f);
        config.get_stream().get_real_stream().parallel_for( sycl::nd_range<1>(((bounds.nIter-1)/VecLen+1)*VecLen,VecLen) , [=] (sycl::nd_item<1> item) {
          if (item.get_global_id(0) < bounds.nIter) {
            callFunctor( sycl_functor_wrapper.get_functor() , bounds , item.get_global_id(0) );
          }
        });
      } else {
        F *fp = (F *) alloc_device(sizeof(F),"functor_buffer");
        auto copyEvent = config.get_stream().get_real_stream().memcpy(fp, &f, sizeof(F));
        config.get_stream().get_real_stream().parallel_for( sycl::nd_range<1>(((bounds.nIter-1)/VecLen+1)*VecLen,VecLen), [=] (sycl::nd_item<1> item) {
          if (item.get_global_id(0) < bounds.nIter) {
            callFunctor( *fp , bounds , item.get_global_id(0) );
          }
        });
        free_device( fp , "functor_buffer" );
        copyEvent.wait();
      }
    #else
      F *fp = (F *) alloc_device(sizeof(F),"functor_buffer");
      auto copyEvent = config.get_stream().get_real_stream().memcpy(fp, &f, sizeof(F));
      config.get_stream().get_real_stream().parallel_for( sycl::nd_range<1>(((bounds.nIter-1)/VecLen+1)*VecLen,VecLen), [=] (sycl::nd_item<1> item) {
        if (item.get_global_id(0) < bounds.nIter) {
          callFunctor( *fp , bounds , item.get_global_id(0) );
        }
      });
      free_device( fp , "functor_buffer" );
      copyEvent.wait();
    #endif

    check_last_error();
  }



  template<class F, int N, bool simple, int VecLen, bool B4B>
  void parallel_outer_sycl( Bounds<N,simple> const &bounds , F const &f , LaunchConfig<VecLen,B4B> config ) {
    #ifdef SYCL_DEVICE_COPYABLE
      if constexpr (sizeof(F) < 1700) {
        SYCL_Functor_Wrapper sycl_functor_wrapper(f);
        config.get_stream().get_real_stream().parallel_for( sycl::nd_range<1>(bounds.nIter*config.inner_size,config.inner_size) ,
                                            [=] (sycl::nd_item<1> item) {
          callFunctorOuter( sycl_functor_wrapper.get_functor() , bounds , item.get_group(0) , InnerHandler(item) );
        });
      } else {
        F *fp = (F *) alloc_device(sizeof(F),"functor_buffer");
        auto copyEvent = config.get_stream().get_real_stream().memcpy(fp, &f, sizeof(F));
        config.get_stream().get_real_stream().parallel_for( sycl::nd_range<1>(bounds.nIter*config.inner_size,config.inner_size) ,
                                            [=] (sycl::nd_item<1> item) {
          callFunctorOuter( *fp , bounds , item.get_group(0) , InnerHandler(item) );
        });
        free_device( fp , "functor_buffer" );
        copyEvent.wait();
      }
    #else
      F *fp = (F *) alloc_device(sizeof(F),"functor_buffer");
      auto copyEvent = config.get_stream().get_real_stream().memcpy(fp, &f, sizeof(F));
      config.get_stream().get_real_stream().parallel_for( sycl::nd_range<1>(bounds.nIter*config.inner_size,config.inner_size) ,
                                          [=] (sycl::nd_item<1> item) {
        callFunctorOuter( *fp , bounds , item.get_group(0) , InnerHandler(item) );
      });
      free_device( fp , "functor_buffer" );
      copyEvent.wait();
    #endif

    check_last_error();
  }



  template<class F, int N, bool simple>
  void parallel_inner_sycl( Bounds<N,simple> const &bounds , F const &f , InnerHandler handler ) {
    #if YAKL_CURRENTLY_ON_DEVICE()
      if (handler.get_item().get_local_id(0) < bounds.nIter) callFunctor( f , bounds , handler.get_item().get_local_id(0) );
    #endif
  }



  template<class F>
  void single_inner_sycl( F const &f , InnerHandler handler ) {
    #if YAKL_CURRENTLY_ON_DEVICE()
      if (handler.get_item().get_local_id(0) == 0) f();
    #endif
  }
#endif


template <bool OUTER> struct DoOuter {};


// These are the CPU routines for parallel_for without a device target. These rely on
// constexpr lower bounds and strides for the compiler to optimize loop arithmetic for
// simple bounds.
// For OMP target offload backend, target teams distribute parallel for simd is used.
// For OMP CPU threading backend, parallel for is used
template <class F, bool simple, int N, bool OUTER=false>
inline void parallel_for_cpu_serial( Bounds<N,simple> const &bounds , F const &f , DoOuter<OUTER> dummy = DoOuter<false>() ) {
  if constexpr (N == 1) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
      if constexpr (OUTER) { f(i0,InnerHandler() ); }
      else                 { f(i0); }
    }
  } else if constexpr (N == 2) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(2)
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
      if constexpr (OUTER) { f(i0,i1,InnerHandler() ); }
      else                 { f(i0,i1); }
    } }
  } else if constexpr (N == 3) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(3)
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
      if constexpr (OUTER) { f(i0,i1,i2,InnerHandler() ); }
      else                 { f(i0,i1,i2); }
    } } }
  } else if constexpr (N == 4) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(4)
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
      if constexpr (OUTER) { f(i0,i1,i2,i3,InnerHandler() ); }
      else                 { f(i0,i1,i2,i3); }
    } } } }
  } else if constexpr (N == 5) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(5)
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
      if constexpr (OUTER) { f(i0,i1,i2,i3,i4,InnerHandler() ); }
      else                 { f(i0,i1,i2,i3,i4); }
    } } } } }
  } else if constexpr (N == 6) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(6)
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
    for (int i5 = bounds.lbound(5); i5 < (int) (bounds.lbound(5)+bounds.dim(5)*bounds.stride(5)); i5+=bounds.stride(5)) {
      if constexpr (OUTER) { f(i0,i1,i2,i3,i4,i5,InnerHandler() ); }
      else                 { f(i0,i1,i2,i3,i4,i5); }
    } } } } } }
  } else if constexpr (N == 7) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(7)
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
    for (int i5 = bounds.lbound(5); i5 < (int) (bounds.lbound(5)+bounds.dim(5)*bounds.stride(5)); i5+=bounds.stride(5)) {
    for (int i6 = bounds.lbound(6); i6 < (int) (bounds.lbound(6)+bounds.dim(6)*bounds.stride(6)); i6+=bounds.stride(6)) {
      if constexpr (OUTER) { f(i0,i1,i2,i3,i4,i5,i6,InnerHandler() ); }
      else                 { f(i0,i1,i2,i3,i4,i5,i6); }
    } } } } } } }
  } else if constexpr (N == 8) {
    #ifdef YAKL_ARCH_OPENMP
      #pragma omp parallel for collapse(8)
    #endif
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
    for (int i5 = bounds.lbound(5); i5 < (int) (bounds.lbound(5)+bounds.dim(5)*bounds.stride(5)); i5+=bounds.stride(5)) {
    for (int i6 = bounds.lbound(6); i6 < (int) (bounds.lbound(6)+bounds.dim(6)*bounds.stride(6)); i6+=bounds.stride(6)) {
    for (int i7 = bounds.lbound(7); i7 < (int) (bounds.lbound(7)+bounds.dim(7)*bounds.stride(7)); i7+=bounds.stride(7)) {
      if constexpr (OUTER) { f(i0,i1,i2,i3,i4,i5,i6,i7,InnerHandler() ); }
      else                 { f(i0,i1,i2,i3,i4,i5,i6,i7); }
    } } } } } } } }
  }
}


template <class F, bool simple, int N>
inline void parallel_inner_cpu_serial( Bounds<N,simple> const &bounds , F const &f ) {
  if constexpr (N == 1) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
      f(i0);
    }
  } else if constexpr (N == 2) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
      f(i0,i1);
    } }
  } else if constexpr (N == 3) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
      f(i0,i1,i2);
    } } }
  } else if constexpr (N == 4) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
      f(i0,i1,i2,i3);
    } } } }
  } else if constexpr (N == 5) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
      f(i0,i1,i2,i3,i4);
    } } } } }
  } else if constexpr (N == 6) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
    for (int i5 = bounds.lbound(5); i5 < (int) (bounds.lbound(5)+bounds.dim(5)*bounds.stride(5)); i5+=bounds.stride(5)) {
      f(i0,i1,i2,i3,i4,i5);
    } } } } } }
  } else if constexpr (N == 7) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
    for (int i5 = bounds.lbound(5); i5 < (int) (bounds.lbound(5)+bounds.dim(5)*bounds.stride(5)); i5+=bounds.stride(5)) {
    for (int i6 = bounds.lbound(6); i6 < (int) (bounds.lbound(6)+bounds.dim(6)*bounds.stride(6)); i6+=bounds.stride(6)) {
      f(i0,i1,i2,i3,i4,i5,i6);
    } } } } } } }
  } else if constexpr (N == 8) {
    for (int i0 = bounds.lbound(0); i0 < (int) (bounds.lbound(0)+bounds.dim(0)*bounds.stride(0)); i0+=bounds.stride(0)) {
    for (int i1 = bounds.lbound(1); i1 < (int) (bounds.lbound(1)+bounds.dim(1)*bounds.stride(1)); i1+=bounds.stride(1)) {
    for (int i2 = bounds.lbound(2); i2 < (int) (bounds.lbound(2)+bounds.dim(2)*bounds.stride(2)); i2+=bounds.stride(2)) {
    for (int i3 = bounds.lbound(3); i3 < (int) (bounds.lbound(3)+bounds.dim(3)*bounds.stride(3)); i3+=bounds.stride(3)) {
    for (int i4 = bounds.lbound(4); i4 < (int) (bounds.lbound(4)+bounds.dim(4)*bounds.stride(4)); i4+=bounds.stride(4)) {
    for (int i5 = bounds.lbound(5); i5 < (int) (bounds.lbound(5)+bounds.dim(5)*bounds.stride(5)); i5+=bounds.stride(5)) {
    for (int i6 = bounds.lbound(6); i6 < (int) (bounds.lbound(6)+bounds.dim(6)*bounds.stride(6)); i6+=bounds.stride(6)) {
    for (int i7 = bounds.lbound(7); i7 < (int) (bounds.lbound(7)+bounds.dim(7)*bounds.stride(7)); i7+=bounds.stride(7)) {
      f(i0,i1,i2,i3,i4,i5,i6,i7);
    } } } } } } } }
  }
}


////////////////////////////////////////////////////////////////////////////////////
// parallel_for
////////////////////////////////////////////////////////////////////////////////////
// Default parameter to config, VecLen, and B4B already specified in YAKL_parallel_for_c.h and YAKL_parallel_for_fortran.h
template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN , bool B4B=false>
inline void parallel_for( char const * str , Bounds<N,simple> const &bounds , F const &f ,
                          LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  #ifdef YAKL_VERBOSE
    verbose_inform(std::string("Launching parallel_for labeled \"")+std::string(str)+std::string("\" with ")+std::to_string(bounds.nIter)+" threads",str);
  #endif
  // Automatically time (if requested) and add nvtx ranges for easier nvprof / nsight profiling
  #ifdef YAKL_AUTO_PROFILE
    timer_start(str);
  #endif
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePushA(str);
  #endif
  #ifdef YAKL_ARCH_HIP
    roctxRangePushA(str);
  #endif

  bool do_b4b = false;
  #ifdef YAKL_B4B
    if constexpr (B4B) do_b4b = true;
  #endif

  if (do_b4b) {
    fence();
    // The if-state ment is redundant, but it shields cpu_serial from compilation if b4b isn't desired
    // For instance, if the lambda is YAKL_DEVICE_LAMBDA, compiling this line will give an error because
    // it isn't available on the host.
    #ifdef YAKL_B4B
      if constexpr (B4B) { parallel_for_cpu_serial( bounds , f ); }
    #endif
  } else {
    #ifdef YAKL_ARCH_CUDA
      parallel_for_cuda( bounds , f , config );
    #elif defined(YAKL_ARCH_HIP)
      parallel_for_hip ( bounds , f , config );
    #elif defined(YAKL_ARCH_SYCL)
      parallel_for_sycl( bounds , f , config );
    #else
      parallel_for_cpu_serial( bounds , f );
    #endif
  }

  #if defined(YAKL_AUTO_FENCE)
    fence();
  #endif

  #ifdef YAKL_ARCH_HIP
    roctxRangePop();
  #endif
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePop();
  #endif
  #ifdef YAKL_AUTO_PROFILE
    timer_stop(str);
  #endif
}

// Default parameter to config, VecLen, and B4B already specified in YAKL_parallel_for_c.h and YAKL_parallel_for_fortran.h
template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B=false>
inline void parallel_for( Bounds<N,simple> const &bounds , F const &f ,
                          LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  parallel_for( "Unlabeled" , bounds , f , config );
}

template <class F, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B=false>
inline void parallel_for( LBnd bnd , F const &f ,
                          LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_for( "Unlabeled" , Bounds<1,true>(bnd.to_scalar()) , f , config );
  } else {
    parallel_for( "Unlabeled" , Bounds<1,false>(bnd) , f , config );
  }
}

template <class F, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B=false>
inline void parallel_for( char const * str , LBnd bnd , F const &f ,
                          LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_for( str , Bounds<1,true>(bnd.to_scalar()) , f , config );
  } else {
    parallel_for( str , Bounds<1,false>(bnd) , f , config );
  }
}


////////////////////////////////////////////////////////////////////////////////////
// parallel_outer
////////////////////////////////////////////////////////////////////////////////////
// Default parameter to config, VecLen, and B4B already specified in YAKL_parallel_for_c.h and YAKL_parallel_for_fortran.h
template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B=false>
inline void parallel_outer( char const * str , Bounds<N,simple> const &bounds , F const &f ,
                            LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  #ifdef YAKL_VERBOSE
    verbose_inform(std::string("Launching parallel_outer labeled \"")+std::string(str)+std::string("\" with ")+std::to_string(bounds.nIter)+
                   std::string(" outer threads and ")+
                   std::to_string(config.inner_size)+std::string(" inner threads"),str);
  #endif
  // Automatically time (if requested) and add nvtx ranges for easier nvprof / nsight profiling
  #ifdef YAKL_AUTO_PROFILE
    timer_start(str);
  #endif
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePushA(str);
  #endif
  #ifdef YAKL_ARCH_HIP
    roctxRangePushA(str);
  #endif

  bool do_b4b = false;
  #ifdef YAKL_B4B
    if constexpr (B4B) do_b4b = true;
  #endif

  if (do_b4b) {
    fence();
    // The if-state ment is redundant, but it shields cpu_serial from compilation if b4b isn't desired
    // For instance, if the lambda is YAKL_DEVICE_LAMBDA, compiling this line will give an error because
    // it isn't available on the host.
    #ifdef YAKL_B4B
      if constexpr (B4B) parallel_for_cpu_serial( bounds , f , DoOuter<true>() );
    #endif
  } else {
    #ifdef YAKL_ARCH_CUDA
      parallel_outer_cuda( bounds , f , config );
    #elif defined(YAKL_ARCH_HIP)
      parallel_outer_hip ( bounds , f , config );
    #elif defined(YAKL_ARCH_SYCL)
      parallel_outer_sycl( bounds , f , config );
    #else
      parallel_for_cpu_serial( bounds , f , DoOuter<true>() );
    #endif
  }

  #if defined(YAKL_AUTO_FENCE)
    fence();
  #endif

  #ifdef YAKL_ARCH_HIP
    roctxRangePop();
  #endif
  #ifdef YAKL_ARCH_CUDA
    nvtxRangePop();
  #endif
  #ifdef YAKL_AUTO_PROFILE
    timer_stop(str);
  #endif
}

// Default parameter to config, VecLen, and B4B already specified in YAKL_parallel_for_c.h and YAKL_parallel_for_fortran.h
template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B=false>
inline void parallel_outer( Bounds<N,simple> const &bounds , F const &f ,
                            LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  parallel_outer( "Unlabeled" , bounds , f , config );
}

template <class F, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B=false>
inline void parallel_outer( LBnd bnd , F const &f ,
                            LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_outer( "Unlabeled" , Bounds<1,true>(bnd.to_scalar()) , f , config );
  } else {
    parallel_outer( "Unlabeled" , Bounds<1,false>(bnd) , f , config );
  }
}

template <class F, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B=false>
inline void parallel_outer( char const * str , LBnd bnd , F const &f ,
                            LaunchConfig<VecLen,B4B> config = LaunchConfig<>() ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_outer( str , Bounds<1,true>(bnd.to_scalar()) , f , config );
  } else {
    parallel_outer( str , Bounds<1,false>(bnd) , f , config );
  }
}


////////////////////////////////////////////////////////////////////////////////////
// parallel_inner
////////////////////////////////////////////////////////////////////////////////////
template <class F, int N, bool simple>
YAKL_INLINE void parallel_inner( Bounds<N,simple> const &bounds , F const &f , InnerHandler handler ) {
  #if YAKL_CURRENTLY_ON_HOST()
    parallel_inner_cpu_serial( bounds , f );
  #else
    #ifdef YAKL_ARCH_CUDA
      parallel_inner_cuda( bounds , f );
    #elif defined(YAKL_ARCH_HIP)
      parallel_inner_hip ( bounds , f );
    #elif defined(YAKL_ARCH_SYCL)
      parallel_inner_sycl( bounds , f , handler );
    #else
      parallel_inner_cpu_serial( bounds , f );
    #endif
  #endif
  #ifdef YAKL_AUTO_FENCE
    fence_inner(handler);
  #endif
}

template <class F>
YAKL_INLINE void parallel_inner( LBnd bnd , F const &f , InnerHandler handler ) {
  if (bnd.l == bnd.default_lbound && bnd.s == 1) {
    parallel_inner( Bounds<1,true>(bnd.to_scalar()) , f , handler );
  } else {
    parallel_inner( Bounds<1,false>(bnd) , f , handler );
  }
}


template <class F>
YAKL_INLINE void single_inner( F const &f , InnerHandler handler ) {
  #if YAKL_CURRENTLY_ON_HOST()
    f();
  #else
    #ifdef YAKL_ARCH_CUDA
      single_inner_cuda( f );
    #elif defined(YAKL_ARCH_HIP)
      single_inner_hip ( f );
    #elif defined(YAKL_ARCH_SYCL)
      single_inner_sycl( f , handler );
    #else
      f();
    #endif
  #endif
  #ifdef YAKL_AUTO_FENCE
    fence_inner(handler);
  #endif
}



