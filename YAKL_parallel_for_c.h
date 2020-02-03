
#pragma once


  #ifdef __USE_CUDA__
    template <class F> __global__ void cudaKernelVal( int n1 , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }

    template <class F> __global__ void cudaKernelRef( int n1 , F const &f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }

    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , F const &f , int vectorSize = 128 ) {
      cudaKernelVal <<< (unsigned int) (n1-1)/vectorSize+1 , vectorSize >>> ( n1 , f );
    }

    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , F const &f , int vectorSize = 128 ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      cudaKernelRef <<< (unsigned int) (n1-1)/vectorSize+1 , vectorSize >>> ( n1 , *fp );
    }
  #endif



  #ifdef __USE_HIP__
    template <class F> __global__ void hipKernel( int n1 , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }

    template<class F> inline void parallel_for_hip( int n1 , F const &f , int vectorSize = 128 ) {
      hipLaunchKernelGGL( hipKernel , dim3((n1-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , f );
    }
  #endif



  template <class F> inline void parallel_for_cpu_serial( int n1 , F const &f ) {
    for (int i=0; i<n1; i++) {
      f(i);
    }
  }



  template <class F> inline void parallel_for( int n1 , F const &f , int vectorSize = 128 ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , f , vectorSize );
    #elif defined(__USE_HIP__)
      parallel_for_hip ( n1 , f , vectorSize );
    #else
      parallel_for_cpu_serial( n1 , f );
    #endif

    #if defined(__AUTO_FENCE__)
      fence();
    #endif
  }



  template <class F> inline void parallel_for( char const * str , int n1 , F const &f , int vectorSize = 128 ) {
    parallel_for( n1 , f , vectorSize );
  }
