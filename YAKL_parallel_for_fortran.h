
#pragma once



  YAKL_INLINE void unpackIndicesFortran( int iGlob , Bounds bounds , int indices[8] ) {
    // First, compute the packed C-sytel indices
    int fac           ; indices[0] = (iGlob    ) % bounds(0,1);
    fac  = bounds(0,1); indices[1] = (iGlob/fac) % bounds(1,1);
    fac *= bounds(1,1); indices[2] = (iGlob/fac) % bounds(2,1);
    fac *= bounds(2,1); indices[3] = (iGlob/fac) % bounds(3,1);
    fac *= bounds(3,1); indices[4] = (iGlob/fac) % bounds(4,1);
    fac *= bounds(4,1); indices[5] = (iGlob/fac) % bounds(5,1);
    fac *= bounds(5,1); indices[6] = (iGlob/fac) % bounds(6,1);
    fac *= bounds(6,1); indices[7] = (iGlob/fac)              ;

    // Next, multiply by the stride and add the lower bound
    indices[0] = indices[0]*bounds(0,2) + bounds(0,0);
    indices[1] = indices[1]*bounds(1,2) + bounds(1,0);
    indices[2] = indices[2]*bounds(2,2) + bounds(2,0);
    indices[3] = indices[3]*bounds(3,2) + bounds(3,0);
    indices[4] = indices[4]*bounds(4,2) + bounds(4,0);
    indices[5] = indices[5]*bounds(5,2) + bounds(5,0);
    indices[6] = indices[6]*bounds(6,2) + bounds(6,0);
    indices[7] = indices[7]*bounds(7,2) + bounds(7,0);
  }



  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 ) {
    i0 = ind[0];
  }
  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1) {
    i0 = ind[0]; i1 = ind[1];
  }
  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1, int &i2) {
    i0 = ind[0]; i1 = ind[1]; i2 = ind[2];
  }
  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1, int &i2, int &i3) {
    i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3];
  }
  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1, int &i2, int &i3, int &i4) {
    i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4];
  }
  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1, int &i2, int &i3, int &i4, int &i5) {
    i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4]; i5 = ind[5];
  }
  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1, int &i2, int &i3, int &i4, int &i5, int &i6) {
    i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4]; i5 = ind[5]; i6 = ind[6];
  }
  YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7) {
    i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4]; i5 = ind[5]; i6 = ind[6]; i7 = ind[7];
  }



  #ifdef __USE_CUDA__
    template <class F> __global__ void cudaKernelVal( Bounds bounds , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < bounds.nIter()) {
        int indices[8];
        unpackIndicesFortran( i , bounds , indices );
        f( indices );
      }
    }

    template <class F> __global__ void cudaKernelRef( Bounds bounds , F const &f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < bounds.nIter()) {
        int indices[8];
        unpackIndicesFortran( i , bounds , indices );
        f( indices );
      }
    }

    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> void parallel_for_cuda( Bounds const &bounds , F const &f , int vectorSize = 128 ) {
      cudaKernelVal <<< (unsigned int) (bounds.nIter()-1)/vectorSize+1 , vectorSize >>> ( bounds , f );
    }

    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> void parallel_for_cuda( Bounds const &bounds , F const &f , int vectorSize = 128 ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      cudaKernelRef <<< (unsigned int) (bounds.nIter()-1)/vectorSize+1 , vectorSize >>> ( bounds , *fp );
    }
  #endif



  #ifdef __USE_HIP__
    template <class F> __global__ void hipKernel( Bounds bounds , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < bounds.nIter()) {
        int indices[8];
        unpackIndicesFortran( i , bounds , indices );
        f( indices );
      }
    }

    template<class F> void parallel_for_hip( Bounds const &bounds , F const &f , int vectorSize = 128 ) {
      hipLaunchKernelGGL( hipKernel , dim3((bounds.nIter()-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , bounds , f );
    }
  #endif



  template <class F> inline void parallel_for_cpu_serial( Bounds const &bounds , F const &f ) {
    for (int i=0; i<bounds.nIter(); i++) {
      int indices[8];
      unpackIndicesFortran( i , bounds , indices );
      f( indices );
    }
  }



  template <class F> inline void parallel_for( Bounds const &bounds , F const &f , int vectorSize = 128 ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( bounds , f , vectorSize );
    #elif defined(__USE_HIP__)
      parallel_for_hip ( bounds , f , vectorSize );
    #else
      parallel_for_cpu_serial( bounds , f );
    #endif

    #if defined(__AUTO_FENCE__)
      fence();
    #endif
  }



  template <class F> inline void parallel_for( char const * str , Bounds const &bounds , F const &f, int vectorSize = 128 ) {
    parallel_for( bounds , f , vectorSize );
  }


