
#pragma once


  template <int N> class Bounds;

  template<> class Bounds<8> {
  public:
    int nIter;
    int lbounds[8];
    int dims[8];
    int strides[8];
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ,
           std::vector<int> b2 = {0,0} ,
           std::vector<int> b3 = {0,0} ,
           std::vector<int> b4 = {0,0} ,
           std::vector<int> b5 = {0,0} ,
           std::vector<int> b6 = {0,0} ,
           std::vector<int> b7 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];
      lbounds[1] = b1[0];   strides[1] = b1.size() >= 3 ? b0[2] : 1;   dims[1] = ( b1[1] - lbounds[1] + 1 ) / strides[1];
      lbounds[2] = b2[0];   strides[2] = b2.size() >= 3 ? b0[2] : 1;   dims[2] = ( b2[1] - lbounds[2] + 1 ) / strides[2];
      lbounds[3] = b3[0];   strides[3] = b3.size() >= 3 ? b0[2] : 1;   dims[3] = ( b3[1] - lbounds[3] + 1 ) / strides[3];
      lbounds[4] = b4[0];   strides[4] = b4.size() >= 3 ? b0[2] : 1;   dims[4] = ( b4[1] - lbounds[4] + 1 ) / strides[4];
      lbounds[5] = b5[0];   strides[5] = b5.size() >= 3 ? b0[2] : 1;   dims[5] = ( b5[1] - lbounds[5] + 1 ) / strides[5];
      lbounds[6] = b6[0];   strides[6] = b6.size() >= 3 ? b0[2] : 1;   dims[6] = ( b6[1] - lbounds[6] + 1 ) / strides[6];
      lbounds[7] = b7[0];   strides[7] = b7.size() >= 3 ? b0[2] : 1;   dims[7] = ( b7[1] - lbounds[7] + 1 ) / strides[7];

      // Process bounds
      nIter = 1;
      for (int i=0; i<8; i++) {
        nIter *= dims[i];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[8] ) const {
      // First, compute the packed C-sytel indices
      int fac       ; indices[0] = (iGlob    ) % dims[0];
      fac  = dims[0]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[4] = (iGlob/fac) % dims[4];
      fac *= dims[4]; indices[5] = (iGlob/fac) % dims[5];
      fac *= dims[5]; indices[6] = (iGlob/fac) % dims[6];
      fac *= dims[6]; indices[7] = (iGlob/fac)          ;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
      indices[1] = indices[1]*strides[1] + lbounds[1];
      indices[2] = indices[2]*strides[2] + lbounds[2];
      indices[3] = indices[3]*strides[3] + lbounds[3];
      indices[4] = indices[4]*strides[4] + lbounds[4];
      indices[5] = indices[5]*strides[5] + lbounds[5];
      indices[6] = indices[6]*strides[6] + lbounds[6];
      indices[7] = indices[7]*strides[7] + lbounds[7];
    }
  };

  template<> class Bounds<7> {
  public:
    int nIter;
    int lbounds[7];
    int dims[7];
    int strides[7];
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ,
           std::vector<int> b2 = {0,0} ,
           std::vector<int> b3 = {0,0} ,
           std::vector<int> b4 = {0,0} ,
           std::vector<int> b5 = {0,0} ,
           std::vector<int> b6 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];
      lbounds[1] = b1[0];   strides[1] = b1.size() >= 3 ? b0[2] : 1;   dims[1] = ( b1[1] - lbounds[1] + 1 ) / strides[1];
      lbounds[2] = b2[0];   strides[2] = b2.size() >= 3 ? b0[2] : 1;   dims[2] = ( b2[1] - lbounds[2] + 1 ) / strides[2];
      lbounds[3] = b3[0];   strides[3] = b3.size() >= 3 ? b0[2] : 1;   dims[3] = ( b3[1] - lbounds[3] + 1 ) / strides[3];
      lbounds[4] = b4[0];   strides[4] = b4.size() >= 3 ? b0[2] : 1;   dims[4] = ( b4[1] - lbounds[4] + 1 ) / strides[4];
      lbounds[5] = b5[0];   strides[5] = b5.size() >= 3 ? b0[2] : 1;   dims[5] = ( b5[1] - lbounds[5] + 1 ) / strides[5];
      lbounds[6] = b6[0];   strides[6] = b6.size() >= 3 ? b0[2] : 1;   dims[6] = ( b6[1] - lbounds[6] + 1 ) / strides[6];

      // Process bounds
      nIter = 1;
      for (int i=0; i<7; i++) {
        nIter *= dims[i];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[7] ) const {
      // First, compute the packed C-sytel indices
      int fac       ; indices[0] = (iGlob    ) % dims[0];
      fac  = dims[0]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[4] = (iGlob/fac) % dims[4];
      fac *= dims[4]; indices[5] = (iGlob/fac) % dims[5];
      fac *= dims[5]; indices[6] = (iGlob/fac)          ;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
      indices[1] = indices[1]*strides[1] + lbounds[1];
      indices[2] = indices[2]*strides[2] + lbounds[2];
      indices[3] = indices[3]*strides[3] + lbounds[3];
      indices[4] = indices[4]*strides[4] + lbounds[4];
      indices[5] = indices[5]*strides[5] + lbounds[5];
      indices[6] = indices[6]*strides[6] + lbounds[6];
    }
  };

  template<> class Bounds<6> {
  public:
    int nIter;
    int lbounds[6];
    int dims[6];
    int strides[6];
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ,
           std::vector<int> b2 = {0,0} ,
           std::vector<int> b3 = {0,0} ,
           std::vector<int> b4 = {0,0} ,
           std::vector<int> b5 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];
      lbounds[1] = b1[0];   strides[1] = b1.size() >= 3 ? b0[2] : 1;   dims[1] = ( b1[1] - lbounds[1] + 1 ) / strides[1];
      lbounds[2] = b2[0];   strides[2] = b2.size() >= 3 ? b0[2] : 1;   dims[2] = ( b2[1] - lbounds[2] + 1 ) / strides[2];
      lbounds[3] = b3[0];   strides[3] = b3.size() >= 3 ? b0[2] : 1;   dims[3] = ( b3[1] - lbounds[3] + 1 ) / strides[3];
      lbounds[4] = b4[0];   strides[4] = b4.size() >= 3 ? b0[2] : 1;   dims[4] = ( b4[1] - lbounds[4] + 1 ) / strides[4];
      lbounds[5] = b5[0];   strides[5] = b5.size() >= 3 ? b0[2] : 1;   dims[5] = ( b5[1] - lbounds[5] + 1 ) / strides[5];

      // Process bounds
      nIter = 1;
      for (int i=0; i<6; i++) {
        nIter *= dims[i];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE void unpackIndices( int iGlob  , int indices[6]) const {
      // First, compute the packed C-sytel indices
      int fac       ; indices[0] = (iGlob    ) % dims[0];
      fac  = dims[0]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[4] = (iGlob/fac) % dims[4];
      fac *= dims[4]; indices[5] = (iGlob/fac)          ;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
      indices[1] = indices[1]*strides[1] + lbounds[1];
      indices[2] = indices[2]*strides[2] + lbounds[2];
      indices[3] = indices[3]*strides[3] + lbounds[3];
      indices[4] = indices[4]*strides[4] + lbounds[4];
      indices[5] = indices[5]*strides[5] + lbounds[5];
    }
  };

  template<> class Bounds<5> {
  public:
    int nIter;
    int lbounds[5];
    int dims[5];
    int strides[5];
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ,
           std::vector<int> b2 = {0,0} ,
           std::vector<int> b3 = {0,0} ,
           std::vector<int> b4 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];
      lbounds[1] = b1[0];   strides[1] = b1.size() >= 3 ? b0[2] : 1;   dims[1] = ( b1[1] - lbounds[1] + 1 ) / strides[1];
      lbounds[2] = b2[0];   strides[2] = b2.size() >= 3 ? b0[2] : 1;   dims[2] = ( b2[1] - lbounds[2] + 1 ) / strides[2];
      lbounds[3] = b3[0];   strides[3] = b3.size() >= 3 ? b0[2] : 1;   dims[3] = ( b3[1] - lbounds[3] + 1 ) / strides[3];
      lbounds[4] = b4[0];   strides[4] = b4.size() >= 3 ? b0[2] : 1;   dims[4] = ( b4[1] - lbounds[4] + 1 ) / strides[4];

      // Process bounds
      nIter = 1;
      for (int i=0; i<5; i++) {
        nIter *= dims[i];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE void unpackIndices( int iGlob  , int indices[5]) const {
      // First, compute the packed C-sytel indices
      int fac       ; indices[0] = (iGlob    ) % dims[0];
      fac  = dims[0]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[4] = (iGlob/fac)          ;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
      indices[1] = indices[1]*strides[1] + lbounds[1];
      indices[2] = indices[2]*strides[2] + lbounds[2];
      indices[3] = indices[3]*strides[3] + lbounds[3];
      indices[4] = indices[4]*strides[4] + lbounds[4];
    }
  };

  template<> class Bounds<4> {
  public:
    int nIter;
    int lbounds[4];
    int dims[4];
    int strides[4];
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ,
           std::vector<int> b2 = {0,0} ,
           std::vector<int> b3 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];
      lbounds[1] = b1[0];   strides[1] = b1.size() >= 3 ? b0[2] : 1;   dims[1] = ( b1[1] - lbounds[1] + 1 ) / strides[1];
      lbounds[2] = b2[0];   strides[2] = b2.size() >= 3 ? b0[2] : 1;   dims[2] = ( b2[1] - lbounds[2] + 1 ) / strides[2];
      lbounds[3] = b3[0];   strides[3] = b3.size() >= 3 ? b0[2] : 1;   dims[3] = ( b3[1] - lbounds[3] + 1 ) / strides[3];

      // Process bounds
      nIter = 1;
      for (int i=0; i<4; i++) {
        nIter *= dims[i];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE void unpackIndices( int iGlob  , int indices[4]) const {
      // First, compute the packed C-sytel indices
      int fac       ; indices[0] = (iGlob    ) % dims[0];
      fac  = dims[0]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[3] = (iGlob/fac)          ;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
      indices[1] = indices[1]*strides[1] + lbounds[1];
      indices[2] = indices[2]*strides[2] + lbounds[2];
      indices[3] = indices[3]*strides[3] + lbounds[3];
    }
  };

  template<> class Bounds<3> {
  public:
    int nIter;
    int lbounds[3];
    int dims[3];
    int strides[3];
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ,
           std::vector<int> b2 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];
      lbounds[1] = b1[0];   strides[1] = b1.size() >= 3 ? b0[2] : 1;   dims[1] = ( b1[1] - lbounds[1] + 1 ) / strides[1];
      lbounds[2] = b2[0];   strides[2] = b2.size() >= 3 ? b0[2] : 1;   dims[2] = ( b2[1] - lbounds[2] + 1 ) / strides[2];

      // Process bounds
      nIter = 1;
      for (int i=0; i<3; i++) {
        nIter *= dims[i];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[3] ) const {
      // First, compute the packed C-sytel indices
      int fac       ; indices[0] = (iGlob    ) % dims[0];
      fac  = dims[0]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[2] = (iGlob/fac)          ;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
      indices[1] = indices[1]*strides[1] + lbounds[1];
      indices[2] = indices[2]*strides[2] + lbounds[2];
    }
  };

  template<> class Bounds<2> {
  public:
    int nIter;
    int lbounds[2];
    int dims[2];
    int strides[2];
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];
      lbounds[1] = b1[0];   strides[1] = b1.size() >= 3 ? b0[2] : 1;   dims[1] = ( b1[1] - lbounds[1] + 1 ) / strides[1];

      // Process bounds
      nIter = 1;
      for (int i=0; i<2; i++) {
        nIter *= dims[i];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[2] ) const {
      // First, compute the packed C-sytel indices
      int fac       ; indices[0] = (iGlob    ) % dims[0];
      fac  = dims[0]; indices[1] = (iGlob/fac)          ;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
      indices[1] = indices[1]*strides[1] + lbounds[1];
    }
  };

  template<> class Bounds<1> {
  public:
    int nIter;
    int lbounds[1];
    int dims[1];
    int strides[1];
    Bounds(std::vector<int> b0) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      lbounds[0] = b0[0];   strides[0] = b0.size() >= 3 ? b0[2] : 1;   dims[0] = ( b0[1] - lbounds[0] + 1 ) / strides[0];

      // Process bounds
      nIter = dims[0];
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[1] ) const {
      // First, compute the packed C-sytel indices
      indices[0] = iGlob;

      // Next, multiply by the stride and add the lower bound
      indices[0] = indices[0]*strides[0] + lbounds[0];
    }
  };






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
    template <class F, int N> __global__ void cudaKernelVal( Bounds<N> bounds , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < bounds.nIter) {
        int indices[N];
        bounds.unpackIndices( i , indices );
        f( indices );
      }
    }

    template <class F, int N> __global__ void cudaKernelRef( Bounds<N> bounds , F const &f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < bounds.nIter) {
        int indices[N];
        bounds.unpackIndices( i , indices );
        f( indices );
      }
    }

    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> void parallel_for_cuda( Bounds const &bounds , F const &f , int vectorSize = 128 ) {
      cudaKernelVal <<< (unsigned int) (bounds.nIter-1)/vectorSize+1 , vectorSize >>> ( bounds , f );
    }

    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> void parallel_for_cuda( Bounds const &bounds , F const &f , int vectorSize = 128 ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      cudaKernelRef <<< (unsigned int) (bounds.nIter-1)/vectorSize+1 , vectorSize >>> ( bounds , *fp );
    }
  #endif



  #ifdef __USE_HIP__
    template <class F, int N> __global__ void hipKernel( Bounds<N> bounds , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < bounds.nIter) {
        int indices[N];
        bounds.unpackIndices( i , indices );
        f( indices );
      }
    }

    template<class F, int N> void parallel_for_hip( Bounds<N> const &bounds , F const &f , int vectorSize = 128 ) {
      hipLaunchKernelGGL( hipKernel , dim3((bounds.nIter-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , bounds , f );
    }
  #endif



  template <class F, int N> inline void parallel_for_cpu_serial( Bounds<N> const &bounds , F const &f ) {
    for (int i=0; i<bounds.nIter; i++) {
      int indices[N];
      bounds.unpackIndices( i , indices );
      f( indices );
    }
  }



  template <class F, int N> inline void parallel_for( Bounds<N> const &bounds , F const &f , int vectorSize = 128 ) {
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



  template <class F, int N> inline void parallel_for( char const * str , Bounds<N> const &bounds , F const &f, int vectorSize = 128 ) {
    parallel_for( bounds , f , vectorSize );
  }


