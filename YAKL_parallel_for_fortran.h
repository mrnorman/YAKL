
#pragma once



namespace fortran {



  class LBnd {
  public:
    int l, u, s;
    LBnd(int u) {
      this->l = 1;
      this->u = u;
      this->s = 1;
    }
    LBnd(int l, int u) {
      this->l = l;
      this->u = u;
      this->s = 1;
    }
    LBnd(int l, int u, int s) {
      this->l = l;
      this->u = u;
      this->s = s;
    }
  };



  template <int N> class Bounds;

  template<> class Bounds<8> {
  public:
    int nIter;
    int lbounds[8];
    int dims[8];
    int strides[8];
    Bounds( LBnd const &b0 , LBnd const &b1 , LBnd const &b2 , LBnd const &b3 , LBnd const &b4 , LBnd const &b5 , LBnd const &b6 , LBnd const &b7 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s;
      lbounds[2] = b2.l;   strides[2] =  b2.s;   dims[2] = ( b2.u - b2.l + 1 ) / b2.s;
      lbounds[3] = b3.l;   strides[3] =  b3.s;   dims[3] = ( b3.u - b3.l + 1 ) / b3.s;
      lbounds[4] = b4.l;   strides[4] =  b4.s;   dims[4] = ( b4.u - b4.l + 1 ) / b4.s;
      lbounds[5] = b5.l;   strides[5] =  b5.s;   dims[5] = ( b5.u - b5.l + 1 ) / b5.s;
      lbounds[6] = b6.l;   strides[6] =  b6.s;   dims[6] = ( b6.u - b6.l + 1 ) / b6.s;
      lbounds[7] = b7.l;   strides[7] =  b7.s;   dims[7] = ( b7.u - b7.l + 1 ) / b7.s;
      nIter = 1;
      for (int i=0; i<8; i++) { nIter *= dims[i]; }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[8] ) const {
      // Compute base indices
      int fac       ; indices[7] = (iGlob    ) % dims[7];
      fac  = dims[7]; indices[6] = (iGlob/fac) % dims[6];
      fac *= dims[6]; indices[5] = (iGlob/fac) % dims[5];
      fac *= dims[5]; indices[4] = (iGlob/fac) % dims[4];
      fac *= dims[4]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[0] = (iGlob/fac)          ;
      // Apply strides and lower bounds
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
    Bounds( LBnd const &b0 , LBnd const &b1 , LBnd const &b2 , LBnd const &b3 , LBnd const &b4 , LBnd const &b5 , LBnd const &b6 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s;
      lbounds[2] = b2.l;   strides[2] =  b2.s;   dims[2] = ( b2.u - b2.l + 1 ) / b2.s;
      lbounds[3] = b3.l;   strides[3] =  b3.s;   dims[3] = ( b3.u - b3.l + 1 ) / b3.s;
      lbounds[4] = b4.l;   strides[4] =  b4.s;   dims[4] = ( b4.u - b4.l + 1 ) / b4.s;
      lbounds[5] = b5.l;   strides[5] =  b5.s;   dims[5] = ( b5.u - b5.l + 1 ) / b5.s;
      lbounds[6] = b6.l;   strides[6] =  b6.s;   dims[6] = ( b6.u - b6.l + 1 ) / b6.s;
      nIter = 1;
      for (int i=0; i<7; i++) { nIter *= dims[i]; }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[7] ) const {
      // Compute base indices
      int fac       ; indices[6] = (iGlob    ) % dims[6];
      fac  = dims[6]; indices[5] = (iGlob/fac) % dims[5];
      fac *= dims[5]; indices[4] = (iGlob/fac) % dims[4];
      fac *= dims[4]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[0] = (iGlob/fac)          ;
      // Apply strides and lower bounds
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
    Bounds( LBnd const &b0 , LBnd const &b1 , LBnd const &b2 , LBnd const &b3 , LBnd const &b4 , LBnd const &b5 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s;
      lbounds[2] = b2.l;   strides[2] =  b2.s;   dims[2] = ( b2.u - b2.l + 1 ) / b2.s;
      lbounds[3] = b3.l;   strides[3] =  b3.s;   dims[3] = ( b3.u - b3.l + 1 ) / b3.s;
      lbounds[4] = b4.l;   strides[4] =  b4.s;   dims[4] = ( b4.u - b4.l + 1 ) / b4.s;
      lbounds[5] = b5.l;   strides[5] =  b5.s;   dims[5] = ( b5.u - b5.l + 1 ) / b5.s;
      nIter = 1;
      for (int i=0; i<6; i++) { nIter *= dims[i]; }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[6] ) const {
      // Compute base indices
      int fac       ; indices[5] = (iGlob    ) % dims[5];
      fac  = dims[5]; indices[4] = (iGlob/fac) % dims[4];
      fac *= dims[4]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[0] = (iGlob/fac)          ;
      // Apply strides and lower bounds
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
    Bounds( LBnd const &b0 , LBnd const &b1 , LBnd const &b2 , LBnd const &b3 , LBnd const &b4 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s;
      lbounds[2] = b2.l;   strides[2] =  b2.s;   dims[2] = ( b2.u - b2.l + 1 ) / b2.s;
      lbounds[3] = b3.l;   strides[3] =  b3.s;   dims[3] = ( b3.u - b3.l + 1 ) / b3.s;
      lbounds[4] = b4.l;   strides[4] =  b4.s;   dims[4] = ( b4.u - b4.l + 1 ) / b4.s;
      nIter = 1;
      for (int i=0; i<5; i++) { nIter *= dims[i]; }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[5] ) const {
      // Compute base indices
      int fac       ; indices[4] = (iGlob    ) % dims[4];
      fac  = dims[4]; indices[3] = (iGlob/fac) % dims[3];
      fac *= dims[3]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[0] = (iGlob/fac)          ;
      // Apply strides and lower bounds
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
    Bounds( LBnd const &b0 , LBnd const &b1 , LBnd const &b2 , LBnd const &b3 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s;
      lbounds[2] = b2.l;   strides[2] =  b2.s;   dims[2] = ( b2.u - b2.l + 1 ) / b2.s;
      lbounds[3] = b3.l;   strides[3] =  b3.s;   dims[3] = ( b3.u - b3.l + 1 ) / b3.s;
      nIter = 1;
      for (int i=0; i<4; i++) { nIter *= dims[i]; }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[4] ) const {
      // Compute base indices
      int fac       ; indices[3] = (iGlob    ) % dims[3];
      fac  = dims[3]; indices[2] = (iGlob/fac) % dims[2];
      fac *= dims[2]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[0] = (iGlob/fac)          ;
      // Apply strides and lower bounds
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
    Bounds( LBnd const &b0 , LBnd const &b1 , LBnd const &b2 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s;
      lbounds[2] = b2.l;   strides[2] =  b2.s;   dims[2] = ( b2.u - b2.l + 1 ) / b2.s;
      nIter = 1;
      for (int i=0; i<3; i++) { nIter *= dims[i]; }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[3] ) const {
      // Compute base indices
      int fac       ; indices[2] = (iGlob    ) % dims[2];
      fac  = dims[2]; indices[1] = (iGlob/fac) % dims[1];
      fac *= dims[1]; indices[0] = (iGlob/fac)          ;
      // Apply strides and lower bounds
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
    Bounds( LBnd const &b0 , LBnd const &b1 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s;
      nIter = 1;
      for (int i=0; i<2; i++) { nIter *= dims[i]; }
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[2] ) const {
      // Compute base indices
      indices[1] = (iGlob        ) % dims[1];
      indices[0] = (iGlob/dims[1])          ;
      // Apply strides and lower bounds
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
    Bounds( LBnd const &b0 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      nIter = dims[0];
    }
    YAKL_INLINE void unpackIndices( int iGlob , int indices[1] ) const {
      // Compute base indices
      indices[0] = iGlob;
      // Apply strides and lower bounds
      indices[0] = indices[0]*strides[0] + lbounds[0];
    }
  };



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

    template<class F , int N , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> void parallel_for_cuda( Bounds<N> const &bounds , F const &f , int vectorSize = 128 ) {
      cudaKernelVal <<< (unsigned int) (bounds.nIter-1)/vectorSize+1 , vectorSize >>> ( bounds , f );
    }

    template<class F , int N , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> void parallel_for_cuda( Bounds<N> const &bounds , F const &f , int vectorSize = 128 ) {
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



  template <class F> inline void parallel_for_cpu_serial( Bounds<1> const &bounds , F const &f ) {
    int indices[1];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
      indices[0] = i0;
      f( indices );
    }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<2> const &bounds , F const &f ) {
    int indices[2];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
    for (int i1 = bounds.lbounds[1]; i1 < bounds.lbounds[1]+bounds.dims[1]; i1++) {
      indices[0] = i0;
      indices[1] = i1;
      f( indices );
    } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<3> const &bounds , F const &f ) {
    int indices[3];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
    for (int i1 = bounds.lbounds[1]; i1 < bounds.lbounds[1]+bounds.dims[1]; i1++) {
    for (int i2 = bounds.lbounds[2]; i2 < bounds.lbounds[2]+bounds.dims[2]; i2++) {
      indices[0] = i0;
      indices[1] = i1;
      indices[2] = i2;
      f( indices );
    } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<4> const &bounds , F const &f ) {
    int indices[4];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
    for (int i1 = bounds.lbounds[1]; i1 < bounds.lbounds[1]+bounds.dims[1]; i1++) {
    for (int i2 = bounds.lbounds[2]; i2 < bounds.lbounds[2]+bounds.dims[2]; i2++) {
    for (int i3 = bounds.lbounds[3]; i3 < bounds.lbounds[3]+bounds.dims[3]; i3++) {
      indices[0] = i0;
      indices[1] = i1;
      indices[2] = i2;
      indices[3] = i3;
      f( indices );
    } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<5> const &bounds , F const &f ) {
    int indices[5];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
    for (int i1 = bounds.lbounds[1]; i1 < bounds.lbounds[1]+bounds.dims[1]; i1++) {
    for (int i2 = bounds.lbounds[2]; i2 < bounds.lbounds[2]+bounds.dims[2]; i2++) {
    for (int i3 = bounds.lbounds[3]; i3 < bounds.lbounds[3]+bounds.dims[3]; i3++) {
    for (int i4 = bounds.lbounds[4]; i4 < bounds.lbounds[4]+bounds.dims[4]; i4++) {
      indices[0] = i0;
      indices[1] = i1;
      indices[2] = i2;
      indices[3] = i3;
      indices[4] = i4;
      f( indices );
    } } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<6> const &bounds , F const &f ) {
    int indices[6];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
    for (int i1 = bounds.lbounds[1]; i1 < bounds.lbounds[1]+bounds.dims[1]; i1++) {
    for (int i2 = bounds.lbounds[2]; i2 < bounds.lbounds[2]+bounds.dims[2]; i2++) {
    for (int i3 = bounds.lbounds[3]; i3 < bounds.lbounds[3]+bounds.dims[3]; i3++) {
    for (int i4 = bounds.lbounds[4]; i4 < bounds.lbounds[4]+bounds.dims[4]; i4++) {
    for (int i5 = bounds.lbounds[5]; i5 < bounds.lbounds[5]+bounds.dims[5]; i5++) {
      indices[0] = i0;
      indices[1] = i1;
      indices[2] = i2;
      indices[3] = i3;
      indices[4] = i4;
      indices[5] = i5;
      f( indices );
    } } } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<7> const &bounds , F const &f ) {
    int indices[7];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
    for (int i1 = bounds.lbounds[1]; i1 < bounds.lbounds[1]+bounds.dims[1]; i1++) {
    for (int i2 = bounds.lbounds[2]; i2 < bounds.lbounds[2]+bounds.dims[2]; i2++) {
    for (int i3 = bounds.lbounds[3]; i3 < bounds.lbounds[3]+bounds.dims[3]; i3++) {
    for (int i4 = bounds.lbounds[4]; i4 < bounds.lbounds[4]+bounds.dims[4]; i4++) {
    for (int i5 = bounds.lbounds[5]; i5 < bounds.lbounds[5]+bounds.dims[5]; i5++) {
    for (int i6 = bounds.lbounds[6]; i6 < bounds.lbounds[6]+bounds.dims[6]; i6++) {
      indices[0] = i0;
      indices[1] = i1;
      indices[2] = i2;
      indices[3] = i3;
      indices[4] = i4;
      indices[5] = i5;
      indices[6] = i6;
      f( indices );
    } } } } } } }
  }
  template <class F> inline void parallel_for_cpu_serial( Bounds<8> const &bounds , F const &f ) {
    int indices[8];
    for (int i0 = bounds.lbounds[0]; i0 < bounds.lbounds[0]+bounds.dims[0]; i0++) {
    for (int i1 = bounds.lbounds[1]; i1 < bounds.lbounds[1]+bounds.dims[1]; i1++) {
    for (int i2 = bounds.lbounds[2]; i2 < bounds.lbounds[2]+bounds.dims[2]; i2++) {
    for (int i3 = bounds.lbounds[3]; i3 < bounds.lbounds[3]+bounds.dims[3]; i3++) {
    for (int i4 = bounds.lbounds[4]; i4 < bounds.lbounds[4]+bounds.dims[4]; i4++) {
    for (int i5 = bounds.lbounds[5]; i5 < bounds.lbounds[5]+bounds.dims[5]; i5++) {
    for (int i6 = bounds.lbounds[6]; i6 < bounds.lbounds[6]+bounds.dims[6]; i6++) {
    for (int i7 = bounds.lbounds[7]; i7 < bounds.lbounds[7]+bounds.dims[7]; i7++) {
      indices[0] = i0;
      indices[1] = i1;
      indices[2] = i2;
      indices[3] = i3;
      indices[4] = i4;
      indices[5] = i5;
      indices[6] = i6;
      indices[7] = i7;
      f( indices );
    } } } } } } } }
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



}
