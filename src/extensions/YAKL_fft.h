
#pragma once

#include "YAKL.h"

namespace yakl {

  template <class T> class RealFFT1D {
  public:
    int batch_size;
    int transform_size;
    int trdim;
    #ifdef YAKL_ARCH_CUDA
      cufftHandle plan_forward;
      cufftHandle plan_inverse;
    #endif

    RealFFT1D() { batch_size = -1;  transform_size = -1;  trdim = -1; }
    ~RealFFT1D() {
      #ifdef YAKL_ARCH_CUDA
        cufftDestroy( plan_forward );
        cufftDestroy( plan_inverse );
      #endif
    }

    template <int N> void init( Array<T,N,memDevice,styleC> &arr , int trdim , int tr_size ) {
      int rank    = 1;
      int n       = tr_size;
      int istride = 1;
      int ostride = 1;
      int idist   = arr.extent(trdim);
      int odist   = idist / 2;
      int batch   = arr.totElems() / arr.extent(trdim);
      int inembed = 0;
      int onembed = 0;
      #ifdef YAKL_ARCH_CUDA
        if        constexpr (std::is_same<T,float >::value) {
          cufftPlanMany(&plan_forward, rank, &n, &inembed, istride, idist, &onembed, ostride, odist, CUFFT_R2C, batch);
          cufftPlanMany(&plan_inverse, rank, &n, &onembed, ostride, odist, &inembed, istride, idist, CUFFT_C2R, batch);
        } else if constexpr (std::is_same<T,double>::value) {
          cufftPlanMany(&plan_forward, rank, &n, &inembed, istride, idist, &onembed, ostride, odist, CUFFT_D2Z, batch);
          cufftPlanMany(&plan_inverse, rank, &n, &onembed, ostride, odist, &inembed, istride, idist, CUFFT_Z2D, batch);
        }
      #endif

      this->batch_size     = batch  ;
      this->transform_size = tr_size;
      this->trdim          = trdim  ;
    }


    template <int N> void forward_real( Array<T,N,memDevice,styleC> &arr ) {
      if (trdim == arr.get_rank()-1) {
        #ifdef YAKL_ARCH_CUDA
          if        constexpr (std::is_same<T,float >::value) {
            cufftExecR2C(plan_forward, (cufftReal       *) arr.data(), (cufftComplex       *) arr.data());
          } else if constexpr (std::is_same<T,double>::value) {
            cufftExecD2Z(plan_forward, (cufftDoubleReal *) arr.data(), (cufftDoubleComplex *) arr.data());
          }
        #endif
      } else {
        auto dims = arr.get_dimensions();
        // Coallesce sizes of fastest varying dimensions of input array inside trdim dimension
        int d2 = 1;
        for (int i=N-1; i > trdim; i--) { d2 *= dims(i); }
        // Get the size of trdim of input array
        int d1 = dims(trdim);
        // Coallesce the sizes of the slowest varying dimensions of input array outside trdim dimension
        int d0 = arr.totElems() / d2 / d1;
        // The array we'll transpose into and out of will have trdim as the fastest varying
        auto in   = arr                   .reshape(d0,d1,d2);  // Cheap reshape, no copy created
        auto copy = arr.createDeviceCopy().reshape(d0,d2,d1);
        c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) {
          copy(i0,i2,i1) = in(i0,i1,i2);
        });
        // Perform the FFT
        #ifdef YAKL_ARCH_CUDA
          if        constexpr (std::is_same<T,float >::value) {
            cufftExecR2C(plan_forward, (cufftReal       *) copy.data(), (cufftComplex       *) copy.data());
          } else if constexpr (std::is_same<T,double>::value) {
            cufftExecD2Z(plan_forward, (cufftDoubleReal *) copy.data(), (cufftDoubleComplex *) copy.data());
          }
        #endif
        // Transpose the data back, overwriting "arr"
        c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) {
          in(i0,i1,i2) = copy(i0,i2,i1);
        });
      }
    }


    template <int N> void inverse_real( Array<T,N,memDevice,styleC> &arr ) {
      if (trdim == arr.get_rank()-1) {
        #ifdef YAKL_ARCH_CUDA
          if        constexpr (std::is_same<T,float >::value) {
            cufftExecC2R(plan_inverse, (cufftComplex       *) arr.data(), (cufftReal       *) arr.data());
          } else if constexpr (std::is_same<T,double>::value) {
            cufftExecZ2D(plan_inverse, (cufftDoubleComplex *) arr.data(), (cufftDoubleReal *) arr.data());
          }
          using yakl::componentwise::operator/;
          arr = arr / transform_size;
        #endif
      } else {
        auto dims = arr.get_dimensions();
        // Coallesce sizes of fastest varying dimensions of input array inside trdim dimension
        int d2 = 1;
        for (int i=N-1; i > trdim; i--) { d2 *= dims(i); }
        // Get the size of trdim of input array
        int d1 = dims(trdim);
        // Coallesce the sizes of the slowest varying dimensions of input array outside trdim dimension
        int d0 = arr.totElems() / d2 / d1;
        // The array we'll transpose into and out of will have trdim as the fastest varying
        auto in   = arr                   .reshape(d0,d1,d2);  // Cheap reshape, no copy created
        auto copy = arr.createDeviceCopy().reshape(d0,d2,d1);
        c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) {
          copy(i0,i2,i1) = in(i0,i1,i2);
        });
        // Perform the FFT
        #ifdef YAKL_ARCH_CUDA
          if        constexpr (std::is_same<T,float >::value) {
            cufftExecC2R(plan_inverse, (cufftComplex       *) copy.data(), (cufftReal       *) copy.data());
          } else if constexpr (std::is_same<T,double>::value) {
            cufftExecZ2D(plan_inverse, (cufftDoubleComplex *) copy.data(), (cufftDoubleReal *) copy.data());
          }
        #endif
        // Transpose the data back, overwriting "arr"
        c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) {
          in(i0,i1,i2) = copy(i0,i2,i1) / transform_size;
        });
      }
    }

  };

}

