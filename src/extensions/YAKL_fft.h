
#pragma once

#include "YAKL.h"
#define POCKETFFT_CACHE_SIZE 2
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace yakl {

  template <class T> class RealFFT1D {
  public:
    int batch_size;
    int transform_size;
    int trdim;
    #if   defined(YAKL_ARCH_CUDA)
      cufftHandle plan_forward;
      cufftHandle plan_inverse;
      #define CHECK(func) { int myloc = func; if (myloc != CUFFT_SUCCESS) { std::cerr << "ERROR: YAKL CUFFT: " << __FILE__ << ": " <<__LINE__ << std::endl; yakl_throw(""); } }
    #elif defined(YAKL_ARCH_HIP)
      rocfft_plan plan_forward;
      rocfft_plan plan_inverse;
      #define CHECK(func) { int myloc = func; if (myloc != rocfft_status_success) { std::cerr << "ERROR: YAKL ROCFFT: " << __FILE__ << ": " <<__LINE__ << std::endl; yakl_throw(""); } }
    #endif

    RealFFT1D() { batch_size = -1;  transform_size = -1;  trdim = -1; }
    ~RealFFT1D() { cleanup(); }

    void cleanup() {
      #if   defined(YAKL_ARCH_CUDA)
        CHECK( cufftDestroy( plan_forward ) );
        CHECK( cufftDestroy( plan_inverse ) );
      #elif defined(YAKL_ARCH_HIP)
        CHECK( rocfft_plan_destroy( plan_forward ) );
        CHECK( rocfft_plan_destroy( plan_inverse ) );
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
      #if   defined(YAKL_ARCH_CUDA)
        if        constexpr (std::is_same<T,float >::value) {
          CHECK( cufftPlanMany(&plan_forward, rank, &n, &inembed, istride, idist, &onembed, ostride, odist, CUFFT_R2C, batch) );
          CHECK( cufftPlanMany(&plan_inverse, rank, &n, &onembed, ostride, odist, &inembed, istride, idist, CUFFT_C2R, batch) );
        } else if constexpr (std::is_same<T,double>::value) {
          CHECK( cufftPlanMany(&plan_forward, rank, &n, &inembed, istride, idist, &onembed, ostride, odist, CUFFT_D2Z, batch) );
          CHECK( cufftPlanMany(&plan_inverse, rank, &n, &onembed, ostride, odist, &inembed, istride, idist, CUFFT_Z2D, batch) );
        }
      #elif defined(YAKL_ARCH_HIP)
        size_t len = tr_size;
        size_t const roc_istride = istride;
        size_t const roc_ostride = ostride;
        size_t const roc_idist = idist;
        size_t const roc_odist = odist;
        size_t const roc_off = 0;
        rocfft_plan_description desc;
        if        constexpr (std::is_same<T,float >::value) {
          CHECK( rocfft_plan_description_create( &desc ) );
          CHECK( rocfft_plan_description_set_data_layout( desc, rocfft_array_type_real, rocfft_array_type_hermitian_interleaved, &roc_off, &roc_off, (size_t) 1, &roc_istride, roc_idist, (size_t) 1, &roc_ostride, roc_odist ) );
          CHECK( rocfft_plan_create(&plan_forward, rocfft_placement_inplace, rocfft_transform_type_real_forward, rocfft_precision_single, (size_t) 1, &len, (size_t) batch, desc) );
          CHECK( rocfft_plan_description_destroy( desc ) );
          CHECK( rocfft_plan_description_create( &desc ) );
          CHECK( rocfft_plan_description_set_data_layout( desc, rocfft_array_type_hermitian_interleaved, rocfft_array_type_real, &roc_off, &roc_off, (size_t) 1, &roc_ostride, roc_odist, (size_t) 1, &roc_istride, roc_idist ) );
          CHECK( rocfft_plan_create(&plan_inverse, rocfft_placement_inplace, rocfft_transform_type_real_inverse, rocfft_precision_single, (size_t) 1, &len, (size_t) batch, desc) );
          CHECK( rocfft_plan_description_destroy( desc ) );
        } else if constexpr (std::is_same<T,double>::value) {
          CHECK( rocfft_plan_description_create( &desc ) );
          CHECK( rocfft_plan_description_set_data_layout( desc, rocfft_array_type_real, rocfft_array_type_hermitian_interleaved, &roc_off, &roc_off, (size_t) 1, &roc_istride, roc_idist, (size_t) 1, &roc_ostride, roc_odist ) );
          CHECK( rocfft_plan_create(&plan_forward, rocfft_placement_inplace, rocfft_transform_type_real_forward, rocfft_precision_double, (size_t) 1, &len, (size_t) batch, desc) );
          CHECK( rocfft_plan_description_destroy( desc ) );
          CHECK( rocfft_plan_description_create( &desc ) );
          CHECK( rocfft_plan_description_set_data_layout( desc, rocfft_array_type_hermitian_interleaved, rocfft_array_type_real, &roc_off, &roc_off, (size_t) 1, &roc_ostride, roc_odist, (size_t) 1, &roc_istride, roc_idist ) );
          CHECK( rocfft_plan_create(&plan_inverse, rocfft_placement_inplace, rocfft_transform_type_real_inverse, rocfft_precision_double, (size_t) 1, &len, (size_t) batch, desc) );
          CHECK( rocfft_plan_description_destroy( desc ) );
        }
      #endif

      this->batch_size     = batch  ;
      this->transform_size = tr_size;
      this->trdim          = trdim  ;
    }


    template <int N> void forward_real( Array<T,N,memDevice,styleC> &arr , int trdim_in = -1 , int transform_size_in = -1 ) {
      // Test if it's been initialized at all
      if (trdim < 0 || transform_size < 0 || batch_size < 0) {
        if (trdim_in < 0 || transform_size_in < 0) yakl_throw("ERROR: Using forward_real before calling init without "
                                                              "specifying both trdim_in and transform_size_in");
        init( arr , trdim_in , transform_size_in );
      }
      // Test if the apparent size of the transform has changed or the batch size has changed
      if ( ( (transform_size%2==1 ? transform_size+1 : transform_size+2) != arr.extent(trdim) ) ||
           ( transform_size_in > 0 && transform_size_in != transform_size ) ||
           ( arr.totElems() / arr.extent(trdim) != batch_size ) ) {
        if (trdim_in < 0 || transform_size_in < 0) yakl_throw("ERROR: Changing transform size  or batch sizewithout "
                                                              "specifying both trdim_in and transform_size_in");
        cleanup();
        init( arr , trdim_in , transform_size_in );
      }
      auto dims = arr.get_dimensions();
      int d2 = 1;   for (int i=N-1; i > trdim; i--) { d2 *= dims(i); } // Fastest varying
      int d1 = dims(trdim);                                            // Transform dimension
      int d0 = arr.totElems() / d2 / d1;                               // Slowest varying
      Array<T,3,memDevice,styleC> copy;
      if (trdim == N-1) {
        copy = arr.reshape(d0,d2,d1);
      } else {
        auto in = arr.reshape(d0,d1,d2);
        copy = arr.createDeviceCopy().reshape(d0,d2,d1);
        c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) { copy(i0,i2,i1) = in(i0,i1,i2); });
      }
      // Perform the FFT
      #if   defined(YAKL_ARCH_CUDA)
        if        constexpr (std::is_same<T,float >::value) {
          CHECK( cufftExecR2C(plan_forward, (cufftReal       *) copy.data(), (cufftComplex       *) copy.data()) );
        } else if constexpr (std::is_same<T,double>::value) {
          CHECK( cufftExecD2Z(plan_forward, (cufftDoubleReal *) copy.data(), (cufftDoubleComplex *) copy.data()) );
        }
      #elif defined(YAKL_ARCH_HIP)
        std::array<double *,1> ibuf( {copy.data()} );
        std::array<double *,1> obuf( {copy.data()} );
        rocfft_execution_info info;
        CHECK( rocfft_execution_info_create( &info ) );
        CHECK( rocfft_execute(plan_forward, (void **) ibuf.data(), (void **) obuf.data(), info) );
        CHECK( rocfft_execution_info_destroy( info ) );
      #else
        Array<T              ,3,memHost,styleC> pfft_in ("pfft_in" ,d0,d2, transform_size     );
        Array<std::complex<T>,3,memHost,styleC> pfft_out("pfft_out",d0,d2,(transform_size+2)/2);
        auto copy_host = copy.createHostCopy();
        for (int i0 = 0; i0 < d0; i0++) {
          for (int i2 = 0; i2 < d2; i2++) {
            for (int i1 = 0; i1 < transform_size; i1++) {
              pfft_in(i0,i2,i1) = copy_host(i0,i2,i1);
            }
          }
        }
        using pocketfft::detail::shape_t;
        using pocketfft::detail::stride_t;
        shape_t  shape_in  (3);   for (int i=0; i < 3; i++) { shape_in[i] = pfft_in.extent(i); }
        stride_t stride_in (3);
        stride_t stride_out(3);
        stride_in [0] = d2*  transform_size      *sizeof(             T );
        stride_in [1] =      transform_size      *sizeof(             T );
        stride_in [2] =                           sizeof(             T );                 
        stride_out[0] = d2*((transform_size+2)/2)*sizeof(std::complex<T>);
        stride_out[1] =    ((transform_size+2)/2)*sizeof(std::complex<T>);
        stride_out[2] =                           sizeof(std::complex<T>);   
        pocketfft::r2c<T>(shape_in, stride_in, stride_out, (size_t) 2, true, pfft_in.data(), pfft_out.data(), (T) 1);
        for (int i0 = 0; i0 < d0; i0++) {
          for (int i2 = 0; i2 < d2; i2++) {
            for (int i1 = 0; i1 < (transform_size+2)/2; i1++) {
              copy_host(i0,i2,2*i1  ) = pfft_out(i0,i2,i1).real();
              copy_host(i0,i2,2*i1+1) = pfft_out(i0,i2,i1).imag();
            }
          }
        }
        copy_host.deep_copy_to( copy );
        fence();
      #endif
      if (trdim != N-1) {
        auto out = arr.reshape(d0,d1,d2);
        c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) { out(i0,i1,i2) = copy(i0,i2,i1); });
      }
    }


    template <int N> void inverse_real( Array<T,N,memDevice,styleC> &arr , int trdim_in = -1 , int transform_size_in = -1 ) {
      // Test if it's been initialized at all
      if (trdim < 0 || transform_size < 0 || batch_size < 0) {
        if (trdim_in < 0 || transform_size_in < 0) yakl_throw("ERROR: Using forward_real before calling init without "
                                                              "specifying both trdim_in and transform_size_in");
        init( arr , trdim_in , transform_size_in );
      }
      // Test if the apparent size of the transform has changed or the batch size has changed
      if ( ( (transform_size%2==1 ? transform_size+1 : transform_size+2) != arr.extent(trdim) ) ||
           ( transform_size_in > 0 && transform_size_in != transform_size ) ||
           ( arr.totElems() / arr.extent(trdim) != batch_size ) ) {
        if (trdim_in < 0 || transform_size_in < 0) yakl_throw("ERROR: Changing transform size  or batch sizewithout "
                                                              "specifying both trdim_in and transform_size_in");
        cleanup();
        init( arr , trdim_in , transform_size_in );
      }
      auto dims = arr.get_dimensions();
      int d2 = 1;   for (int i=N-1; i > trdim; i--) { d2 *= dims(i); } // Fastest varying
      int d1 = dims(trdim);                                            // Transform dimension
      int d0 = arr.totElems() / d2 / d1;                               // Slowest varying
      Array<T,3,memDevice,styleC> copy;
      if (trdim == N-1) {
        copy = arr.reshape(d0,d2,d1);
      } else {
        auto in = arr.reshape(d0,d1,d2);
        copy = arr.createDeviceCopy().reshape(d0,d2,d1);
        c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) { copy(i0,i2,i1) = in(i0,i1,i2); });
      }
      // Perform the FFT
      #if   defined(YAKL_ARCH_CUDA)
        if        constexpr (std::is_same<T,float >::value) {
          CHECK( cufftExecC2R(plan_inverse, (cufftComplex       *) copy.data(), (cufftReal       *) copy.data()) );
        } else if constexpr (std::is_same<T,double>::value) {
          CHECK( cufftExecZ2D(plan_inverse, (cufftDoubleComplex *) copy.data(), (cufftDoubleReal *) copy.data()) );
        }
      #elif defined(YAKL_ARCH_HIP)
        std::array<double *,1> ibuf( {copy.data()} );
        std::array<double *,1> obuf( {copy.data()} );
        rocfft_execution_info info;
        CHECK( rocfft_execution_info_create( &info ) );
        CHECK( rocfft_execute(plan_inverse, (void **) ibuf.data(), (void **) obuf.data(), info) );
        CHECK( rocfft_execution_info_destroy( info ) );
      #else
        Array<std::complex<T>,3,memHost,styleC> pfft_in ("pfft_in" ,d0,d2,(transform_size+2)/2);
        Array<T              ,3,memHost,styleC> pfft_out("pfft_out",d0,d2, transform_size     );
        auto copy_host = copy.createHostCopy();
        for (int i0 = 0; i0 < d0; i0++) {
          for (int i2 = 0; i2 < d2; i2++) {
            for (int i1 = 0; i1 < (transform_size+2)/2; i1++) {
              pfft_in(i0,i2,i1) = std::complex<T>( copy_host(i0,i2,2*i1  ) , copy_host(i0,i2,2*i1+1) );
            }
          }
        }
        using pocketfft::detail::shape_t;
        using pocketfft::detail::stride_t;
        shape_t  shape_out (3);   for (int i=0; i < 3; i++) { shape_out [i] = pfft_out.extent(i); }
        stride_t stride_in (3);
        stride_t stride_out(3);
        stride_in [0] = d2*((transform_size+2)/2)*sizeof(std::complex<T>);
        stride_in [1] =    ((transform_size+2)/2)*sizeof(std::complex<T>);
        stride_in [2] =                           sizeof(std::complex<T>);   
        stride_out[0] = d2*  transform_size      *sizeof(             T );
        stride_out[1] =      transform_size      *sizeof(             T );
        stride_out[2] =                           sizeof(             T );                 
        pocketfft::c2r<T>(shape_out, stride_in, stride_out, (size_t) 2, false, pfft_in.data() , pfft_out.data() , (T) 1 );
        for (int i0 = 0; i0 < d0; i0++) {
          for (int i2 = 0; i2 < d2; i2++) {
            for (int i1 = 0; i1 < transform_size; i1++) {
              copy_host(i0,i2,i1) = pfft_out(i0,i2,i1);
            }
          }
        }
        copy_host.deep_copy_to( copy );
        fence();
      #endif
      auto out = arr.reshape(d0,d1,d2);
      YAKL_SCOPE( transform_size , this->transform_size );
      c::parallel_for( c::SimpleBounds<3>(d0,d1,d2) , YAKL_LAMBDA (int i0, int i1, int i2) { out(i0,i1,i2) = copy(i0,i2,i1) / transform_size; });
    }

  };

}

