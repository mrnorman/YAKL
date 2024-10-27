
#pragma once

#include "YAKL.h"
#if !defined(YAKL_ARCH_CUDA) && !defined(YAKL_ARCH_HIP) && !defined(YAKL_ARCH_SYCL)
  #define POCKETFFT_CACHE_SIZE 2
  #define POCKETFFT_NO_MULTITHREADING
  #include "pocketfft_hdronly.h"
#endif

#if defined(YAKL_ARCH_SYCL) && defined(YAKL_SYCL_BBFFT) && defined(YAKL_SYCL_BBFFT_AOT)
extern std::uint8_t _binary_kernels_bin_start, _binary_kernels_bin_end;
#endif

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  /** @brief Compute batched real-to-complex forward and inverse FFTs on yakl::Array objects using vendor libraries.
    * 
    * This class uses metadata from the yakl::Array object to provide a simplified interface for computing
    * 1-D real-to-complex FFTs batched over the non-transformed dimensions of the Array object.
    * If the user calls `init()`, then calls to forward_real() and reverse_real() do not need the optional transform
    * dimension and transform size parameters. The user can avoid calling `init()`, though, and then provide
    * these parameters to `forward_real()` and `reverse_real()`.
    * 
    * Complex results of a forward transform are stored in interleaved real,imag,real,imag format in-place in the array.
    * 
    * Since it's real-to-complex, for even-element transforms, you'll need `n+2` elements available in the transform dimension(s).
    * For odd-element transforms, you'll need `n+1`.
    * 
    * Example Usage:
    * ```
    * int nz = 100, ny = 50, nx = 40;
    * // Allocate space for forward transforms (even numbers add 2, odd numbers add 1)
    * Array<float,3,memDevice,styleC> data("data",nz,ny+2,nx+2);
    * // Initialize data
    * RealFFT<float> fft_y;
    * RealFFT<float> fft_x;
    * fft_x.init(data , 2 , nx );
    * fft_y.init(data , 1 , ny );
    * // Forward transform
    * fft_x.forward_real( data );  // Batched over y and z dimensions
    * fft_y.forward_real( data );  // Batched over x and z dimensions
    * // Do stuff in Fourier space
    * fft_y.inverse_real( data );  // Batched over x and z dimensions
    * fft_x.inverse_real( data );  // Batched over y and z dimensions
    * // Do stuff in physical space
    * ```
    *
    * Twiddle and chirp factors are not re-computed unless `init()` is called or forward_real() is called
    * with a different dimension to transform or a different transform size (or the batch size changes).
    */
  template <class T> class RealFFT1D {
  public:
    /** @private */
    int batch_size;
    /** @private */
    int transform_size;
    /** @private */
    int trdim;
    #if   defined(YAKL_ARCH_CUDA)
      cufftHandle plan_forward;
      cufftHandle plan_inverse;
      #define CHECK(func) { int myloc = func; if (myloc != CUFFT_SUCCESS) { std::cerr << "ERROR: YAKL CUFFT: " << __FILE__ << ": " <<__LINE__ << std::endl; yakl_throw(""); } }
    #elif defined(YAKL_ARCH_HIP)
      rocfft_plan plan_forward;
      rocfft_plan plan_inverse;
      #define CHECK(func) { int myloc = func; if (myloc != rocfft_status_success) { std::cerr << "ERROR: YAKL ROCFFT: " << __FILE__ << ": " <<__LINE__ << std::endl; yakl_throw(""); } }
    #elif defined(YAKL_ARCH_SYCL)
      
      #if defined(YAKL_SYCL_BBFFT)
        bbfft::plan<sycl::event> plan_forward, plan_inverse;      
      #else
        typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL> desc_single_t;
        typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> desc_double_t;
        void *plan_forward, *plan_inverse;
        #define CHECK(func) {                                                        \
          try {                                                                      \
            func;                                                                    \
          }                                                                          \
          catch (oneapi::mkl::exception const &ex) {                                 \
            std::cerr << "ERROR: YAKL ONEMKL-FFT: " << __FILE__ << " : " << __LINE__ \
            << std::endl; yakl_throw("");                                            \
          }                                                                          \
        }
      #endif
    #endif

    void nullify() {batch_size = -1;  transform_size = -1;  trdim = -1;}

    RealFFT1D() {
      #ifdef YAKL_ARCH_HIP
        if (! get_yakl_instance().rocfft_is_initialized) {
          rocfft_setup();
          get_yakl_instance().rocfft_is_initialized = true;
        }
      #endif
      nullify();
    }
    ~RealFFT1D() { cleanup(); }

    /** @private */
    void cleanup() {
      if (transform_size != -1) {
        #if   defined(YAKL_ARCH_CUDA)
          CHECK( cufftDestroy( plan_forward ) );
          CHECK( cufftDestroy( plan_inverse ) );
        #elif defined(YAKL_ARCH_HIP)
          CHECK( rocfft_plan_destroy( plan_forward ) );
          CHECK( rocfft_plan_destroy( plan_inverse ) );
        #endif
      }
      nullify();
    }

    /** @brief Setup FFT plans, allocate, compute needed data.
      * @details This is not a necessary call. You can pass the `trdim` and `tr_size` parameters to forward_real() and inverse_real() if you want. */
    template <int N> void init( Array<T,N,memDevice,styleC> &arr , int trdim , int tr_size ) {
      int rank    = 1;
      int n       = tr_size;
      int istride = 1;
      int ostride = 1;
      int idist   = arr.extent(trdim);
      int odist   = idist / 2;
      int batch   = arr.totElems() / arr.extent(trdim);
      int *inembed = nullptr;
      int *onembed = nullptr;

      #if   defined(YAKL_ARCH_CUDA)
        if        constexpr (std::is_same<T,float >::value) {
          CHECK( cufftPlanMany(&plan_forward, rank, &n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch) );
          CHECK( cufftPlanMany(&plan_inverse, rank, &n, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, batch) );
        } else if constexpr (std::is_same<T,double>::value) {
          CHECK( cufftPlanMany(&plan_forward, rank, &n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batch) );
          CHECK( cufftPlanMany(&plan_inverse, rank, &n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batch) );
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
      #elif defined(YAKL_ARCH_SYCL)
        #if defined(YAKL_SYCL_BBFFT)
          
          auto precision = std::is_same_v<T,float> ? bbfft::precision::f32 : bbfft::precision::f64;  

          bbfft::configuration cfg_forward = {1, {1, (unsigned long) tr_size, (unsigned long) batch}, precision, bbfft::direction::forward, bbfft::transform_type::r2c};
          cfg_forward.set_strides_default(true);

          bbfft::configuration cfg_inverse = {1, {1, (unsigned long) tr_size, (unsigned long) batch}, precision, bbfft::direction::backward, bbfft::transform_type::c2r};
          cfg_inverse.set_strides_default(true);

          #if defined(YAKL_SYCL_BBFFT_AOT)
            auto cache = bbfft::aot_cache{};
            try {
              cache.register_module(bbfft::sycl::create_aot_module(&_binary_kernels_bin_start, 
                                                                   &_binary_kernels_bin_end - &_binary_kernels_bin_start,
                                                                   bbfft::module_format::native, sycl_default_stream().get_context(), sycl_default_stream().get_device()));
            } catch (std::exception const &e) {
              std::cerr << "Could not load ahead-of-time compiled FFT kernels:" << std::endl << e.what() << std::endl;
            }
            plan_forward = bbfft::make_plan(cfg_forward, sycl_default_stream(), &cache);
            plan_inverse = bbfft::make_plan(cfg_inverse, sycl_default_stream(), &cache);
          #else
            plan_forward = bbfft::make_plan(cfg_forward, sycl_default_stream());
            plan_inverse = bbfft::make_plan(cfg_inverse, sycl_default_stream());
          #endif

        #else
          if constexpr (std::is_same_v<T,float>) {
            plan_forward = new desc_single_t(n);
            plan_inverse = new desc_single_t(n);

            CHECK( static_cast<desc_single_t*>(plan_forward)->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch) );
            CHECK( static_cast<desc_single_t*>(plan_forward)->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,         idist) );
            CHECK( static_cast<desc_single_t*>(plan_forward)->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,         odist) );

            CHECK( static_cast<desc_single_t*>(plan_inverse)->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch) );
            CHECK( static_cast<desc_single_t*>(plan_inverse)->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,         odist) );
            CHECK( static_cast<desc_single_t*>(plan_inverse)->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,         idist) );

            CHECK( static_cast<desc_single_t*>(plan_forward)->commit( sycl_default_stream() ) );
            CHECK( static_cast<desc_single_t*>(plan_inverse)->commit( sycl_default_stream() ) );
          } else if constexpr (std::is_same_v<T,double>) {
            plan_forward = new desc_double_t(n);
            plan_inverse = new desc_double_t(n);

            CHECK( static_cast<desc_double_t*>(plan_forward)->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch) );
            CHECK( static_cast<desc_double_t*>(plan_forward)->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,         idist) );
            CHECK( static_cast<desc_double_t*>(plan_forward)->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,         odist) );

            CHECK( static_cast<desc_double_t*>(plan_inverse)->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch) );
            CHECK( static_cast<desc_double_t*>(plan_inverse)->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,         idist) );
            CHECK( static_cast<desc_double_t*>(plan_inverse)->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,         odist) );

            CHECK( static_cast<desc_double_t*>(plan_forward)->commit( sycl_default_stream() ) );
            CHECK( static_cast<desc_double_t*>(plan_inverse)->commit( sycl_default_stream() ) );
          }
        #endif

      #endif

      this->batch_size     = batch  ;
      this->transform_size = tr_size;
      this->trdim          = trdim  ;
    }


    /** @brief Perform a forward transform (real-to-complex)
      * @details `trdim_in` and `transform_size_in` are only needed if you did not call `init()` or you're changing
      * the parameters of the transform (batch size, transform dim, transform size). */
    template <int N> void forward_real( Array<T,N,memDevice,styleC> &arr , int trdim_in = -1 , int transform_size_in = -1 ) {
      if constexpr (streams_enabled) fence();
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
      #elif defined(YAKL_ARCH_SYCL)
        #if defined(YAKL_SYCL_BBFFT)
          plan_forward.execute(static_cast<T*>(copy.data()));          
        #else
          if        constexpr (std::is_same<T,float >::value) {
            CHECK( oneapi::mkl::dft::compute_forward(*(static_cast<desc_single_t*>(plan_forward)), static_cast<T*>(copy.data())) );
          } else if constexpr (std::is_same<T,double>::value) {
            CHECK( oneapi::mkl::dft::compute_forward(*(static_cast<desc_double_t*>(plan_forward)), static_cast<T*>(copy.data())) );
          }
        #endif
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
      if constexpr (streams_enabled) fence();
    }


    /** @brief Perform an inverse transform (complex-to-real)
      * @details `trdim_in` and `transform_size_in` are only needed if you did not call `init()` or you're changing
      * the parameters of the transform (batch size, transform dim, transform size). */
    template <int N> void inverse_real( Array<T,N,memDevice,styleC> &arr , int trdim_in = -1 , int transform_size_in = -1 ) {
      if constexpr (streams_enabled) fence();
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
      #elif defined(YAKL_ARCH_SYCL)
        #if defined(YAKL_SYCL_BBFFT)
          plan_inverse.execute(static_cast<T*>(copy.data()));
        #else
          if        constexpr (std::is_same<T,float >::value) {
            CHECK( oneapi::mkl::dft::compute_backward(*(static_cast<desc_single_t*>(plan_inverse)), static_cast<T*>(copy.data())) );
          } else if constexpr (std::is_same<T,double>::value) {
            CHECK( oneapi::mkl::dft::compute_backward(*(static_cast<desc_double_t*>(plan_inverse)), static_cast<T*>(copy.data())) );
          }
        #endif
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
      if constexpr (streams_enabled) fence();
    }

  };

}
__YAKL_NAMESPACE_WRAPPER_END__

