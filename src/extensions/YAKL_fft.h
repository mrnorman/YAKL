
#pragma once

#include "YAKL.h"

namespace yakl {

  template <class T> class Complex {
  protected:
    T re, im;
  public:
    YAKL_INLINE Complex(T re, T im) { this->re=re; this->im=im; }
    YAKL_INLINE Complex(T re      ) { this->re=re; this->im=0 ; }
    YAKL_INLINE Complex(          ) { this->re=0 ; this->im=0 ; }
    YAKL_INLINE ~Complex() { }
    YAKL_INLINE Complex(Complex<T> const  &rhs) { re=rhs.re; im=rhs.im; }
    YAKL_INLINE Complex(Complex<T> const &&rhs) { re=rhs.re; im=rhs.im; }
    YAKL_INLINE Complex<T> &operator=(Complex<T> const  &rhs) { re=rhs.re; im=rhs.im; return *this; }
    YAKL_INLINE Complex<T> &operator=(Complex<T> const &&rhs) { re=rhs.re; im=rhs.im; return *this; }
    YAKL_INLINE T real() const { return re; }
    YAKL_INLINE T imag() const { return im; }
    YAKL_INLINE T &real() { return re; }
    YAKL_INLINE T &imag() { return im; }
    inline friend std::ostream &operator<<(std::ostream& os, Complex<T> const &c) {
      os << "(" << c.real() << " , " << c.imag() << ")";
      return os;
    }
  };
  template <class T> YAKL_INLINE Complex<T> operator-(Complex<T> const &rhs) {
    return Complex<T>(-rhs.real(),-rhs.imag());
  }
  template <class T> YAKL_INLINE Complex<T> operator+(Complex<T> const &c1, Complex<T> const &c2) {
    T x = c1.real();   T y = c1.imag();   T u = c2.real();   T v = c2.imag();
    return Complex<T>( x+u , y+v );
  }
  template <class T> YAKL_INLINE Complex<T> operator-(Complex<T> const &c1, Complex<T> const &c2) {
    T x = c1.real();   T y = c1.imag();   T u = c2.real();   T v = c2.imag();
    return Complex<T>( x-u , y-v );
  }
  template <class T> YAKL_INLINE Complex<T> operator*(Complex<T> const &c1, Complex<T> const &c2) {
    T x = c1.real();   T y = c1.imag();   T u = c2.real();   T v = c2.imag();
    return Complex<T>( x*u - y*v , x*v + y*u );
  }
  template <class T> YAKL_INLINE Complex<T> operator/(Complex<T> const &c, T d) {
    return Complex<T>( c.real()/d , c.imag()/d );
  }
  template <class T> YAKL_INLINE Complex<T> exp(Complex<T> const &c) {
    T x = c.real();   T y = c.imag();
    return Complex<T>( std::exp(x)*cos(y) , std::exp(x)*sin(y) );
  }
  template <class T> YAKL_INLINE Complex<T> conj(Complex<T> const &c) { return Complex<T>(c.real(),-c.imag()); }



  template <class T> class RealFFT1D {
  public:
    typedef Array<Complex<T>,1,memDevice,styleC> complex1d;
    typedef Array<Complex<T>,2,memDevice,styleC> complex2d;

    struct Internals {
      complex1d    omega_m      ;  // For Cooley-Tukey algorithm
      complex1d    chirp        ;  // For chirp-z transform (non power of two FFT)
      complex2d    chirp_fft    ;  // For chirp-z transform (non power of two FFT)
      complex2d    chirp_tmp    ;  // For chirp-z transform (non power of two FFT)
      unsigned int data_size    ;  // Number of data being transformed
      unsigned int fft_size     ;  // Size of the power of two Cooley-Tukey transform
      unsigned int log2_fft_size;  // log2(fft_size)
      bool         power_of_two ;  // Is data_size a power of two?
    };

    Internals internals;

    // Reverses the bits of an unsigned integer with the specified number of bits (for Cooley-Tukey algorithm)
    YAKL_INLINE static unsigned int reverse_bits(unsigned int num, unsigned num_bits) {
      unsigned int reverse_num = 0;
      for (int i = 0; i < num_bits; i++) { if ((num & (1 << i))) reverse_num |= 1 << ((num_bits - 1) - i); }
      return reverse_num;
    }


    // Does an in-place bit-reverse copy of the data
    YAKL_INLINE static void bit_reverse_copy_inplace(complex2d const &data, unsigned int size, unsigned int num_bits,
                                                     unsigned int ibatch) {
      for (int i=0; i < size; i++) {
        auto j = reverse_bits(i,num_bits);
        if (i < j) { auto tmp = data(i,ibatch);  data(i,ibatch) = data(j,ibatch);  data(j,ibatch) = tmp; }
      }
    }


    // Perform a forward complex-to-complex FFT of arbitrary size
    YAKL_INLINE static void forward(complex2d const &data, Internals const &internals, unsigned int ibatch) {
      auto &chirp        = internals.chirp       ;
      auto &chirp_fft    = internals.chirp_fft   ;
      auto &data_size    = internals.data_size   ;
      auto &fft_size     = internals.fft_size    ;
      auto &power_of_two = internals.power_of_two;
      auto &chirp_tmp    = internals.chirp_tmp   ;
      if (power_of_two) {
        forward_pow2( data , internals , ibatch );  // Vanilla Cooley-Tukey
      } else {
        // Chirp-z transform (zero-padding + convolution + Cooley-Tukey)
        for (int i=0; i < data_size; i++) { chirp_tmp(i,ibatch) = data(i,ibatch)*chirp(i); }
        for (int i=0; i < fft_size-data_size; i++) { chirp_tmp(data_size+i,ibatch) = 0; }
        forward_pow2( chirp_tmp , internals , ibatch );
        for (int i=0; i < fft_size; i++) { chirp_tmp(i,ibatch) = chirp_tmp(i,ibatch) * chirp_fft(i,0); }
        inverse_pow2( chirp_tmp , internals , ibatch );
        for (int i=0; i < data_size; i++) { data(i,ibatch) = chirp_tmp(i,ibatch) * chirp(i); }
      }
    }


    // Perform a forward complex-to-complex FFT of power of two size using Cooley-Tukey
    YAKL_INLINE static void forward_pow2(complex2d const &data, Internals const &internals, unsigned int ibatch) {
      auto &omega_m       = internals.omega_m      ;
      auto &fft_size      = internals.fft_size     ;
      auto &log2_fft_size = internals.log2_fft_size;
      // Classic Cooley-Tukey non-recursive algorithm
      bit_reverse_copy_inplace( data , fft_size , log2_fft_size , ibatch );
      unsigned int m = 1;
      for (int s=1; s <= log2_fft_size; s++) {
        m *= 2;
        for (int k=0; k < fft_size; k+=m) {
          Complex<T> omega = 1;
          for (int j=0; j < m/2; j++) {
            Complex<T> t = omega * data(k+j+m/2,ibatch);
            Complex<T> u =         data(k+j    ,ibatch);
            data(k+j    ,ibatch) = u+t;
            data(k+j+m/2,ibatch) = u-t;
            omega = omega * omega_m(s-1);
          }
        }
      }
    }


    // Compute inverse complex-to-complex FFT of arbitrary size using the forward FFT and complex conjugation.
    // Scale the data so that you get back the original values.
    YAKL_INLINE static void inverse(complex2d const &data, Internals const &internals, unsigned int ibatch) {
      auto &data_size = internals.data_size;
      for (int i=0; i < data_size; i++) { data(i,ibatch) = conj( data(i,ibatch) ); }
      forward( data , internals , ibatch );
      for (int i=0; i < data_size; i++) { data(i,ibatch) = conj( data(i,ibatch) ) / (T) data_size; }
    }


    // Compute an in-place inverse transform of size power of two
    YAKL_INLINE static void inverse_pow2(complex2d const &data, Internals const &internals, unsigned int ibatch) {
      auto &fft_size = internals.fft_size;
      for (int i=0; i < fft_size; i++) { data(i,ibatch) = conj( data(i,ibatch) ); }
      forward_pow2( data , internals , ibatch );
      for (int i=0; i < fft_size; i++) { data(i,ibatch) = conj( data(i,ibatch) ) / (T) fft_size; }
    }


    RealFFT1D() {
      internals.data_size     = 0;
      internals.fft_size      = 0;
      internals.log2_fft_size = 0;
      internals.power_of_two  = false;
    }
    ~RealFFT1D() { internals.omega_m = complex1d();  internals.chirp = complex1d();   internals.chirp_fft = complex2d(); }


    void init(unsigned int size) {
      using c::parallel_for;
      using c::SimpleBounds;
      auto &omega_m       = internals.omega_m      ;
      auto &chirp         = internals.chirp        ;
      auto &chirp_fft     = internals.chirp_fft    ;
      auto &data_size     = internals.data_size    ;
      auto &fft_size      = internals.fft_size     ;
      auto &log2_fft_size = internals.log2_fft_size;
      auto &power_of_two  = internals.power_of_two ;
      YAKL_SCOPE( internals , this->internals );
      power_of_two  = size == (unsigned int) std::exp2( (unsigned int) std::log2( size ) );
      data_size     = size;
      fft_size      = power_of_two ? data_size : (unsigned int) std::exp2( std::ceil( std::log2( size*2-1 ) ) );
      log2_fft_size = std::log2(fft_size);
      // Term needed for Cooley-Tukey algorithm (FFT for power of two size)
      omega_m = complex1d("yakl_internal_fft_omega_m",log2_fft_size);
      parallel_for( "yakl_internal_fft_initialize_omega_m" , log2_fft_size , YAKL_LAMBDA (int s) {
        omega_m(s) = exp( Complex<T>(0,-2.*M_PI/std::exp2(s+1)) );
      });
      // Term needed for Chirp-Z transform for FFTs of non power of two size
      if (! power_of_two) {
        chirp = complex1d("yakl_internal_fft_chirp",size);
        parallel_for( "yakl_internal_fft_initialize_chirp" , size , YAKL_LAMBDA (int i) {
          chirp(i) = exp( Complex<T>(0,-M_PI*i*i/size) );
        });
        chirp_fft = complex2d("yakl_internal_fft_chirp_fft",fft_size,1);
        parallel_for( "yakl_internal_fft_initialize_chipr_fft" , 1 , YAKL_LAMBDA (int dummy) {
          for (int i=0; i < data_size; i++) { chirp_fft(i,0) = conj(chirp(i)); }
          for (int i=0; i < fft_size-2*data_size+1; i++) { chirp_fft(data_size+i,0) = 0; }
          for (int i=0; i < data_size-1; i++) { chirp_fft(fft_size-(data_size-1)+i,0) = conj(chirp(data_size-1-i)); }
          forward_pow2( chirp_fft , internals , 0 );
        });
      }
    }


    template <int N>
    void forward_real( Array<T,N,memDevice,styleC> &data_in , int trdim ) {
      auto &data_size    = internals.data_size   ;
      auto &fft_size     = internals.fft_size    ;
      auto &chirp_tmp    = internals.chirp_tmp   ;
      auto &power_of_two = internals.power_of_two;
      YAKL_SCOPE( internals , this->internals );
      // Reshape input array into a 3-D array for convenience
      unsigned int dim_slowest   = 1;    for (int i=0  ; i < trdim; i++) { dim_slowest *= data_in.extent(i); }
      unsigned int dim_fastest   = 1;    for (int i=N-1; i > trdim; i--) { dim_fastest *= data_in.extent(i); }
      unsigned int dim_transform = data_in.extent(trdim);
      unsigned int batch_size    = dim_slowest * dim_fastest;
      auto data = data_in.reshape(dim_slowest,dim_transform,dim_fastest);
      if ( (data_size%2==1 ? data_size+1 : data_size+2) != dim_transform ) {
        yakl_throw("ERROR: dimension being transformed is not consistent with the initialized FFT size");
      }
      // Split batches into groups of two to use the "two for the price of one" real FFT algorithm
      complex2d aggregate  ("yakl_internal_fft_aggregate",data_size,(batch_size+1)/2);
      if (! power_of_two) chirp_tmp = complex2d("yakl_internal_fft_chirp_tmp",fft_size ,(batch_size+1)/2);
      c::parallel_for( "yakl_internal_fft_forward_real" , (batch_size+1)/2 , YAKL_LAMBDA (int ibatch_in) {
        unsigned int ibatch  = (unsigned int) ibatch_in;
        unsigned int ibatch1 = 2*ibatch;
        unsigned int ibatch2 = std::min( 2*ibatch+1 , batch_size-1 );
        unsigned int islow1 = ibatch1 / dim_fastest;
        unsigned int ifast1 = ibatch1 % dim_fastest;
        unsigned int islow2 = ibatch2 / dim_fastest;
        unsigned int ifast2 = ibatch2 % dim_fastest;
        for (int i=0; i < data_size; i++) {
          aggregate(i,ibatch).real() = data(islow1,i,ifast1);
          aggregate(i,ibatch).imag() = data(islow2,i,ifast2);
        }
        forward( aggregate , internals , ibatch );
        data(islow1,0,ifast1) = aggregate(0,ibatch).real();
        data(islow1,1,ifast1) = 0;
        data(islow2,0,ifast2) = aggregate(0,ibatch).imag();
        data(islow2,1,ifast2) = 0;
        for (int i=1; i < (data_size+2)/2; i++) {
          int ind_real = 2*i;
          int ind_imag = 2*i+1;
          data(islow1,ind_real,ifast1) = ( aggregate(i,ibatch).real()+aggregate(data_size-i,ibatch).real())/2;
          data(islow1,ind_imag,ifast1) = ( aggregate(i,ibatch).imag()-aggregate(data_size-i,ibatch).imag())/2;
          data(islow2,ind_real,ifast2) = ( aggregate(i,ibatch).imag()+aggregate(data_size-i,ibatch).imag())/2;
          data(islow2,ind_imag,ifast2) = (-aggregate(i,ibatch).real()+aggregate(data_size-i,ibatch).real())/2;
        }
      });
    }


    template <int N>
    void inverse_real( Array<T,N,memDevice,styleC> &data_in , int trdim ) {
      auto &data_size    = internals.data_size   ;
      auto &fft_size     = internals.fft_size    ;
      auto &chirp_tmp    = internals.chirp_tmp   ;
      auto &power_of_two = internals.power_of_two;
      YAKL_SCOPE( internals , this->internals );
      // Reshape input array into a 3-D array for convenience
      unsigned int dim_slowest   = 1;    for (int i=0  ; i < trdim; i++) { dim_slowest *= data_in.extent(i); }
      unsigned int dim_fastest   = 1;    for (int i=N-1; i > trdim; i--) { dim_fastest *= data_in.extent(i); }
      unsigned int dim_transform = data_in.extent(trdim);
      unsigned int batch_size    = dim_slowest * dim_fastest;
      auto data = data_in.reshape(dim_slowest,dim_transform,dim_fastest);
      if ( (data_size%2==1 ? data_size+1 : data_size+2) != dim_transform ) {
        yakl_throw("ERROR: dimension being transformed is not consistent with the initialized FFT size");
      }
      // Split batches into groups of two to use the "two for the price of one" real FFT algorithm
      complex2d aggregate  ("yakl_internal_fft_aggregate",dim_transform,(batch_size+1)/2);
      if (! power_of_two) chirp_tmp = complex2d("yakl_internal_fft_chirp_tmp",fft_size ,(batch_size+1)/2);
      c::parallel_for( "yakl_internal_fft_inverse_real" , (batch_size+1)/2 , YAKL_LAMBDA (int ibatch_in) {
        unsigned int ibatch  = (unsigned int) ibatch_in;
        unsigned int ibatch1 = 2*ibatch;
        unsigned int ibatch2 = std::min( 2*ibatch+1 , batch_size-1 );
        unsigned int islow1 = ibatch1 / dim_fastest;
        unsigned int ifast1 = ibatch1 % dim_fastest;
        unsigned int islow2 = ibatch2 / dim_fastest;
        unsigned int ifast2 = ibatch2 % dim_fastest;
        for (int i=0; i < (data_size+2)/2; i++) {
          unsigned int ind_real = 2*i;
          unsigned int ind_imag = 2*i+1;
          aggregate(i,ibatch).real() = data(islow1,ind_real,ifast1) - data(islow2,ind_imag,ifast2);
          aggregate(i,ibatch).imag() = data(islow1,ind_imag,ifast1) + data(islow2,ind_real,ifast2);
        }
        for (int i=(data_size+2)/2; i < data_size; i++) {
          unsigned int ind = data_size - i;
          unsigned int ind_real = 2*ind;
          unsigned int ind_imag = 2*ind+1;
          aggregate(i,ibatch).real() =  data(islow1,ind_real,ifast1) + data(islow2,ind_imag,ifast2);
          aggregate(i,ibatch).imag() = -data(islow1,ind_imag,ifast1) + data(islow2,ind_real,ifast2);
        }
        inverse( aggregate , internals , ibatch );
        for (int i=0; i < data_size; i++) {
          data(islow1,i,ifast1) = aggregate(i,ibatch).real();
          data(islow2,i,ifast2) = aggregate(i,ibatch).imag();
        }
        for (int i=data_size; i < dim_transform; i++) {
          data(islow1,i,ifast1) = 0;
          data(islow2,i,ifast2) = 0;
        }
      });
    }

  };

}

