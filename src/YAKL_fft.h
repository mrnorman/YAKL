
#pragma once

#include "YAKL.h"

/*************************************************************************************************************
**************************************************************************************************************
YAKL RealFFT1D class
Matt Norman, Oak Ridge National Laboratory, normanmr@ornl.gov

Based off the Cooley-Tukey iterative algorithm here:
https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

Using real-to-complex optimizations described here:
http://www.robinscheibler.org/2013/02/13/real-fft.html
https://dsp.stackexchange.com/questions/30185/fft-of-a-n-length-real-sequence-via-fft-of-a-n-2-length-complex-sequence

Inverse FFTs performed using method 4 of:
https://www.dsprelated.com/showarticle/800.php

This class provides a simple solution to performance portable real-to-complex 1-D FFTs. This is meant for
small FFTs, and in CUDA, if you use > 2^14 data, you will run out of stack memory in the kernels.
The data is assumed to be on the stack (the YAKL SArray class), but it can be on the heap as well (i.e., 
the YAKL Array class). The data size must be a power of 2.

The specific use case for this is when you have to do many 1-D FFTs (say, in a 3-D model). The call to
RealFFT1D.forward(...) and RealFFT1D.inverse(...) is intended to be inside a parallel_for loop.

The trig arrays are read-only and shared, so they are allocated when the class object is constructed and
deallocated when the class object is destroyed.

The class is templated on the size of the data because tmp data must be placed on the stack inside
the parallel_for in order to make it thread-private, and CUDA and other GPUs do not allow flexible sized
declarations of stack data during runtime.

The data array passed to RealFFT1D.forward(...) and RealFFT1D.inverse(...) must be of size n+2, where n
is the dimension of the data to be transformed. This is because we need to store n/2+1 Fourier modes.
Even though the imaginary components of the first and last modes are zero, they are still stored.

The FFTs are performed in place, overwriting the data.

There are two different types of data scaling: FFT_SCALE_STANDARD, which is like numpy.fft.rfft
                                               FFT_SCALE_ECMWF, taken from the ECMWF fft code

You'll notice some headache with passing around the RealFFT Trig struct. This is required on GPUs because
you cannot access internal class data on the GPU because the "this' pointer isn't valid in GPU memory.

Example usage:

  int constexpr nx = 128;
  int constexpr ny = 64;
  int constexpr nz = 72;

  Array<double,3,memDevice,styleC> pressure("pressure",nz,ny+2,nx+2);

  parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
    pressure(k,j,i) = ...;
  });

  yakl::RealFFT1D<n,double> fft_x;
  fft_x.init(fft_x.trig);

  yakl::RealFFT1D<n,double> fft_y;
  fft_y.init(fft_y.trig);

  // x-direction forward FFTs
  parallel_for( Bounds<2>(nz,ny) , YAKL_LAMBDA (int k, int j) {
    SArray<real,1,nx+2> data;
    for (int i=0; i < nx; i++) {
      data(i) = pressure(k,j,i);
    }
    fft_x.forward(data,fft_x.trig,yakl::FFT_SCALE_ECMWF);
    for (int i=0; i < nx+2; i++) {
      pressure(k,j,i) = data(i);
    }
  });

  // y-direction forward FFTs
  parallel_for( Bounds<2>(nz,nx+2) , YAKL_LAMBDA (int k, int i) {
    SArray<real,1,ny+2> data;
    for (int j=0; j < ny; j++) {
      data(j) = pressure(k,j,i);
    }
    fft_y.forward(data,fft_y.trig,yakl::FFT_SCALE_ECMWF);
    for (int j=0; j < ny+2; j++) {
      pressure(k,j,i) = data(j);
    }
  });

  // Do stuff in Fourier space

  // y-direction inverse FFTs
  parallel_for( Bounds<2>(nz,nx+2) , YAKL_LAMBDA (int k, int i) {
    SArray<real,1,ny+2> data;
    for (int j=0; j < ny+2; j++) {
      data(j) = pressure(k,j,i);
    }
    fft_y.inverse(data,fft_y.trig,yakl::FFT_SCALE_ECMWF);
    for (int j=0; j < ny; j++) {
      pressure(k,j,i) = data(j);
    }
  });

  // x-direction inverse FFTs
  parallel_for( Bounds<2>(nz,ny) , YAKL_LAMBDA (int k, int j) {
    SArray<real,1,nx+2> data;
    for (int i=0; i < nx+2; i++) {
      data(i) = pressure(k,j,i);
    }
    fft_x.inverse(data,fft_x.trig,yakl::FFT_SCALE_ECMWF);
    for (int i=0; i < nx; i++) {
      pressure(k,j,i) = data(i);
    }
  });
**************************************************************************************************************
*************************************************************************************************************/

namespace yakl {

  int constexpr FFT_SCALE_STANDARD = 1;
  int constexpr FFT_SCALE_ECMWF    = 2;

  template <unsigned int x> struct mylog2 { enum { value = 1 + mylog2<x/2>::value }; };
  template <> struct mylog2<1> { enum { value = 0 }; };

  template <unsigned int x> struct mypow2 { enum { value = 2*mypow2<x-1>::value }; };
  template <> struct mypow2<0> { enum { value = 1 }; };

  template <unsigned int SIZE, class real = double>
  class RealFFT1D {
    typedef Array<real,1,memDevice,styleC> real1d;

    public:

    int static constexpr OFF_COS1 = 0;
    int static constexpr OFF_SIN1 = mylog2<SIZE/2>::value;
    int static constexpr OFF_COS2 = 2*mylog2<SIZE/2>::value;
    int static constexpr OFF_SIN2 = 2*mylog2<SIZE/2>::value + SIZE/2;

    real1d trig;

    RealFFT1D() {
      trig = real1d("trig",2*mylog2<SIZE/2>::value + SIZE);
    }


    YAKL_INLINE ~RealFFT1D() {
      trig = real1d();
    }


    inline void init(real1d &trig) {
      int constexpr log2_no2 = mylog2<SIZE/2>::value;
      c::parallel_for( log2_no2 , YAKL_LAMBDA (int i) {
        unsigned int m = 1;
        for (int j=1; j <= i+1; j++) { m *= 2; }
        trig(OFF_COS1+i) = cos(2*M_PI/static_cast<real>(m));
        trig(OFF_SIN1+i) = sin(2*M_PI/static_cast<real>(m));
      });
      c::parallel_for( SIZE/2 , YAKL_LAMBDA (int i) {
        trig(OFF_COS2+i) = cos(2*M_PI*i/static_cast<real>(SIZE));
        trig(OFF_SIN2+i) = sin(2*M_PI*i/static_cast<real>(SIZE));
      });
    }


    template <class ARR>
    YAKL_INLINE void forward(ARR &data, real1d const &trig, int scale = FFT_SCALE_STANDARD) const {
      SArray<real,1,SIZE> tmp;
      int constexpr n = SIZE;
      bit_reverse_copy_real_forward(data,tmp);
      fft_post_bit_reverse(tmp,trig);

      // Post-process
      data(0) = tmp(0) + tmp(1);
      data(1) = 0;
      for (int k=1; k < n/2; k++) {
        real a = tmp(2*(    k)  );
        real b = tmp(2*(    k)+1);
        real c = tmp(2*(n/2-k)  );
        real d = tmp(2*(n/2-k)+1);
        real cterm = trig(OFF_COS2+k);
        real sterm = trig(OFF_SIN2+k);
        data(2*k  ) = 0.5*( (b+d)*cterm + (c-a)*sterm + a + c );
        data(2*k+1) = 0.5*( (c-a)*cterm - (b+d)*sterm + b - d );
      }
      data(2*(n/2)  ) = tmp(0) - tmp(1);
      data(2*(n/2)+1) = 0;
      if (scale == FFT_SCALE_ECMWF) {
        for (int k=0; k < n+2; k++) {
          data(k) /= n;
        }
      }
    }


    template <class ARR>
    YAKL_INLINE void inverse(ARR &data, real1d const &trig, int scale = FFT_SCALE_STANDARD) const {
      SArray<real,1,SIZE> tmp;
      int constexpr  n = SIZE;
      bit_reverse_copy_real_inverse(data,tmp,trig);
      fft_post_bit_reverse(tmp,trig);

      // Post-process
      for (int k=0; k < n/2; k++) {
        data(2*k  ) =  tmp(2*k  );
        data(2*k+1) = -tmp(2*k+1);
      }
      if (scale == FFT_SCALE_STANDARD) {
        for (int k=0; k < n; k++) {
          data(k) /= (n/2);
        }
      } else if (scale == FFT_SCALE_ECMWF) {
        for (int k=0; k < n; k++) {
          data(k) *= 2;
        }
      }
    }


    private:

    YAKL_INLINE unsigned int reverse_bits(unsigned int num) const {
      int constexpr num_bits = mylog2<SIZE/2>::value;
      unsigned int reverse_num = 0;
      int i;
      for (i = 0; i < num_bits; i++) {
        if ((num & (1 << i)))
          reverse_num |= 1 << ((num_bits - 1) - i);
      }
      return reverse_num;
    }


    template <class ARR_IN, class ARR_OUT>
    YAKL_INLINE void bit_reverse_copy_real_forward(ARR_IN const &in, ARR_OUT &out) const {
      int constexpr n = SIZE/2;
      for (unsigned int k=0; k < n; k++) {
        unsigned int br_ind = reverse_bits(k);
        out(2*br_ind  ) = in(2*k  );
        out(2*br_ind+1) = in(2*k+1);
      }
    }


    template <class ARR_IN, class ARR_OUT>
    YAKL_INLINE void bit_reverse_copy_real_inverse(ARR_IN const &in, ARR_OUT &out, real1d const &trig) const {
      int constexpr n = SIZE/2;
      for (unsigned int k=0; k < n; k++) {
        real a = in(2*k  );
        real b = in(2*k+1);
        real c = in(2*(n-k)  );
        real d = in(2*(n-k)+1);
        real cterm = trig(OFF_COS2+k);
        real sterm = trig(OFF_SIN2+k);
        real re = 0.5*( -(b+d)*cterm - (a-c)*sterm + a + c );
        real im = 0.5*(  (a-c)*cterm - (b+d)*sterm + b - d );
        unsigned int br_ind = reverse_bits(k);
        out(2*br_ind  ) =  re;
        out(2*br_ind+1) = -im;
      }
    }


    template <class ARR>
    YAKL_INLINE void fft_post_bit_reverse(ARR &data, real1d const &trig) const {
      unsigned int m = 1;
      int constexpr n = SIZE/2;
      int constexpr log2_n = mylog2<n>::value;
      for (unsigned int s = 1; s <= log2_n; s++) {
        m *= 2;
        real omega_m_re =  trig(OFF_COS1+s-1);
        real omega_m_im = -trig(OFF_SIN1+s-1);
        for (unsigned int k = 0; k < n; k+=m) {
          real omega_re = 1;
          real omega_im = 0;
          for (unsigned int j=0; j < m/2; j++) {
            real a = data(2*(k+j+m/2)  );
            real b = data(2*(k+j+m/2)+1);
            real c = data(2*(k+j    )  );
            real d = data(2*(k+j    )+1);
            data(2*(k+j    )  ) = -b*omega_im + a*omega_re + c;
            data(2*(k+j    )+1) =  a*omega_im + b*omega_re + d;
            data(2*(k+j+m/2)  ) =  b*omega_im - a*omega_re + c;
            data(2*(k+j+m/2)+1) = -a*omega_im - b*omega_re + d;
            real omega_new_re = -omega_im*omega_m_im + omega_m_re*omega_re;
            real omega_new_im =  omega_im*omega_m_re + omega_m_im*omega_re;
            omega_re = omega_new_re;
            omega_im = omega_new_im;
          }
        }
      }
    }

  };

}
