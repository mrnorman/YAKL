
#include <iostream>
#include "YAKL.h"
#include "YAKL_fft.h"

int main() {
  yakl::init();
  {
    {
      int n = 7;
      int ntot = n%2==1 ? n+1 : n+2;
      typedef double real;
      typedef yakl::Array<real,1,yakl::memHost  ,yakl::styleC> realHost1d;
      typedef yakl::Array<real,3,yakl::memDevice,yakl::styleC> real3d;

      realHost1d fft_exact("fft_exact",ntot);
      fft_exact(0) = 28.                ;
      fft_exact(1) = 0.                 ;
      fft_exact(2) = -3.5000000000000018;
      fft_exact(3) = 7.267824888003178  ;
      fft_exact(4) = -3.500000000000001 ;
      fft_exact(5) = 2.7911568610884143 ;
      fft_exact(6) = -3.5000000000000013;
      fft_exact(7) = 0.7988521603655251 ;

      {
        int trdim = 2;
        yakl::RealFFT1D<real> fft;

        real3d data("data",ntot,ntot,ntot);

        yakl::c::parallel_for( yakl::c::Bounds<3>(n,n,n) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = i+1;
        });

#ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("fft.forward_real");
#endif
        fft.forward_real( data , trdim , n );
#ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("fft.forward_real");
#endif

        {
          auto data_host = data.createHostCopy();
          for (int k=0; k < n; k++) {
            for (int j=0; j < n; j++) {
              for (int i=0; i < ntot; i++) {
                if (std::abs( data_host(k,j,i) - fft_exact(i) ) > 1.e-12) {
                  yakl::yakl_throw("ERROR: wrong forward FFT value in dim 2 transform");
                }
              }
            }
          }
        }

#ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("fft.inverse_real");
#endif
        fft.inverse_real( data , trdim , n );
#ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("fft.inverse_real");
#endif

        {
          auto data_host = data.createHostCopy();
          for (int k=0; k < n; k++) {
            for (int j=0; j < n; j++) {
              for (int i=0; i < n; i++) {
                if (std::abs( data_host(k,j,i) - (i+1) ) > 1.e-12) {
                  yakl::yakl_throw("ERROR: wrong inverse FFT value in dim 2 transform");
                }
              }
            }
          }
        }
      }


      {
        int trdim = 1;
        yakl::RealFFT1D<real> fft;

        real3d data("data",ntot,ntot,ntot);

        yakl::c::parallel_for( yakl::c::Bounds<3>(n,n,n) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = j+1;
        });

#ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("fft.forward_real");
#endif
        fft.forward_real( data , trdim , n );
#ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("fft.forward_real");
#endif

        {
          auto data_host = data.createHostCopy();
          for (int k=0; k < n; k++) {
            for (int i=0; i < n; i++) {
              for (int j=0; j < ntot; j++) {
                if ( std::abs( data_host(k,j,i) - fft_exact(j) ) > 1.e-12) {
                  yakl::yakl_throw("ERROR: wrong forward FFT value in dim 1 transform");
                }
              }
            }
          }
        }

#ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("fft.inverse_real");
#endif
        fft.inverse_real( data , trdim , n );
#ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("fft.inverse_real");
#endif

        {
          auto data_host = data.createHostCopy();
          for (int k=0; k < n; k++) {
            for (int i=0; i < n; i++) {
              for (int j=0; j < n; j++) {
                if ( std::abs( data_host(k,j,i) - (j+1) ) > 1.e-12 ) {
                  yakl::yakl_throw("ERROR: wrong inverse FFT value in dim 1 transform");
                }
              }
            }
          }
        }
      }


      {
        int trdim = 0;
        yakl::RealFFT1D<real> fft;

        real3d data("data",ntot,ntot,ntot);

        fft.init(data,trdim,n);

        yakl::c::parallel_for( yakl::c::Bounds<3>(n,n,n) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = k+1;
        });

#ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("fft.forward_real");
#endif
        fft.forward_real( data );
#ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("fft.forward_real");
#endif

        {
          auto data_host = data.createHostCopy();
          for (int j=0; j < n; j++) {
            for (int i=0; i < n; i++) {
              for (int k=0; k < ntot; k++) {
                if ( std::abs( data_host(k,j,i) - fft_exact(k) )  > 1.e-12 ) {
                  yakl::yakl_throw("ERROR: wrong forward FFT value in dim 0 transform");
                }
              }
            }
          }
        }

#ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("fft.inverse_real");
#endif
        fft.inverse_real( data );
#ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("fft.inverse_real");
#endif

        {
          auto data_host = data.createHostCopy();
          for (int j=0; j < n; j++) {
            for (int i=0; i < n; i++) {
              for (int k=0; k < n; k++) {
                if ( std::abs( data_host(k,j,i) - (k+1) ) > 1.e-12 ) {
                  yakl::yakl_throw("ERROR: wrong inverse FFT value in dim 0 transform");
                }
              }
            }
          }
        }
      }
    }


  }
  yakl::finalize();

  yakl::init();
  {
    yakl::Array<double, 1, yakl::memDevice, yakl::styleC> b("b", 1);
    int nx = 100;
    yakl::Array<double, 1, yakl::memDevice, yakl::styleC> a("a", nx+2);
    yakl::RealFFT1D<double> fft;
    fft.init(a, 0, nx);
    fft.forward_real(a);
  }
  yakl::finalize();

}


