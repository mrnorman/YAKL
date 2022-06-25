
#include <iostream>
#include "YAKL.h"

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
        fft.init(n);

        real3d data("data",ntot,ntot,ntot);

        yakl::c::parallel_for( yakl::c::Bounds<3>(n,n,n) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = i+1;
        });

        fft.forward_real( data , trdim );

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

        fft.inverse_real( data , trdim );

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
        fft.init(n);

        real3d data("data",ntot,ntot,ntot);

        yakl::c::parallel_for( yakl::c::Bounds<3>(n,n,n) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = j+1;
        });

        fft.forward_real( data , trdim );

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

        fft.inverse_real( data , trdim );

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
        fft.init(n);

        real3d data("data",ntot,ntot,ntot);

        yakl::c::parallel_for( yakl::c::Bounds<3>(n,n,n) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = k+1;
        });

        fft.forward_real( data , trdim );

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

        fft.inverse_real( data , trdim );

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



    {
      int n = 32;
      int ntot = n%2==1 ? n+1 : n+2;
      typedef float real;
      typedef yakl::Array<real,1,yakl::memHost  ,yakl::styleC> realHost1d;
      typedef yakl::Array<real,3,yakl::memDevice,yakl::styleC> real3d;
      yakl::RealFFT1D<real> fft;
      fft.init(n);
      real3d data("data",ntot,ntot,ntot);
      yakl::c::parallel_for( yakl::c::Bounds<3>(n,n,n) , YAKL_LAMBDA (int k, int j, int i) {
        data(k,j,i) = k*n*n + j*n + i;
      });

      {
        int trdim = 0;
        yakl::timer_start("trdim_0_loop");
        for (int i=0; i < 10; i++) {
          fft.forward_real( data , trdim );
          fft.inverse_real( data , trdim );
        }
        yakl::timer_stop("trdim_0_loop");
      }

      {
        int trdim = 1;
        yakl::timer_start("trdim_1_loop");
        for (int i=0; i < 10; i++) {
          fft.forward_real( data , trdim );
          fft.inverse_real( data , trdim );
        }
        yakl::timer_stop("trdim_1_loop");
      }

      {
        int trdim = 2;
        yakl::timer_start("trdim_2_loop");
        for (int i=0; i < 10; i++) {
          fft.forward_real( data , trdim );
          fft.inverse_real( data , trdim );
        }
        yakl::timer_stop("trdim_2_loop");
      }

    }



    {
      int nz = 50;
      int ny = 32;
      int nx = 32;
      int ncrms = 10;
      typedef double real;
      typedef yakl::Array<real,1,yakl::memHost  ,yakl::styleC> realHost1d;
      typedef yakl::Array<real,3,yakl::memDevice,yakl::styleC> real3d;
      typedef yakl::Array<real,4,yakl::memDevice,yakl::styleC> real4d;
      yakl::RealFFT1D<real> fft_y;
      yakl::RealFFT1D<real> fft_x;
      fft_y.init(ny);
      fft_x.init(nx);
      real4d data("data",nz,ny+2,nx+2,ncrms);
      yakl::c::parallel_for( yakl::c::Bounds<4>(nz,ny,nx,ncrms) , YAKL_LAMBDA (int k, int j, int i, int icrm) {
        data(k,j,i,icrm) = k*ny*nx*ncrms + j*nx*ncrms + i*ncrms + icrm;
      });

      {
        yakl::timer_start("crm_3d");
        for (int i=0; i < 10; i++) {
          fft_x.forward_real( data , 2 );
          fft_y.forward_real( data , 1 );
          fft_y.inverse_real( data , 1 );
          fft_x.inverse_real( data , 2 );
        }
        yakl::timer_stop("crm_3d");
      }

    }



    {
      int nz = 50;
      int ny = 1;
      int nx = 128;
      int ncrms = 10;
      typedef double real;
      typedef yakl::Array<real,1,yakl::memHost  ,yakl::styleC> realHost1d;
      typedef yakl::Array<real,3,yakl::memDevice,yakl::styleC> real3d;
      typedef yakl::Array<real,4,yakl::memDevice,yakl::styleC> real4d;
      yakl::RealFFT1D<real> fft_x;
      fft_x.init(nx);
      real4d data("data",nz,ny,nx+2,ncrms);
      yakl::c::parallel_for( yakl::c::Bounds<4>(nz,ny,nx,ncrms) , YAKL_LAMBDA (int k, int j, int i, int icrm) {
        data(k,j,i,icrm) = k*ny*nx*ncrms + j*nx*ncrms + i*ncrms + icrm;
      });

      {
        yakl::timer_start("crm_2d");
        for (int i=0; i < 10; i++) {
          fft_x.forward_real( data , 2 );
          fft_x.inverse_real( data , 2 );
        }
        yakl::timer_stop("crm_2d");
      }

    }


  }
  yakl::finalize();
}


