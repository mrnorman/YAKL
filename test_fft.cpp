
#include <iostream>
#include "FArray.h"
#include "FSArray.h"
#include "YAKL.h"

// To compile for CPU:
// g++ -O3 BuddyAllocator.cpp YAKL.cpp test_fft.cpp

// To compile for Nvidia compute-capability 35 GPU:
// nvcc --expt-extended-lambda -x cu -arch sm_35 -D__USE_CUDA__ -D__MANAGED__ --std=c++14 BuddyAllocator.cpp YAKL.cpp test_fft.cpp -I${PATH_TO_CUB}


using yakl::SBnd;
using yakl::FArray;
using yakl::FSArray;
using yakl::FFT;
using yakl::fortran::parallel_for;
using yakl::storeIndices;
using yakl::fortran::Bounds;
using yakl::memDevice;


typedef double real;


YAKL_INLINE unsigned constexpr wrap( unsigned ind, unsigned n ) {
  return ind - (ind/n)*n;
}


int main() {
  unsigned constexpr nx = 64;
  unsigned constexpr nz = 58;
  unsigned constexpr ncrms = 200;
  unsigned constexpr iters = 10;
  yakl::init();

  std::cout << "Testing FFTs\n";
  {
    FArray<real,3,memDevice> data("data",ncrms,nx+2,nz);
    FFT<nx> fft;

    // Initialize data
    parallel_for( Bounds<3>({1,nz},{1,nx},{1,ncrms}) , YAKL_LAMBDA (int const indices[]) {
      int k, i, icrm;
      storeIndices( indices , k,i,icrm );
      data(icrm,i,k) = k*i*icrm * k*i*icrm;
    });

    auto t1 = std::clock();

    for (int iter=1; iter<=iters; iter++) {
      parallel_for( Bounds<2>({1,nz},{1,ncrms}) , YAKL_LAMBDA (int const indices[]) {
        int k, icrm;
        storeIndices( indices , k,icrm );

        FSArray<real,SBnd<1,nx+2>> data1d;
        FSArray<real,SBnd<1,nx  >> tmp;
        for (int i=1; i<=nx; i++) { data1d(i) = data(icrm,i,k); }
        fft.forward( data1d.data() , tmp.data() );
      });

      parallel_for( Bounds<2>({1,nz},{1,ncrms}) , YAKL_LAMBDA (int const indices[]) {
        int k, icrm;
        storeIndices( indices , k,icrm );

        FSArray<real,SBnd<1,nx+2>> data1d;
        FSArray<real,SBnd<1,nx  >> tmp;
        for (int i=1; i<=nx; i++) { data1d(i) = data(icrm,i,k); }
        fft.inverse( data1d.data() , tmp.data() );
      });
    }

    yakl::fence();

    auto tm = (std::clock() - t1) / (double) CLOCKS_PER_SEC;
    std::cout << data(1,1,1) << "\n";
    std::cout << "Time: " << tm << "\n";
  }

  std::cout << "Testing explicitly coded DFTs for a comparison point\n";
  {
    FArray<real,3,memDevice> data("data",ncrms,nx  ,nz);
    FArray<real,3,memDevice> fft ("fft ",ncrms,nx+2,nz);
    FArray<real,1,memDevice> trig_table("trig_table",2*nx);

    parallel_for( Bounds<1>({1,nx}) , YAKL_LAMBDA ( int const indices[] ) {
      int i = indices[0];
      trig_table(   i) = cos(2*M_PI*(i-1)/nx);
      trig_table(nx+i) = sin(2*M_PI*(i-1)/nx);
    });

    // Initialize data
    parallel_for( Bounds<3>({1,nz},{1,nx},{1,ncrms}) , YAKL_LAMBDA (int const indices[]) {
      int k, i, icrm;
      storeIndices( indices , k,i,icrm );
      data(icrm,i,k) = k*i*icrm * k*i*icrm;
    });

    auto t1 = std::clock();

    for (int iter=1; iter<=iters; iter++) {
      parallel_for( Bounds<3>({1,nz},{0,nx+1},{1,ncrms}) , YAKL_LAMBDA (int const indices[]) {
        int k, l, icrm;
        storeIndices( indices , k,l,icrm );

        real tmp = 0;
        // The i-loop is sequential
        for (unsigned i=0; i<nx; i++) {
          unsigned ind = wrap(i*(l/2),nx);
          real trig;
          if (l - ( (l >> 1) << 1 ) == 0) { // This is a cheap l%2 operator
            trig =  trig_table(   ind+1);
          } else {
            trig = -trig_table(nx+ind+1);
          }
          tmp += trig*data(icrm,i+1,k);
        }
        fft(icrm,l+1,k) = tmp/nx;
      });

      parallel_for( Bounds<3>({1,nz},{0,nx-1},{1,ncrms}) , YAKL_LAMBDA (int const indices[]) {
        int k, l, icrm;
        storeIndices( indices , k,l,icrm );

        real tmp = 0;
        // The i-loop is sequential
        for (unsigned i=0; i<nx; i++) {
          unsigned ind_tab = wrap(i*l,nx);
          unsigned ind_fft = 2*i;
          int      sgn     = -1;
          if (i > nx/2) {
            ind_fft = 2*(nx-i);
            sgn     = 1;
          }
          tmp +=     fft(icrm,ind_fft  +1,k)*trig_table(   ind_tab+1) +
                 sgn*fft(icrm,ind_fft+1+1,k)*trig_table(nx+ind_tab+1);
        }
        data(icrm,l+1,k) = tmp;
      });
    }

    yakl::fence();

    auto tm = (std::clock() - t1) / (double) CLOCKS_PER_SEC;
    std::cout << data(1,1,1) << "\n";
    std::cout << "Time: " << tm << "\n";
  }

  yakl::finalize();
}


