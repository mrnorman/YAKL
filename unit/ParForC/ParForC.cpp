
#include <iostream>
#include "YAKL.h"

/////////////////////////////////////////////////////////////////////
// Most of YAKL_parallel_for_c.h is tested in the CArray unit test.
// This is to cover what was not covered in CArray
/////////////////////////////////////////////////////////////////////

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_outer;
using yakl::c::parallel_inner;
using yakl::c::single_inner;
using yakl::LaunchConfig;
using yakl::InnerHandler;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;
using yakl::fence_inner;
using yakl::intrinsics::sum;

typedef double real;

typedef Array<real,1,memDevice,styleC> real1d;
typedef Array<real,2,memDevice,styleC> real2d;
typedef Array<real,3,memDevice,styleC> real3d;
typedef Array<real,4,memDevice,styleC> real4d;


void die(std::string msg) {
  yakl::yakl_throw(msg.c_str());
}


int main() {
  yakl::init();
  {
    int constexpr n1 = 128;
    int constexpr n2 = 16;
    int constexpr n3 = 4;
    real1d arr1d("arr1d",n1);
    real3d arr3d("arr3d",n1,n2,n3);
    // Test with labels and SimpleBounds
    parallel_for( "mylabel" , n1 , YAKL_LAMBDA (int i) {
      arr1d(i) = 1;
    });
    if ( abs(sum(arr1d) - n1) / (n1) > 1.e-13) die("ERROR: Wrong sum for arr1d");
    parallel_for( "mylabel" , SimpleBounds<3>(n1,n2,n3) , YAKL_LAMBDA (int k, int j, int i) {
      arr3d(k,j,i) = 1;
    });
    if ( abs(sum(arr3d) - (double) n1*n2*n3) / (double) (n1*n2*n3) > 1.e-13) die("ERROR: Wrong sum for arr3d");

    yakl::memset(arr3d,0.);

    parallel_outer( YAKL_AUTO_LABEL() , Bounds<1>(n1) , YAKL_LAMBDA (int k, InnerHandler handler ) {
      parallel_inner( Bounds<2>(n2,n3) , [&] (int j, int i) {
        arr3d(k,j,i) = 2.;
      } , handler );
      fence_inner( handler );
      parallel_inner( Bounds<2>(n2,n3) , [&] (int j, int i) {
        arr3d(k,j,i) = 3.;
      } , handler );
      fence_inner( handler );
      single_inner( [&] () {
        arr3d(k,0,0) = 0;
      } , handler );
    } , LaunchConfig<n2*n3>() );

    real exact = (double) n1*n2*n3*3 - (double) n1*3;
    if ( abs(sum(arr3d) - exact) / exact > 1.e-13) die("ERROR: Wrong sum for arr3d");

    #ifdef YAKL_ARCH_OPENMP
    {
      int constexpr nz = 8;
      int constexpr nx = 8;
      yakl::ScalarLiveOut<int> tot_outer(0);
      yakl::ScalarLiveOut<int> tot      (0);
      parallel_outer( nz , YAKL_LAMBDA (int k, InnerHandler handler) {
        yakl::atomicAdd( tot_outer() , omp_get_thread_num() );
        parallel_inner( nx , [&] (int i) {
          yakl::atomicAdd( tot() , omp_get_thread_num() );
        } , handler );
      } , LaunchConfig<nz>() );
      if (tot_outer.hostRead() != 28 ) yakl::yakl_throw("ERROR: Wrong tot_outer");
      if (tot      .hostRead() != 224) yakl::yakl_throw("ERROR: Wrong tot");
    }
    #endif

    int nz = 128;
    int ny = 128;
    int nx = 128;
    real3d data("data",nz,ny,nx);
    real4d limits_x("limits_x",2,nz,ny,nx+1);
    limits_x = 0;
    data = 0;
    yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      if (i > nx/4 && i < 3*nx/4 && j > ny/4 && j < 3*ny/4 && k > nz/4 && k < 3*nz/4) data(k,j,i) = 1;
    });
    int constexpr ord = 9;
    int constexpr hs = (ord-1)/2;
    yakl::SArray<real,2,ord,2> s2g = 1;
    s2g(0,0)=-0.0019841269841269841269841269841269841270;
    s2g(0,1)=0.0015873015873015873015873015873015873016;
    s2g(1,0)=0.021825396825396825396825396825396825397;
    s2g(1,1)=-0.016269841269841269841269841269841269841;
    s2g(2,0)=-0.12103174603174603174603174603174603175;
    s2g(2,1)=0.078968253968253968253968253968253968254;
    s2g(3,0)=0.54563492063492063492063492063492063492;
    s2g(3,1)=-0.25436507936507936507936507936507936508;
    s2g(4,0)=0.74563492063492063492063492063492063492;
    s2g(4,1)=0.74563492063492063492063492063492063492;
    s2g(5,0)=-0.25436507936507936507936507936507936508;
    s2g(5,1)=0.54563492063492063492063492063492063492;
    s2g(6,0)=0.078968253968253968253968253968253968254;
    s2g(6,1)=-0.12103174603174603174603174603174603175;
    s2g(7,0)=-0.016269841269841269841269841269841269841;
    s2g(7,1)=0.021825396825396825396825396825396825397;
    s2g(8,0)=0.0015873015873015873015873015873015873016;
    s2g(8,1)=-0.0019841269841269841269841269841269841270;

    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      yakl::SArray<real,1,ord> stencil;
      // Load stencil with periodic boundary conditions
      for (int ii=-hs; ii <= hs; ii++) {
        int ind = i+ii;  if (ind < 0) ind += nx;  if (ind > nx-1) ind -= nx;
        stencil(ii+hs) = data(k,j,ind);
      }
      // Reconstruct values at cell edges
      yakl::SArray<real,1,2> gll;
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) { tmp += s2g(s,ii) * stencil(s); }
        gll(ii) = tmp;
      }
      // Store values at cell edges
      limits_x(1,k,j,i  ) = gll(0);
      limits_x(0,k,j,i+1) = gll(1);
    });
    real mn = yakl::intrinsics::minval(limits_x);
    real mx = yakl::intrinsics::maxval(limits_x);



    limits_x = 0;
    data = 0;
    yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      if (i > nx/4 && i < 3*nx/4 && j > ny/4 && j < 3*ny/4 && k > nz/4 && k < 3*nz/4) data(k,j,i) = 1;
    });
    parallel_outer( YAKL_AUTO_LABEL() , Bounds<2>(nz,ny) , YAKL_LAMBDA (int k, int j , InnerHandler const &handler) {
      parallel_inner( nx , [&] (int i) {
        yakl::SArray<real,1,ord> stencil;
        // Load stencil with periodic boundary conditions
        for (int ii=-hs; ii <= hs; ii++) {
          int ind = i+ii;  if (ind < 0) ind += nx;  if (ind > nx-1) ind -= nx;
          stencil(ii+hs) = data(k,j,ind);
        }
        // Reconstruct values at cell edges
        yakl::SArray<real,1,2> gll;
        for (int ii=0; ii<2; ii++) {
          real tmp = 0;
          for (int s=0; s < ord; s++) { tmp += s2g(s,ii) * stencil(s); }
          gll(ii) = tmp;
        }
        // Store values at cell edges
        limits_x(1,k,j,i  ) = gll(0);
        limits_x(0,k,j,i+1) = gll(1);
      } , handler );
    } , yakl::LaunchConfig<128>() );
    if (std::abs(yakl::intrinsics::minval(limits_x) - mn) > 1.e-13) die("Incorrect mn");
    if (std::abs(yakl::intrinsics::maxval(limits_x) - mx) > 1.e-13) die("Incorrect mx");



    limits_x = 0;
    data = 0;
    yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      if (i > nx/4 && i < 3*nx/4 && j > ny/4 && j < 3*ny/4 && k > nz/4 && k < 3*nz/4) data(k,j,i) = 1;
    });
    parallel_outer( YAKL_AUTO_LABEL() , Bounds<2>(nz,ny) , YAKL_LAMBDA (int k, int j , InnerHandler &handler) {
      real2d s2g_slm(nullptr,handler.get_inner_cache_pointer<real>(),9,2);
      parallel_inner( s2g.size() , [&] (int ii) {
        s2g_slm.data()[ii] = s2g.data()[ii];
      } , handler );
      fence_inner( handler );
      parallel_inner( nx , [&] (int i) {
        yakl::SArray<real,1,ord> stencil;
        // Load stencil with periodic boundary conditions
        for (int ii=-hs; ii <= hs; ii++) {
          int ind = i+ii;  if (ind < 0) ind += nx;  if (ind > nx-1) ind -= nx;
          stencil(ii+hs) = data(k,j,ind);
        }
        // Reconstruct values at cell edges
        yakl::SArray<real,1,2> gll;
        for (int ii=0; ii<2; ii++) {
          real tmp = 0;
          for (int s=0; s < ord; s++) { tmp += s2g_slm(s,ii) * stencil(s); }
          gll(ii) = tmp;
        }
        // Store values at cell edges
        limits_x(1,k,j,i  ) = gll(0);
        limits_x(0,k,j,i+1) = gll(1);
      } , handler );
    } , yakl::LaunchConfig<128>().set_inner_cache_bytes(s2g.size()*sizeof(real)) );
    if (std::abs(yakl::intrinsics::minval(limits_x) - mn) > 1.e-13) die("Incorrect mn");
    if (std::abs(yakl::intrinsics::maxval(limits_x) - mx) > 1.e-13) die("Incorrect mx");
  }
  yakl::finalize();
  
  return 0;
}

