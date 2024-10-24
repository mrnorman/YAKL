
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::styleFortran;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;
using yakl::SArray;
using yakl::FSArray;
using yakl::SB;

typedef double real;

typedef Array<real,1,memDevice,styleC> real_c_1d;
typedef Array<real,2,memDevice,styleC> real_c_2d;
typedef Array<real,3,memDevice,styleC> real_c_3d;

typedef Array<real,1,memDevice,styleFortran> real_f_1d;
typedef Array<real,2,memDevice,styleFortran> real_f_2d;
typedef Array<real,3,memDevice,styleFortran> real_f_3d;

typedef Array<int,1,memDevice,styleC> int_c_1d;
typedef Array<int,2,memDevice,styleC> int_c_2d;
typedef Array<int,3,memDevice,styleC> int_c_3d;

typedef Array<int,1,memDevice,styleFortran> int_f_1d;
typedef Array<int,2,memDevice,styleFortran> int_f_2d;
typedef Array<int,3,memDevice,styleFortran> int_f_3d;

typedef Array<bool,1,memDevice,styleC> bool_c_1d;
typedef Array<bool,2,memDevice,styleC> bool_c_2d;
typedef Array<bool,3,memDevice,styleC> bool_c_3d;

typedef Array<bool,1,memDevice,styleFortran> bool_f_1d;
typedef Array<bool,2,memDevice,styleFortran> bool_f_2d;
typedef Array<bool,3,memDevice,styleFortran> bool_f_3d;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    int constexpr n1 = 5;
    int constexpr n2 = 10;
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // size, shape, lbound, ubound, epsilon, sign, mod, merge, abs, minval, maxval, minloc, maxloc
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using yakl::intrinsics::size;
      real_c_2d arr_c("arr_c",n1,n2);
      real_f_2d arr_f("arr_c",{-1,n1-2},n2);
      SArray<real,2,n1,n2> sarr_c;
      FSArray<real,2,SB<-1,n1-2>,SB<n2>> sarr_f;
      real scalar = 1;

      if (size(arr_c ) != n1*n2) die("arr_c wrong size tot");
      if (size(arr_f ) != n1*n2) die("arr_f wrong size tot");
      if (size(sarr_c) != n1*n2) die("sarr_c wrong size tot");
      if (size(sarr_f) != n1*n2) die("sarr_f wrong size tot");
      if (size(arr_c ,0) != n1) die("arr_c wrong size 1");
      if (size(arr_f ,1) != n1) die("arr_f wrong size 1");
      if (size(sarr_c,0) != n1) die("sarr_c wrong size 1");
      if (size(sarr_f,1) != n1) die("sarr_f wrong size 1");
      if (size(arr_c ,1) != n2) die("arr_c wrong size 2");
      if (size(arr_f ,2) != n2) die("arr_f wrong size 2");
      if (size(sarr_c,1) != n2) die("sarr_c wrong size 2");
      if (size(sarr_f,2) != n2) die("sarr_f wrong size 2");

      using yakl::intrinsics::shape;
      auto shp_arr_c = shape(arr_c);
      auto shp_arr_f = shape(arr_f);
      auto shp_sarr_c = shape(sarr_c);
      auto shp_sarr_f = shape(sarr_f);
      if (shp_arr_c (0) != n1) die("arr_c wrong shape 1");
      if (shp_arr_f (1) != n1) die("arr_f wrong shape 1");
      if (shp_sarr_c(0) != n1) die("sarr_c wrong shape 1");
      if (shp_sarr_f(1) != n1) die("sarr_f wrong shape 1");
      if (shp_arr_c (1) != n2) die("arr_c wrong shape 2");
      if (shp_arr_f (2) != n2) die("arr_f wrong shape 2");
      if (shp_sarr_c(1) != n2) die("sarr_c wrong shape 2");
      if (shp_sarr_f(2) != n2) die("sarr_f wrong shape 2");

      using yakl::intrinsics::lbound;
      auto lb_arr_c = lbound(arr_c);
      auto lb_arr_f = lbound(arr_f);
      auto lb_sarr_c = lbound(sarr_c);
      auto lb_sarr_f = lbound(sarr_f);
      if (lb_arr_c (0) != 0) die("arr_c wrong lbound 1");
      if (lb_arr_f (1) != -1) die("arr_f wrong lbound 1");
      if (lb_sarr_c(0) != 0) die("sarr_c wrong lbound 1");
      if (lb_sarr_f(1) != -1) die("sarr_f wrong lbound 1");
      if (lb_arr_c (1) != 0) die("arr_c wrong lbound 2");
      if (lb_arr_f (2) != 1) die("arr_f wrong lbound 2");
      if (lb_sarr_c(1) != 0) die("sarr_c wrong lbound 2");
      if (lb_sarr_f(2) != 1) die("sarr_f wrong lbound 2");
      if (lb_arr_c (0) != 0) die("arr_c wrong lbound 1");
      if (lbound(arr_f ,1) != -1) die("arr_f wrong lbound 1");
      if (lbound(sarr_c,0) != 0) die("sarr_c wrong lbound 1");
      if (lbound(sarr_f,1) != -1) die("sarr_f wrong lbound 1");
      if (lbound(arr_c ,1) != 0) die("arr_c wrong lbound 2");
      if (lbound(arr_f ,2) != 1) die("arr_f wrong lbound 2");
      if (lbound(sarr_c,1) != 0) die("sarr_c wrong lbound 2");
      if (lbound(sarr_f,2) != 1) die("sarr_f wrong lbound 2");

      using yakl::intrinsics::ubound;
      auto ub_arr_c = ubound(arr_c);
      auto ub_arr_f = ubound(arr_f);
      auto ub_sarr_c = ubound(sarr_c);
      auto ub_sarr_f = ubound(sarr_f);
      if (ub_arr_c (0) != n1-1) die("arr_c wrong ubound 1");
      if (ub_arr_f (1) != n1-2) die("arr_f wrong ubound 1");
      if (ub_sarr_c(0) != n1-1) die("sarr_c wrong ubound 1");
      if (ub_sarr_f(1) != n1-2) die("sarr_f wrong ubound 1");
      if (ub_arr_c (1) != n2-1) die("arr_c wrong ubound 2");
      if (ub_arr_f (2) != n2  ) die("arr_f wrong ubound 2");
      if (ub_sarr_c(1) != n2-1) die("sarr_c wrong ubound 2");
      if (ub_sarr_f(2) != n2  ) die("sarr_f wrong ubound 2");
      if (ubound(arr_c ,0) != n1-1) die("arr_c wrong ubound 1");
      if (ubound(arr_f ,1) != n1-2) die("arr_f wrong ubound 1");
      if (ubound(sarr_c,0) != n1-1) die("sarr_c wrong ubound 1");
      if (ubound(sarr_f,1) != n1-2) die("sarr_f wrong ubound 1");
      if (ubound(arr_c ,1) != n2-1) die("arr_c wrong ubound 2");
      if (ubound(arr_f ,2) != n2  ) die("arr_f wrong ubound 2");
      if (ubound(sarr_c,1) != n2-1) die("sarr_c wrong ubound 2");
      if (ubound(sarr_f,2) != n2  ) die("sarr_f wrong ubound 2");

      using yakl::intrinsics::epsilon;
      if (epsilon(arr_c) != std::numeric_limits<real>::epsilon()) die("arr_c wrong epsilon");
      if (epsilon(arr_f) != std::numeric_limits<real>::epsilon()) die("arr_f wrong epsilon");
      if (epsilon(sarr_c) != std::numeric_limits<real>::epsilon()) die("sarr_c wrong epsilon");
      if (epsilon(sarr_f) != std::numeric_limits<real>::epsilon()) die("sarr_f wrong epsilon");
      if (epsilon(scalar) != std::numeric_limits<real>::epsilon()) die("scalar wrong epsilon");

      using yakl::intrinsics::tiny;
      if (tiny(arr_c) != std::numeric_limits<real>::min()) die("arr_c wrong tiny");
      if (tiny(arr_f) != std::numeric_limits<real>::min()) die("arr_f wrong tiny");
      if (tiny(sarr_c) != std::numeric_limits<real>::min()) die("sarr_c wrong tiny");
      if (tiny(sarr_f) != std::numeric_limits<real>::min()) die("sarr_f wrong tiny");
      if (tiny(scalar) != std::numeric_limits<real>::min()) die("scalar wrong tiny");

      using yakl::intrinsics::huge;
      if (huge(arr_c) != std::numeric_limits<real>::max()) die("arr_c wrong huge");
      if (huge(arr_f) != std::numeric_limits<real>::max()) die("arr_f wrong huge");
      if (huge(sarr_c) != std::numeric_limits<real>::max()) die("sarr_c wrong huge");
      if (huge(sarr_f) != std::numeric_limits<real>::max()) die("sarr_f wrong huge");
      if (huge(scalar) != std::numeric_limits<real>::max()) die("scalar wrong huge");

      if (yakl::intrinsics::sign(13.1 , -0.1 ) != -13.1) die("ERROR: sign does not work");

      if (yakl::intrinsics::mod(12,5) != 2) die("ERROR: mod doesn't work");

      using yakl::intrinsics::sum;
      using yakl::intrinsics::abs;
      arr_c = -2;
      arr_f = -3;
      sarr_c = -4;
      sarr_f = -5;
      if (sum(abs(arr_c ))/size(arr_c ) != 2) die("ERROR: Wrong value for arr_c ");
      if (sum(abs(arr_f ))/size(arr_f ) != 3) die("ERROR: Wrong value for arr_f ");
      if (sum(abs(sarr_c))/size(sarr_c) != 4) die("ERROR: Wrong value for sarr_c");
      if (sum(abs(sarr_f))/size(sarr_f) != 5) die("ERROR: Wrong value for sarr_f");

      yakl::c::parallel_for( size(arr_c) , KOKKOS_LAMBDA (int i) { arr_c.data()[i] = i; });
      yakl::c::parallel_for( size(arr_f) , KOKKOS_LAMBDA (int i) { arr_f.data()[i] = i; });
      for (int i=0; i < size(sarr_c); i++) { sarr_c.data()[i] = i; }
      for (int i=0; i < size(sarr_f); i++) { sarr_f.data()[i] = i; }
    }
    ///////////////////////////////////////////
    // merge, minval, maxval
    ///////////////////////////////////////////
    {
      using yakl::intrinsics::sum;
      using yakl::intrinsics::merge;
      int constexpr n = 1024;
      Array<double,1,memHost,styleC> h_a1  ("h_a1"  ,n);
      Array<double,1,memHost,styleC> h_a2  ("h_a2"  ,n);
      Array<bool ,1,memHost,styleC> h_mask("h_mask",n);
      SArray<double,1,n> sarr_a1  ;
      SArray<double,1,n> sarr_a2  ;
      SArray<bool ,1,n> sarr_mask;
      FSArray<double,1,SB<n>> fsarr_a1  ;
      FSArray<double,1,SB<n>> fsarr_a2  ;
      FSArray<bool ,1,SB<n>> fsarr_mask;
      for (int i=0; i < n; i++) {
        h_a1.data()[i] = 2;
        h_a2.data()[i] = 3;
        h_mask.data()[i] = i%2 == 1;
        sarr_a1.data()[i] = 2;
        sarr_a2.data()[i] = 3;
        sarr_mask.data()[i] = i%2 == 1;
        fsarr_a1.data()[i] = 2;
        fsarr_a2.data()[i] = 3;
        fsarr_mask.data()[i] = i%2 == 1;
      }
      auto d_a1   = h_a1  .createDeviceCopy();
      auto d_a2   = h_a2  .createDeviceCopy();
      auto d_mask = h_mask.createDeviceCopy();

      if (std::abs(sum(merge(h_a1    ,h_a2    ,h_mask    ))/n-2.5)>=1.e-10) die("ERROR: Wrong value for merge(h_)");
      if (std::abs(sum(merge(d_a1    ,d_a2    ,d_mask    ))/n-2.5)>=1.e-10) die("ERROR: Wrong value for merge(h_)");
      if (std::abs(sum(merge(sarr_a1 ,sarr_a2 ,sarr_mask ))/n-2.5)>=1.e-10) die("ERROR: Wrong value for merge(sarr_)");
      if (std::abs(sum(merge(fsarr_a1,fsarr_a2,fsarr_mask))/n-2.5)>=1.e-10) die("ERROR: Wrong value for merge(fsarr_)");

      using yakl::intrinsics::minval;
      using yakl::intrinsics::maxval;
      Array<double,1,memDevice,styleFortran> d_a1_f("d_a1_f",n);
      Array<double,1,memHost  ,styleFortran> h_a1_f("h_a1_f",n);
      for (int i=0; i < n; i++) {
        h_a1    .data()[i] = n-i;
        sarr_a1 .data()[i] = n-i;
        fsarr_a1.data()[i] = n-i;
        h_a1_f  .data()[i] = n-i;
      }
      h_a1  .deep_copy_to(d_a1  );
      h_a1_f.deep_copy_to(d_a1_f);
      if ( minval(h_a1    ) != 1 ) die("ERROR: wrong minval h_a1    ");
      if ( minval(d_a1    ) != 1 ) die("ERROR: wrong minval d_a1    ");
      if ( minval(sarr_a1 ) != 1 ) die("ERROR: wrong minval sarr_a1 ");
      if ( minval(fsarr_a1) != 1 ) die("ERROR: wrong minval fsarr_a1");
      if ( maxval(h_a1    ) != n ) die("ERROR: wrong maxval h_a1    ");
      if ( maxval(d_a1    ) != n ) die("ERROR: wrong maxval d_a1    ");
      if ( maxval(sarr_a1 ) != n ) die("ERROR: wrong maxval sarr_a1 ");
      if ( maxval(fsarr_a1) != n ) die("ERROR: wrong maxval fsarr_a1");

      using yakl::intrinsics::minloc;
      using yakl::intrinsics::maxloc;
      if ( minloc(h_a1    ) != n-1 ) die("ERROR: wrong minloc h_a1    ");
      if ( minloc(d_a1    ) != n-1 ) die("ERROR: wrong minloc d_a1    ");
      if ( minloc(sarr_a1 ) != n-1 ) die("ERROR: wrong minloc sarr_a1 ");
      if ( minloc(fsarr_a1) != n   ) die("ERROR: wrong minloc fsarr_a1");
      if ( minloc(h_a1_f  ) != n   ) die("ERROR: wrong minloc h_a1_f  ");
      if ( minloc(d_a1_f  ) != n   ) die("ERROR: wrong minloc d_a1_f  ");

      if ( maxloc(h_a1    ) != 0 ) die("ERROR: maxloc(h_a1    ) != 0");
      if ( maxloc(d_a1    ) != 0 ) die("ERROR: maxloc(d_a1    ) != 0");
      if ( maxloc(sarr_a1 ) != 0 ) die("ERROR: maxloc(sarr_a1 ) != 0");
      if ( maxloc(fsarr_a1) != 1 ) die("ERROR: maxloc(fsarr_a1) != 1");
      if ( maxloc(h_a1_f  ) != 1 ) die("ERROR: maxloc(h_a1_f  ) != 1");
      if ( maxloc(d_a1_f  ) != 1 ) die("ERROR: maxloc(d_a1_f  ) != 1");
    }

    ///////////////////////////////////////
    // allocated, associated
    ///////////////////////////////////////
    {
      using yakl::intrinsics::allocated;
      using yakl::intrinsics::associated;
      real_c_2d arr_c("arr_c",n1,n2);
      real_f_2d arr_f("arr_c",{-1,n1-2},n2);
      SArray<real,2,n1,n2> sarr_c;
      FSArray<real,2,SB<-1,n1-2>,SB<n2>> sarr_f;
      real_c_2d arr_c_no;
      real_f_2d arr_f_no;
      if (!allocated(arr_c )) die("arr_c error allocated");
      if (!allocated(arr_f )) die("arr_f error allocated");
      if (!allocated(sarr_c)) die("sarr_c error allocated");
      if (!allocated(sarr_f)) die("sarr_f error allocated");
      if (!associated(arr_c )) die("arr_c error associated");
      if (!associated(arr_f )) die("arr_f error associated");
      if (!associated(sarr_c)) die("sarr_c error associated");
      if (!associated(sarr_f)) die("sarr_f error associated");
      if (allocated(arr_c_no)) die("arr_c_no error allocated");
      if (allocated(arr_f_no)) die("arr_f_no error allocated");
      if (associated(arr_c_no)) die("arr_c_no error associated");
      if (associated(arr_f_no)) die("arr_f_no error associated");
    }


    ////////////////////////////////////////////////
    // minloc, maxloc, minval, maxval, sum, product
    ////////////////////////////////////////////////
    {
      using yakl::intrinsics::minloc;
      using yakl::intrinsics::minval;
      using yakl::intrinsics::maxloc;
      using yakl::intrinsics::maxval;
      using yakl::intrinsics::sum;
      using yakl::intrinsics::product;
      SArray<real,1,n1> sarr_c;
      FSArray<real,1,SB<n1>> sarr_f;
      sarr_c(0) = -1;
      sarr_c(1) = -2;
      sarr_c(2) = 4;
      sarr_c(3) = 5;
      sarr_c(4) = 1;
      sarr_f(1) = -1;
      sarr_f(2) = -2;
      sarr_f(3) = 4;
      sarr_f(4) = 5;
      sarr_f(5) = 1;
      if (minloc(sarr_c) != 1) die("sarr_c error minloc");
      if (maxloc(sarr_c) != 3) die("sarr_c error maxloc");
      if (minloc(sarr_f) != 2) die("sarr_f error minloc");
      if (maxloc(sarr_f) != 4) die("sarr_f error maxloc");
      if (minval(sarr_c) != -2) die("sarr_c error minval");
      if (maxval(sarr_c) !=  5) die("sarr_c error maxval");
      if (minval(sarr_f) != -2) die("sarr_f error minval");
      if (maxval(sarr_f) !=  5) die("sarr_f error maxval");
      if (sum(sarr_c) != 7) die("sarr_c error sum");
      if (sum(sarr_f) != 7) die("sarr_f error sum");
      // minval, maxval, and sum tested for dynamic arrays elsewhere already
      if (product(sarr_c) != 40) die("sarr_c error product");
      if (product(sarr_f) != 40) die("sarr_f error product");
    }

    ////////////////////////////////////////////////
    // product
    ////////////////////////////////////////////////
    {
      int constexpr n = 1024;
      using yakl::intrinsics::minloc;
      using yakl::intrinsics::minval;
      using yakl::intrinsics::maxloc;
      using yakl::intrinsics::maxval;
      using yakl::intrinsics::sum;
      using yakl::intrinsics::product;
      Array  <double,1,memDevice,styleC> d_arr("d_arr",n);
      Array  <double,1,memHost  ,styleC> h_arr("h_arr",n);
      SArray <double,1,n>                cs_arr;
      FSArray<double,1,SB<n>>            fs_arr;
      for (int i=0; i < n; i++) {
        h_arr .data()[i] = 1 + i / 100000.;
        cs_arr.data()[i] = 1 + i / 100000.;
        fs_arr.data()[i] = 1 + i / 100000.;
      }
      h_arr.deep_copy_to(d_arr);

      double answer = h_arr(0);
      for (int i=1; i < n; i++) { answer *= h_arr(i); }
        
      if (std::abs(answer - product(d_arr )) > 1.e-10) die("ERROR: wrong product(d_arr )");
      if (std::abs(answer - product(h_arr )) > 1.e-10) die("ERROR: wrong product(h_arr )");
      if (std::abs(answer - product(cs_arr)) > 1.e-10) die("ERROR: wrong product(cs_arr)");
      if (std::abs(answer - product(fs_arr)) > 1.e-10) die("ERROR: wrong product(fs_arr)");
    }



    ////////////////////////////////////////////////
    // any
    ////////////////////////////////////////////////
    {
      using yakl::intrinsics::any;
      using yakl::intrinsics::all;
      using yakl::componentwise::operator<;
      using yakl::componentwise::operator>=;

      real_c_1d arr_c("arr_c",5);
      real_f_1d arr_f("arr_f",5);
      SArray<real,1,5> sarr_c;
      FSArray<real,1,SB<5>> sarr_f;

      parallel_for( 5 , KOKKOS_LAMBDA (int i) {
        arr_c (i  ) = i-2;
        arr_f (i+1) = i-2;
      });

      for (int i=0; i < 5; i++) {
        sarr_c (i  ) = i-2;
        sarr_f (i+1) = i-2;
      }

      if ( any( arr_c  < -2 ) ) die("arr_c  any fail 1");
      if ( any( arr_f  < -2 ) ) die("arr_f  any fail 1");
      if ( any( sarr_c < -2 ) ) die("sarr_c any fail 1");
      if ( any( sarr_f < -2 ) ) die("sarr_f any fail 1");

      if ( ! all( arr_c  >= -2 ) ) die("! all( arr_c  >= -2 )");
      if ( ! all( arr_f  >= -2 ) ) die("! all( arr_f  >= -2 )");
      if ( ! all( sarr_c >= -2 ) ) die("! all( sarr_c >= -2 )");
      if ( ! all( sarr_f >= -2 ) ) die("! all( sarr_f >= -2 )");
    }


    //////////////////////////////////////////////////////////
    // matmul_cr, matmul_rc, matinv, transpose
    //////////////////////////////////////////////////////////
    {
      using yakl::intrinsics::matmul_rc;
      using yakl::intrinsics::matmul_cr;
      using yakl::intrinsics::matinv_ge;
      using yakl::intrinsics::transpose;
      SArray<real,2,3,3> A1_c;
      SArray<real,2,3,3> A2_c;
      SArray<real,1,3> b_c;
      FSArray<real,2,SB<3>,SB<3>> A1_f;
      FSArray<real,2,SB<3>,SB<3>> A2_f;
      FSArray<real,1,SB<3>> b_f;

      SArray<real,1,3> A1_b_ref;
      SArray<real,2,3,3> A1_A2_ref;
      
      A1_c(0,0) = 1;
      A1_c(0,1) = 2;
      A1_c(0,2) = 3;
      A1_c(1,0) = 1.5;
      A1_c(1,1) = 2.5;
      A1_c(1,2) = 3.5;
      A1_c(2,0) = 1.2;
      A1_c(2,1) = 2.2;
      A1_c(2,2) = 3.2;

      A2_c(0,0) = 1.9;
      A2_c(0,1) = 2.9;
      A2_c(0,2) = 3.9;
      A2_c(1,0) = 1.1;
      A2_c(1,1) = 2.1;
      A2_c(1,2) = 3.1;
      A2_c(2,0) = 1.4;
      A2_c(2,1) = 2.4;
      A2_c(2,2) = 3.4;

      b_c(0) = 0.3;
      b_c(1) = 4.2;
      b_c(2) = 1.9;

      for (int j=0; j < 3; j++) {
        for (int i=0; i < 3; i++) {
          A1_f(j+1,i+1) = A1_c(j,i);
          A2_f(j+1,i+1) = A2_c(j,i);
          if (j == 0) b_f(i+1) = b_c(i);
        }
      }

      A1_b_ref(0) = 14.4;
      A1_b_ref(1) = 17.6;
      A1_b_ref(2) = 15.68;

      A1_A2_ref(0,0) = 8.3;
      A1_A2_ref(0,1) = 14.3;
      A1_A2_ref(0,2) = 20.3;
      A1_A2_ref(1,0) = 10.5;
      A1_A2_ref(1,1) = 18.0;
      A1_A2_ref(1,2) = 25.5;
      A1_A2_ref(2,0) = 9.18;
      A1_A2_ref(2,1) = 15.78;
      A1_A2_ref(2,2) = 22.38;

      auto A1_b_c  = matmul_rc( A1_c , b_c  );
      auto A1_A2_c = matmul_rc( A1_c , A2_c );
      auto A1_b_f  = matmul_rc( A1_f , b_f  );
      auto A1_A2_f = matmul_rc( A1_f , A2_f );

      real adiff_A1_b_c  = 0;
      real adiff_A1_A2_c = 0;
      real adiff_A1_b_f  = 0;
      real adiff_A1_A2_f = 0;
      for (int i=0; i < 3; i++) {
        adiff_A1_b_c += abs( A1_b_c(i  ) - A1_b_ref(i) );
        adiff_A1_b_f += abs( A1_b_f(i+1) - A1_b_ref(i) );
      }
      for (int j=0; j < 3; j++) {
        for (int i=0; i < 3; i++) {
          adiff_A1_A2_c += abs( A1_A2_c(j  ,i  ) - A1_A2_ref(j  ,i) );
          adiff_A1_A2_f += abs( A1_A2_f(j+1,i+1) - A1_A2_ref(j  ,i) );
        }
      }

      if (adiff_A1_b_c  >= 1.e-13) die("ERROR: incorrect adiff_A1_b_c  rc");
      if (adiff_A1_A2_c >= 1.e-13) die("ERROR: incorrect adiff_A1_A2_c rc");
      if (adiff_A1_b_f  >= 1.e-13) die("ERROR: incorrect adiff_A1_b_f  rc");
      if (adiff_A1_A2_f >= 1.e-13) die("ERROR: incorrect adiff_A1_A2_f rc");

      auto trans_A1_c = transpose( A1_c );
      auto trans_A2_c = transpose( A2_c );
      auto trans_A1_f = transpose( A1_f );
      auto trans_A2_f = transpose( A2_f );

      A1_b_c  = matmul_cr( trans_A1_c , b_c  );
      A1_A2_c = matmul_cr( trans_A1_c , trans_A2_c );
      A1_b_f  = matmul_cr( trans_A1_f , b_f  );
      A1_A2_f = matmul_cr( trans_A1_f , trans_A2_f );

      A1_A2_c = transpose( A1_A2_c );
      A1_A2_f = transpose( A1_A2_f );

      adiff_A1_b_c  = 0;
      adiff_A1_A2_c = 0;
      adiff_A1_b_f  = 0;
      adiff_A1_A2_f = 0;
      for (int i=0; i < 3; i++) {
        adiff_A1_b_c += abs( A1_b_c(i  ) - A1_b_ref(i) );
        adiff_A1_b_f += abs( A1_b_f(i+1) - A1_b_ref(i) );
      }
      for (int j=0; j < 3; j++) {
        for (int i=0; i < 3; i++) {
          adiff_A1_A2_c += abs( A1_A2_c(j  ,i  ) - A1_A2_ref(j  ,i) );
          adiff_A1_A2_f += abs( A1_A2_f(j+1,i+1) - A1_A2_ref(j  ,i) );
        }
      }

      if (adiff_A1_b_c  >= 1.e-13) die("ERROR: incorrect adiff_A1_b_c  cr");
      if (adiff_A1_A2_c >= 1.e-13) die("ERROR: incorrect adiff_A1_A2_c cr");
      if (adiff_A1_b_f  >= 1.e-13) die("ERROR: incorrect adiff_A1_b_f  cr");
      if (adiff_A1_A2_f >= 1.e-13) die("ERROR: incorrect adiff_A1_A2_f cr");


      A1_c(0,0) = 1;
      A1_c(0,1) = 0;
      A1_c(0,2) = 0;
      A1_c(1,0) = 1;
      A1_c(1,1) = 0.5;
      A1_c(1,2) = 0.25;
      A1_c(2,0) = 1;
      A1_c(2,1) = 1;
      A1_c(2,2) = 1;

      A1_f(1,1) = 1;
      A1_f(1,2) = 0;
      A1_f(1,3) = 0;
      A1_f(2,1) = 1;
      A1_f(2,2) = 0.5;
      A1_f(2,3) = 0.25;
      A1_f(3,1) = 1;
      A1_f(3,2) = 1;
      A1_f(3,3) = 1;

      auto A1_inv_c = matinv_ge( A1_c );
      auto identity_c = matmul_rc( A1_inv_c , A1_c );
      auto A1_inv_f = matinv_ge( A1_f );
      auto identity_f = matmul_rc( A1_inv_f , A1_f );

      real adiff_inv_c = 0;
      real adiff_inv_f = 0;
      for (int j=0; j < 3; j++) {
        for (int i=0; i < 3; i++) {
          if (i == j) {
            adiff_inv_c += abs( identity_c(j  ,i  ) - 1 );
            adiff_inv_f += abs( identity_f(j+1,i+1) - 1 );
          } else {
            adiff_inv_c += abs( identity_c(j  ,i  )     );
            adiff_inv_f += abs( identity_f(j+1,i+1)     );
          }
        }
      }
      if (adiff_inv_c >= 1.e-13) die("ERROR: incorrect adiff_inv_c");
      if (adiff_inv_f >= 1.e-13) die("ERROR: incorrect adiff_inv_f");
    }



    //////////////////////////////////////////////////////////
    // transpose (dyn arrays)
    //////////////////////////////////////////////////////////
    {
      using yakl::intrinsics::transpose;
      using yakl::intrinsics::size;
      Array<int,2,memHost  ,styleC      > h_c("h_c",3,4);
      Array<int,2,memDevice,styleC      > d_c("d_c",3,4);
      Array<int,2,memHost  ,styleFortran> h_f("h_f",3,4);
      Array<int,2,memDevice,styleFortran> d_f("d_f",3,4);
      for (int i=0; i < h_c.totElems(); i++) {
        h_c.data()[i] = i+1;
        h_f.data()[i] = i+1;
      }
      h_c.deep_copy_to(d_c);
      h_f.deep_copy_to(d_f);

      auto h_f_t = transpose( h_f );
      auto h_c_t = transpose( h_c );
      auto d_f_t = transpose( d_f ).createHostCopy();
      auto d_c_t = transpose( d_c ).createHostCopy();
      if ( size(h_c_t,0) != 4 ) die("ERROR: transpose: size(h_c_t,0) != 4");
      if ( size(h_f_t,1) != 4 ) die("ERROR: transpose: size(h_f_t,1) != 4");
      if ( size(d_c_t,0) != 4 ) die("ERROR: transpose: size(d_c_t,0) != 4");
      if ( size(d_f_t,1) != 4 ) die("ERROR: transpose: size(d_f_t,1) != 4");
      if ( h_c_t(3,0) != h_c(0,3) ) die("ERROR: transpose: h_c_t(3,0) != h_c(0,3)");
      if ( d_c_t(3,0) != d_c.createHostCopy()(0,3) ) die("ERROR: transpose: d_c_t(3,0) != d_c(0,3)");
      if ( h_f_t(4,1) != h_f(1,4) ) die("ERROR: transpose: h_f_t(4,1) != h_f(1,4)");
      if ( d_f_t(4,1) != d_f.createHostCopy()(1,4) ) die("ERROR: transpose: d_f_t(4,1) != d_f(1,4)");
    }



    //////////////////////////////////////////////////////////
    // count, pack
    //////////////////////////////////////////////////////////
    {
      using yakl::intrinsics::count;
      using yakl::intrinsics::pack;
      bool_c_1d c("c",10);
      bool_f_1d f("f",10);
      SArray<bool,1,10> sc;
      FSArray<bool,1,SB<10>> sf;
      
      parallel_for( 10 , KOKKOS_LAMBDA( int i ) {
        c(i) = i%2 == 0;
        f(i+1) = i%2 == 0;
      });
      for (int i=0; i < 10; i++) {
        sc(i) = i%2 == 0;
        sf(i+1) = i%2 == 0;
      }

      if (count(c)  != 5) die("ERROR: incorrect count c");
      if (count(sc) != 5) die("ERROR: incorrect count sc");
      if (count(f)  != 5) die("ERROR: incorrect count f");
      if (count(sf) != 5) die("ERROR: incorrect count sf");

      {
        real_c_1d vals("vals",10);
        auto vals_host = vals.createHostCopy();
        for (int i=0; i < 10; i++) { vals_host(i) = i; }
        vals_host.deep_copy_to(vals);

        auto packed        = pack(vals_host);
        auto packed_masked = pack(vals_host,c.createHostCopy());

        if ( yakl::intrinsics::sum(packed       ) != 45 ) die("ERROR: host packed c");
        if ( yakl::intrinsics::sum(packed_masked) != 20 ) die("ERROR: host packed masked c");

        packed        = pack(vals).createHostCopy();
        packed_masked = pack(vals,c).createHostCopy();

        if ( yakl::intrinsics::sum(packed       ) != 45 ) die("ERROR: device packed c");
        if ( yakl::intrinsics::sum(packed_masked) != 20 ) die("ERROR: device packed masked c");
      }

      {
        real_f_1d vals("vals",10);
        auto vals_host = vals.createHostCopy();
        for (int i=0; i < 10; i++) { vals_host(i+1) = i; }
        vals_host.deep_copy_to(vals);

        auto packed        = pack(vals_host);
        auto packed_masked = pack(vals_host,f.createHostCopy());

        if ( yakl::intrinsics::sum(packed       ) != 45 ) die("ERROR: host packed f");
        if ( yakl::intrinsics::sum(packed_masked) != 20 ) die("ERROR: host packed masked f");

        packed        = pack(vals).createHostCopy();
        packed_masked = pack(vals,f).createHostCopy();

        if ( yakl::intrinsics::sum(packed       ) != 45 ) die("ERROR: device packed f");
        if ( yakl::intrinsics::sum(packed_masked) != 20 ) die("ERROR: device packed masked f");
      }
    }


    {
      using namespace yakl::componentwise;
      using yakl::intrinsics::sum;
      using yakl::intrinsics::count;
      int constexpr n = 1024;
      Array  <float ,1,memDevice,styleC> a("a",n);
      Array  <double,1,memDevice,styleC> b("b",n);
      Array  <float ,1,memHost  ,styleC> c("c",n);
      Array  <double,1,memHost  ,styleC> d("d",n);
      SArray <float ,1,n>                e;
      SArray <double,1,n>                f;
      FSArray<float ,1,SB<n>>            g;
      FSArray<double,1,SB<n>>            h;
      double                             i;
      float  av = 1;         a = av;
      double bv = 2;         b = bv;
      float  cv = 3;         c = cv;
      double dv = 4;         d = dv;
      float  ev = 5;         e = ev;
      double fv = 6;         f = fv;
      float  gv = 7;         g = gv;
      double hv = 8;         h = hv;
      double iv = 9;         i = iv;

      if ( std::abs( sum(a+b) - n*(av+bv) ) / (n*(av+bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a+b)");
      if ( std::abs( sum(a+i) - n*(av+iv) ) / (n*(av+iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a+i)");
      if ( std::abs( sum(i+b) - n*(iv+bv) ) / (n*(iv+bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i+b)");
      if ( std::abs( sum(c+d) - n*(cv+dv) ) / (n*(cv+dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c+d)");
      if ( std::abs( sum(c+i) - n*(cv+iv) ) / (n*(cv+iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c+i)");
      if ( std::abs( sum(i+d) - n*(iv+dv) ) / (n*(iv+dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i+d)");
      if ( std::abs( sum(e+f) - n*(ev+fv) ) / (n*(ev+fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e+f)");
      if ( std::abs( sum(e+i) - n*(ev+iv) ) / (n*(ev+iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e+i)");
      if ( std::abs( sum(i+f) - n*(iv+fv) ) / (n*(iv+fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i+f)");
      if ( std::abs( sum(g+h) - n*(gv+hv) ) / (n*(gv+hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g+h)");
      if ( std::abs( sum(g+i) - n*(gv+iv) ) / (n*(gv+iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g+i)");
      if ( std::abs( sum(i+h) - n*(iv+hv) ) / (n*(iv+hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i+h)");

      if ( std::abs( sum(a-b) - n*(av-bv) ) / (n*(av-bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a-b)");
      if ( std::abs( sum(a-i) - n*(av-iv) ) / (n*(av-iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a-i)");
      if ( std::abs( sum(i-b) - n*(iv-bv) ) / (n*(iv-bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i-b)");
      if ( std::abs( sum(c-d) - n*(cv-dv) ) / (n*(cv-dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c-d)");
      if ( std::abs( sum(c-i) - n*(cv-iv) ) / (n*(cv-iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c-i)");
      if ( std::abs( sum(i-d) - n*(iv-dv) ) / (n*(iv-dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i-d)");
      if ( std::abs( sum(e-f) - n*(ev-fv) ) / (n*(ev-fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e-f)");
      if ( std::abs( sum(e-i) - n*(ev-iv) ) / (n*(ev-iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e-i)");
      if ( std::abs( sum(i-f) - n*(iv-fv) ) / (n*(iv-fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i-f)");
      if ( std::abs( sum(g-h) - n*(gv-hv) ) / (n*(gv-hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g-h)");
      if ( std::abs( sum(g-i) - n*(gv-iv) ) / (n*(gv-iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g-i)");
      if ( std::abs( sum(i-h) - n*(iv-hv) ) / (n*(iv-hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i-h)");

      if ( std::abs( sum(a*b) - n*(av*bv) ) / (n*(av*bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a*b)");
      if ( std::abs( sum(a*i) - n*(av*iv) ) / (n*(av*iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a*i)");
      if ( std::abs( sum(i*b) - n*(iv*bv) ) / (n*(iv*bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i*b)");
      if ( std::abs( sum(c*d) - n*(cv*dv) ) / (n*(cv*dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c*d)");
      if ( std::abs( sum(c*i) - n*(cv*iv) ) / (n*(cv*iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c*i)");
      if ( std::abs( sum(i*d) - n*(iv*dv) ) / (n*(iv*dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i*d)");
      if ( std::abs( sum(e*f) - n*(ev*fv) ) / (n*(ev*fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e*f)");
      if ( std::abs( sum(e*i) - n*(ev*iv) ) / (n*(ev*iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e*i)");
      if ( std::abs( sum(i*f) - n*(iv*fv) ) / (n*(iv*fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i*f)");
      if ( std::abs( sum(g*h) - n*(gv*hv) ) / (n*(gv*hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g*h)");
      if ( std::abs( sum(g*i) - n*(gv*iv) ) / (n*(gv*iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g*i)");
      if ( std::abs( sum(i*h) - n*(iv*hv) ) / (n*(iv*hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i*h)");

      if ( std::abs( sum(a/b) - n*(av/bv) ) / (n*(av/bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a/b)");
      if ( std::abs( sum(a/i) - n*(av/iv) ) / (n*(av/iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(a/i)");
      if ( std::abs( sum(i/b) - n*(iv/bv) ) / (n*(iv/bv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i/b)");
      if ( std::abs( sum(c/d) - n*(cv/dv) ) / (n*(cv/dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c/d)");
      if ( std::abs( sum(c/i) - n*(cv/iv) ) / (n*(cv/iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(c/i)");
      if ( std::abs( sum(i/d) - n*(iv/dv) ) / (n*(iv/dv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i/d)");
      if ( std::abs( sum(e/f) - n*(ev/fv) ) / (n*(ev/fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e/f)");
      if ( std::abs( sum(e/i) - n*(ev/iv) ) / (n*(ev/iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(e/i)");
      if ( std::abs( sum(i/f) - n*(iv/fv) ) / (n*(iv/fv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i/f)");
      if ( std::abs( sum(g/h) - n*(gv/hv) ) / (n*(gv/hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g/h)");
      if ( std::abs( sum(g/i) - n*(gv/iv) ) / (n*(gv/iv)) > 1.e-7 ) die("ERROR: wrong sum: sum(g/i)");
      if ( std::abs( sum(i/h) - n*(iv/hv) ) / (n*(iv/hv)) > 1.e-7 ) die("ERROR: wrong sum: sum(i/h)");

      if ( count( a > 0 ) != n ) die("ERROR: wrong count: a > 0");
      if ( count( 1 < b ) != n ) die("ERROR: wrong count: 1 < b");
      if ( count( c > 2 ) != n ) die("ERROR: wrong count: c > 2");
      if ( count( 3 < d ) != n ) die("ERROR: wrong count: 3 < d");
      if ( count( e > 4 ) != n ) die("ERROR: wrong count: e > 4");
      if ( count( 5 < f ) != n ) die("ERROR: wrong count: 5 < f");
      if ( count( g > 6 ) != n ) die("ERROR: wrong count: g > 6");
      if ( count( 7 < h ) != n ) die("ERROR: wrong count: 7 < h");
      if ( count( b > a ) != n ) die("ERROR: wrong count: b > a");
      if ( count( d > c ) != n ) die("ERROR: wrong count: d > c");
      if ( count( f > e ) != n ) die("ERROR: wrong count: f > e");
      if ( count( h > g ) != n ) die("ERROR: wrong count: h > g");

      if ( count( a >= 1 ) != n ) die("ERROR: wrong count: a >= 1");
      if ( count( 2 <= b ) != n ) die("ERROR: wrong count: 2 <= b");
      if ( count( c >= 3 ) != n ) die("ERROR: wrong count: c >= 3");
      if ( count( 4 <= d ) != n ) die("ERROR: wrong count: 4 <= d");
      if ( count( e >= 5 ) != n ) die("ERROR: wrong count: e >= 5");
      if ( count( 6 <= f ) != n ) die("ERROR: wrong count: 6 <= f");
      if ( count( g >= 7 ) != n ) die("ERROR: wrong count: g >= 7");
      if ( count( 8 <= h ) != n ) die("ERROR: wrong count: 8 <= h");
      if ( count( b >= (a+1) ) != n ) die("ERROR: wrong count: b >= (a+1)");
      if ( count( d >= (c+1) ) != n ) die("ERROR: wrong count: d >= (c+1)");
      if ( count( f >= (e+1) ) != n ) die("ERROR: wrong count: f >= (e+1)");
      if ( count( h >= (g+1) ) != n ) die("ERROR: wrong count: h >= (g+1)");

      if ( count( a < 2 ) != n ) die("ERROR: wrong count: a < 2");
      if ( count( 3 > b ) != n ) die("ERROR: wrong count: 3 > b");
      if ( count( c < 4 ) != n ) die("ERROR: wrong count: c < 4");
      if ( count( 5 > d ) != n ) die("ERROR: wrong count: 5 > d");
      if ( count( e < 6 ) != n ) die("ERROR: wrong count: e < 6");
      if ( count( 7 > f ) != n ) die("ERROR: wrong count: 7 > f");
      if ( count( g < 8 ) != n ) die("ERROR: wrong count: g < 8");
      if ( count( 9 > h ) != n ) die("ERROR: wrong count: 9 > h");
      if ( count( a < b ) != n ) die("ERROR: wrong count: a < b");
      if ( count( c < d ) != n ) die("ERROR: wrong count: c < d");
      if ( count( e < f ) != n ) die("ERROR: wrong count: e < f");
      if ( count( g < h ) != n ) die("ERROR: wrong count: g < h");

      if ( count( a <= 1 ) != n ) die("ERROR: wrong count: a <= 1");
      if ( count( 2 >= b ) != n ) die("ERROR: wrong count: 2 >= b");
      if ( count( c <= 3 ) != n ) die("ERROR: wrong count: c <= 3");
      if ( count( 4 >= d ) != n ) die("ERROR: wrong count: 4 >= d");
      if ( count( e <= 5 ) != n ) die("ERROR: wrong count: e <= 5");
      if ( count( 6 >= f ) != n ) die("ERROR: wrong count: 6 >= f");
      if ( count( g <= 7 ) != n ) die("ERROR: wrong count: g <= 7");
      if ( count( 8 >= h ) != n ) die("ERROR: wrong count: 8 >= h");
      if ( count( (a+1) <= b ) != n ) die("ERROR: wrong count: (a+1) <= b");
      if ( count( (c+1) <= d ) != n ) die("ERROR: wrong count: (c+1) <= d");
      if ( count( (e+1) <= f ) != n ) die("ERROR: wrong count: (e+1) <= f");
      if ( count( (g+1) <= h ) != n ) die("ERROR: wrong count: (g+1) <= h");

      if ( count( a == 1 ) != n ) die("ERROR: wrong count: a == 1");
      if ( count( 2 == b ) != n ) die("ERROR: wrong count: 2 == b");
      if ( count( c == 3 ) != n ) die("ERROR: wrong count: c == 3");
      if ( count( 4 == d ) != n ) die("ERROR: wrong count: 4 == d");
      if ( count( e == 5 ) != n ) die("ERROR: wrong count: e == 5");
      if ( count( 6 == f ) != n ) die("ERROR: wrong count: 6 == f");
      if ( count( g == 7 ) != n ) die("ERROR: wrong count: g == 7");
      if ( count( 8 == h ) != n ) die("ERROR: wrong count: 8 == h");

      if ( count( ! (a == 1) ) != 0 ) die("ERROR: wrong count: a == 1");
      if ( count( ! (2 == b) ) != 0 ) die("ERROR: wrong count: 2 == b");
      if ( count( ! (c == 3) ) != 0 ) die("ERROR: wrong count: c == 3");
      if ( count( ! (4 == d) ) != 0 ) die("ERROR: wrong count: 4 == d");
      if ( count( ! (e == 5) ) != 0 ) die("ERROR: wrong count: e == 5");
      if ( count( ! (6 == f) ) != 0 ) die("ERROR: wrong count: 6 == f");
      if ( count( ! (g == 7) ) != 0 ) die("ERROR: wrong count: g == 7");
      if ( count( ! (8 == h) ) != 0 ) die("ERROR: wrong count: 8 == h");

      if ( count( a != 0 ) != n ) die("ERROR: wrong count: a != 0");
      if ( count( 0 != b ) != n ) die("ERROR: wrong count: 0 != b");
      if ( count( c != 0 ) != n ) die("ERROR: wrong count: c != 0");
      if ( count( 0 != d ) != n ) die("ERROR: wrong count: 0 != d");
      if ( count( e != 0 ) != n ) die("ERROR: wrong count: e != 0");
      if ( count( 0 != f ) != n ) die("ERROR: wrong count: 0 != f");
      if ( count( g != 0 ) != n ) die("ERROR: wrong count: g != 0");
      if ( count( 0 != h ) != n ) die("ERROR: wrong count: 0 != h");

      if ( count( true && ( ( (a > 0) && (b > 0) ) && true ) ) != n ) die("ERROR: wrong count: true && ( ( (a > 0) && (b > 0) ) && true )");
      if ( count( true && ( ( (c > 0) && (d > 0) ) && true ) ) != n ) die("ERROR: wrong count: true && ( ( (c > 0) && (d > 0) ) && true )");
      if ( count( true && ( ( (e > 0) && (f > 0) ) && true ) ) != n ) die("ERROR: wrong count: true && ( ( (e > 0) && (f > 0) ) && true )");
      if ( count( true && ( ( (g > 0) && (h > 0) ) && true ) ) != n ) die("ERROR: wrong count: true && ( ( (g > 0) && (h > 0) ) && true )");

      if ( count( false || ( ( (a > 0) || (b < 0) ) || false ) ) != n ) die("ERROR: wrong count: false || ( ( (a > 0) || (b < 0) ) || false )");
      if ( count( false || ( ( (c > 0) || (d < 0) ) || false ) ) != n ) die("ERROR: wrong count: false || ( ( (c > 0) || (d < 0) ) || false )");
      if ( count( false || ( ( (e > 0) || (f < 0) ) || false ) ) != n ) die("ERROR: wrong count: false || ( ( (e > 0) || (f < 0) ) || false )");
      if ( count( false || ( ( (g > 0) || (h < 0) ) || false ) ) != n ) die("ERROR: wrong count: false || ( ( (g > 0) || (h < 0) ) || false )");

      if ( std::abs( sum(a++) - n*(av+1) ) / (n*(av+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(a++)");
      if ( std::abs( sum(++b) - n*(bv+1) ) / (n*(bv+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(++b)");
      if ( std::abs( sum(c++) - n*(cv+1) ) / (n*(cv+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(c++)");
      if ( std::abs( sum(++d) - n*(dv+1) ) / (n*(dv+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(++d)");
      if ( std::abs( sum(e++) - n*(ev+1) ) / (n*(ev+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(e++)");
      if ( std::abs( sum(++f) - n*(fv+1) ) / (n*(fv+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(++f)");
      if ( std::abs( sum(g++) - n*(gv+1) ) / (n*(gv+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(g++)");
      if ( std::abs( sum(++h) - n*(hv+1) ) / (n*(hv+1)) > 1.e-7 ) die("ERROR: wrong sum: sum(++h)");
    }



  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}

