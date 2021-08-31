
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
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n1 = 5;
    int constexpr n2 = 10;
    ////////////////////////////////////////
    // size, shape, lbound, ubound, epsilon
    ////////////////////////////////////////
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
    // anyLT, anyLTE, anyGT, anyGTE, anyEQ, anyNEQ
    ////////////////////////////////////////////////
    {
      using yakl::intrinsics::anyLT;
      using yakl::intrinsics::anyLTE;
      using yakl::intrinsics::anyGT;
      using yakl::intrinsics::anyGTE;
      using yakl::intrinsics::anyEQ;
      using yakl::intrinsics::anyNEQ;

      real_c_1d arr_c("arr_c",5);
      real_f_1d arr_f("arr_f",5);
      SArray<real,1,5> sarr_c;
      FSArray<real,1,SB<5>> sarr_f;

      bool_c_1d mask_c("mask_c",5);
      bool_f_1d mask_f("mask_f",5);
      SArray<bool,1,5> smask_c;
      FSArray<bool,1,SB<5>> smask_f;

      parallel_for( 5 , YAKL_LAMBDA (int i) {
        arr_c (i  ) = i-2;
        arr_f (i+1) = i-2;
        mask_c(i  ) = (i+1)%2 == 0;
        mask_f(i+1) = (i+1)%2 == 0;
      });

      for (int i=0; i < 5; i++) {
        sarr_c (i  ) = i-2;
        sarr_f (i+1) = i-2;
        smask_c(i  ) = (i+1)%2 == 0;
        smask_f(i+1) = (i+1)%2 == 0;
      }

      if ( anyLT( arr_c  , -2 ) ) die("arr_c  anyLT fail 1");
      if ( anyLT( arr_f  , -2 ) ) die("arr_f  anyLT fail 1");
      if ( anyLT( sarr_c , -2 ) ) die("sarr_c anyLT fail 1");
      if ( anyLT( sarr_f , -2 ) ) die("sarr_f anyLT fail 1");

      if ( !anyLT( arr_c  , -1 ) ) die("arr_c  anyLT fail 2");
      if ( !anyLT( arr_f  , -1 ) ) die("arr_f  anyLT fail 2");
      if ( !anyLT( sarr_c , -1 ) ) die("sarr_c anyLT fail 2");
      if ( !anyLT( sarr_f , -1 ) ) die("sarr_f anyLT fail 2");

      if ( anyLT( arr_c  , mask_c  , -1 ) ) die("arr_c anyLT masked fail");
      if ( anyLT( arr_f  , mask_f  , -1 ) ) die("arr_f anyLT masked fail");
      if ( anyLT( sarr_c , smask_c , -1 ) ) die("sarr_c anyLT masked fail");
      if ( anyLT( sarr_f , smask_f , -1 ) ) die("sarr_f anyLT masked fail");

      if ( anyLTE( arr_c  , -3 ) ) die("arr_c  anyLTE fail 1");
      if ( anyLTE( arr_f  , -3 ) ) die("arr_f  anyLTE fail 1");
      if ( anyLTE( sarr_c , -3 ) ) die("sarr_c anyLTE fail 1");
      if ( anyLTE( sarr_f , -3 ) ) die("sarr_f anyLTE fail 1");

      if ( !anyLTE( arr_c  , -2 ) ) die("arr_c  anyLTE fail 2");
      if ( !anyLTE( arr_f  , -2 ) ) die("arr_f  anyLTE fail 2");
      if ( !anyLTE( sarr_c , -2 ) ) die("sarr_c anyLTE fail 2");
      if ( !anyLTE( sarr_f , -2 ) ) die("sarr_f anyLTE fail 2");

      if ( anyLTE( arr_c  , mask_c  , -2 ) ) die("arr_c anyLTE masked fail");
      if ( anyLTE( arr_f  , mask_f  , -2 ) ) die("arr_f anyLTE masked fail");
      if ( anyLTE( sarr_c , smask_c , -2 ) ) die("sarr_c anyLTE masked fail");
      if ( anyLTE( sarr_f , smask_f , -2 ) ) die("sarr_f anyLTE masked fail");


      if ( anyGT( arr_c  , 2 ) ) die("arr_c  anyGT fail 1");
      if ( anyGT( arr_f  , 2 ) ) die("arr_f  anyGT fail 1");
      if ( anyGT( sarr_c , 2 ) ) die("sarr_c anyGT fail 1");
      if ( anyGT( sarr_f , 2 ) ) die("sarr_f anyGT fail 1");

      if ( !anyGT( arr_c  , 1 ) ) die("arr_c  anyGT fail 2");
      if ( !anyGT( arr_f  , 1 ) ) die("arr_f  anyGT fail 2");
      if ( !anyGT( sarr_c , 1 ) ) die("sarr_c anyGT fail 2");
      if ( !anyGT( sarr_f , 1 ) ) die("sarr_f anyGT fail 2");

      if ( anyGT( arr_c  , mask_c  , 1 ) ) die("arr_c anyGT masked fail");
      if ( anyGT( arr_f  , mask_f  , 1 ) ) die("arr_f anyGT masked fail");
      if ( anyGT( sarr_c , smask_c , 1 ) ) die("sarr_c anyGT masked fail");
      if ( anyGT( sarr_f , smask_f , 1 ) ) die("sarr_f anyGT masked fail");

      if ( anyGTE( arr_c  , 3 ) ) die("arr_c  anyGTE fail 1");
      if ( anyGTE( arr_f  , 3 ) ) die("arr_f  anyGTE fail 1");
      if ( anyGTE( sarr_c , 3 ) ) die("sarr_c anyGTE fail 1");
      if ( anyGTE( sarr_f , 3 ) ) die("sarr_f anyGTE fail 1");

      if ( !anyGTE( arr_c  , 2 ) ) die("arr_c  anyGTE fail 2");
      if ( !anyGTE( arr_f  , 2 ) ) die("arr_f  anyGTE fail 2");
      if ( !anyGTE( sarr_c , 2 ) ) die("sarr_c anyGTE fail 2");
      if ( !anyGTE( sarr_f , 2 ) ) die("sarr_f anyGTE fail 2");

      if ( anyGTE( arr_c  , mask_c  , 2 ) ) die("arr_c anyGTE masked fail");
      if ( anyGTE( arr_f  , mask_f  , 2 ) ) die("arr_f anyGTE masked fail");
      if ( anyGTE( sarr_c , smask_c , 2 ) ) die("sarr_c anyGTE masked fail");
      if ( anyGTE( sarr_f , smask_f , 2 ) ) die("sarr_f anyGTE masked fail");


      if ( anyEQ( arr_c  , 3 ) ) die("arr_c  anyEQ fail 1");
      if ( anyEQ( arr_f  , 3 ) ) die("arr_f  anyEQ fail 1");
      if ( anyEQ( sarr_c , 3 ) ) die("sarr_c anyEQ fail 1");
      if ( anyEQ( sarr_f , 3 ) ) die("sarr_f anyEQ fail 1");

      if ( !anyEQ( arr_c  , 1 ) ) die("arr_c  anyEQ fail 2");
      if ( !anyEQ( arr_f  , 1 ) ) die("arr_f  anyEQ fail 2");
      if ( !anyEQ( sarr_c , 1 ) ) die("sarr_c anyEQ fail 2");
      if ( !anyEQ( sarr_f , 1 ) ) die("sarr_f anyEQ fail 2");

      if ( anyEQ( arr_c  , mask_c  , 2 ) ) die("arr_c anyEQ masked fail");
      if ( anyEQ( arr_f  , mask_f  , 2 ) ) die("arr_f anyEQ masked fail");
      if ( anyEQ( sarr_c , smask_c , 2 ) ) die("sarr_c anyEQ masked fail");
      if ( anyEQ( sarr_f , smask_f , 2 ) ) die("sarr_f anyEQ masked fail");


      parallel_for( 5 , YAKL_LAMBDA (int i) {
        mask_c(i  ) = i == 2;
        mask_f(i+1) = i == 2;
      });
      for (int i=0; i < 5; i++) {
        smask_c(i  ) = i == 2;
        smask_f(i+1) = i == 2;
      }

      if ( !anyNEQ( arr_c  , 0 ) ) die("arr_c  anyNEQ fail 1");
      if ( !anyNEQ( arr_f  , 0 ) ) die("arr_f  anyNEQ fail 1");
      if ( !anyNEQ( sarr_c , 0 ) ) die("sarr_c anyNEQ fail 1");
      if ( !anyNEQ( sarr_f , 0 ) ) die("sarr_f anyNEQ fail 1");

      if ( anyNEQ( arr_c  , mask_c  , 0 ) ) die("arr_c anyNEQ masked fail");
      if ( anyNEQ( arr_f  , mask_f  , 0 ) ) die("arr_f anyNEQ masked fail");
      if ( anyNEQ( sarr_c , smask_c , 0 ) ) die("sarr_c anyNEQ masked fail");
      if ( anyNEQ( sarr_f , smask_f , 0 ) ) die("sarr_f anyNEQ masked fail");


      parallel_for( 5 , YAKL_LAMBDA (int i) {
        arr_c(i  ) = 1;
        arr_f(i+1) = 1;
      });
      for (int i=0; i < 5; i++) {
        sarr_c(i  ) = 1;
        sarr_f(i+1) = 1;
      }

      if ( anyNEQ( arr_c  , 1 ) ) die("arr_c  anyNEQ fail 2");
      if ( anyNEQ( arr_f  , 1 ) ) die("arr_f  anyNEQ fail 2");
      if ( anyNEQ( sarr_c , 1 ) ) die("sarr_c anyNEQ fail 2");
      if ( anyNEQ( sarr_f , 1 ) ) die("sarr_f anyNEQ fail 2");
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
    // count, pack
    //////////////////////////////////////////////////////////
    {
      using yakl::intrinsics::count;
      using yakl::intrinsics::pack;
      bool_c_1d c("c",10);
      bool_f_1d f("f",10);
      SArray<bool,1,10> sc;
      FSArray<bool,1,SB<10>> sf;
      
      parallel_for( 10 , YAKL_LAMBDA( int i ) {
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

        auto packed        = pack(vals_host);
        auto packed_masked = pack(vals_host,c.createHostCopy());

        if ( yakl::intrinsics::sum(packed       ) != 45 ) die("ERROR: packed c");
        if ( yakl::intrinsics::sum(packed_masked) != 20 ) die("ERROR: packed masked c");
      }

      {
        real_f_1d vals("vals",10);
        auto vals_host = vals.createHostCopy();
        for (int i=0; i < 10; i++) { vals_host(i+1) = i; }

        auto packed        = pack(vals_host);
        auto packed_masked = pack(vals_host,f.createHostCopy());

        if ( yakl::intrinsics::sum(packed       ) != 45 ) die("ERROR: packed f");
        if ( yakl::intrinsics::sum(packed_masked) != 20 ) die("ERROR: packed masked f");
      }
    }



  }
  yakl::finalize();
  
  return 0;
}

