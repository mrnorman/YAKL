
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

typedef float real;

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
      real scalar;

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






  }
  yakl::finalize();
  
  return 0;
}

