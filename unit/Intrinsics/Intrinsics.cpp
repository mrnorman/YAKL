
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::Array_F;
using yakl::parallel_for;
using yakl::SArray;
using yakl::SArray_F;
using yakl::Bnds;

typedef double real;

typedef Array  <real ** ,yakl::DeviceSpace> real_c_2d;

typedef Array_F<real ** ,yakl::DeviceSpace> real_f_2d;

typedef Array  <bool *  ,yakl::DeviceSpace> bool_c_1d;

void test_host_device_intrinsics();


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    // Scalar inquiry overloads are not covered by the comprehensive host/device array tests.
    {
      real scalar = 1;
      using yakl::intrinsics::epsilon;
      if (epsilon(scalar) != std::numeric_limits<real>::epsilon()) die("scalar wrong epsilon");
      using yakl::intrinsics::tiny;
      if (tiny(scalar) != std::numeric_limits<real>::min()) die("scalar wrong tiny");
      using yakl::intrinsics::huge;
      if (huge(scalar) != std::numeric_limits<real>::max()) die("scalar wrong huge");
      if (yakl::intrinsics::sign(13.1 , -0.1 ) != -13.1) die("ERROR: sign does not work");
    }
    // Large minval/maxval inputs complement the small comprehensive host/device cases.
    {
      int constexpr n = 1024;
      Array<double *,Kokkos::HostSpace> host("minmax host",n);
      SArray<double,n> cStack;
      SArray_F<double,Bnds{1,n}> fStack;
      for (int i=0; i < n; i++) {
        host  .data()[i] = n-i;
        cStack.data()[i] = n-i;
        fStack.data()[i] = n-i;
      }
      auto device = host.createDeviceCopy();

      using yakl::intrinsics::minval;
      using yakl::intrinsics::maxval;
      if (minval(host) != 1 || minval(device) != 1 || minval(cStack) != 1 || minval(fStack) != 1) {
        die("ERROR: wrong large-input minval");
      }
      if (maxval(host) != n || maxval(device) != n || maxval(cStack) != n || maxval(fStack) != n) {
        die("ERROR: wrong large-input maxval");
      }

      yakl::ScalarLiveOut<int> stackErrors(0);
      yakl::parallel_for( 1 , KOKKOS_LAMBDA (int i) {
        if (minval(cStack) != 1 || minval(fStack) != 1 || maxval(cStack) != n || maxval(fStack) != n) stackErrors = 1;
      });
      if (stackErrors.hostRead() != 0) die("ERROR: wrong large stack-array minval or maxval on device");
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Exhaustive multidimensional unpack_global_index, minloc, and maxloc
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    {
      int constexpr d1 = 2;
      int constexpr d2 = 3;
      int constexpr d3 = 4;
      int constexpr numElements = d1*d2*d3;
      using CStack = SArray<int,d1,d2,d3>;
      using FStack = SArray_F<int,Bnds{-2,-1},Bnds{3,5},Bnds{7,10}>;

      Array<int ***,Kokkos::HostSpace> cHost("cHost",d1,d2,d3);
      Array_F<int ***,Kokkos::HostSpace> fHost("fHost",{-2,-1},{3,5},{7,10});
      CStack cStack;
      FStack fStack;
      for (int iglob=0; iglob < numElements; iglob++) {
        cHost.data()[iglob] = 100 + iglob;
        fHost.data()[iglob] = 200 + iglob;
        cStack.data()[iglob] = 300 + iglob;
        fStack.data()[iglob] = 400 + iglob;
      }

      // Check every linear index, not only corners. The indexed value verifies
      // that returned coordinates address the same physical element.
      for (int iglob=0; iglob < numElements; iglob++) {
        int const ci =  iglob / (d2*d3);
        int const cj = (iglob / d3) % d2;
        int const ck =  iglob             % d3;
        auto const cHostLoc = cHost.unpack_global_index(iglob);
        auto const cStackLoc = cStack.unpack_global_index(iglob);
        if (cHostLoc(0) != ci || cHostLoc(1) != cj || cHostLoc(2) != ck ||
            cHost(cHostLoc(0),cHostLoc(1),cHostLoc(2)) != cHost.data()[iglob]) {
          die("ERROR: Array::unpack_global_index returned incorrect C-style coordinates");
        }
        if (cStackLoc(0) != ci || cStackLoc(1) != cj || cStackLoc(2) != ck ||
            cStack(cStackLoc(0),cStackLoc(1),cStackLoc(2)) != cStack.data()[iglob]) {
          die("ERROR: SArray::unpack_global_index returned incorrect C-style coordinates");
        }

        int const fi = -2 +  iglob             % d1;
        int const fj =  3 + (iglob / d1)       % d2;
        int const fk =  7 + (iglob / (d1*d2))  % d3;
        auto const fHostLoc = fHost.unpack_global_index(iglob);
        auto const fStackLoc = fStack.unpack_global_index(iglob);
        if (fHostLoc(1) != fi || fHostLoc(2) != fj || fHostLoc(3) != fk ||
            fHost(fHostLoc(1),fHostLoc(2),fHostLoc(3)) != fHost.data()[iglob]) {
          die("ERROR: Array_F::unpack_global_index returned incorrect Fortran-style coordinates");
        }
        if (fStackLoc(1) != fi || fStackLoc(2) != fj || fStackLoc(3) != fk ||
            fStack(fStackLoc(1),fStackLoc(2),fStackLoc(3)) != fStack.data()[iglob]) {
          die("ERROR: SArray_F::unpack_global_index returned incorrect Fortran-style coordinates");
        }
      }

      // Place unique extrema where every coordinate matters and none of the
      // locations is the first or last linear element.
      cHost(1,0,3) = -1000;
      cHost(0,2,1) =  1000;
      cStack(1,0,3) = -1000;
      cStack(0,2,1) =  1000;
      fHost (-1,3,10) = -2000;
      fHost (-2,5, 8) =  2000;
      fStack(-1,3,10) = -2000;
      fStack(-2,5, 8) =  2000;

      auto cDevice = cHost.createDeviceCopy();
      auto fDevice = fHost.createDeviceCopy();

      // Compile and execute all four decoders on the device as well. Dynamic
      // arrays use one thread per element; stack arrays are checked in one thread.
      yakl::ScalarLiveOut<int> deviceErrors(0);
      yakl::parallel_for( "unpack_global_index dynamic arrays" , numElements , KOKKOS_LAMBDA (size_t iglob) {
        size_t const ci =  iglob / (d2*d3);
        size_t const cj = (iglob / d3) % d2;
        size_t const ck =  iglob             % d3;
        auto const cLoc = cDevice.unpack_global_index(iglob);
        if (cLoc(0) != ci || cLoc(1) != cj || cLoc(2) != ck ||
            cDevice(cLoc(0),cLoc(1),cLoc(2)) != cDevice.data()[iglob]) {
          Kokkos::atomic_add(&deviceErrors(),1);
        }

        ptrdiff_t const fi = -2 + static_cast<ptrdiff_t>( iglob             % d1);
        ptrdiff_t const fj =  3 + static_cast<ptrdiff_t>((iglob / d1)      % d2);
        ptrdiff_t const fk =  7 + static_cast<ptrdiff_t>((iglob / (d1*d2)) % d3);
        auto const fLoc = fDevice.unpack_global_index(iglob);
        if (fLoc(1) != fi || fLoc(2) != fj || fLoc(3) != fk ||
            fDevice(fLoc(1),fLoc(2),fLoc(3)) != fDevice.data()[iglob]) {
          Kokkos::atomic_add(&deviceErrors(),1);
        }
      });
      yakl::parallel_for( "unpack_global_index stack arrays" , 1 , KOKKOS_LAMBDA (int) {
        for (int iglob=0; iglob < numElements; iglob++) {
          int const ci =  iglob / (d2*d3);
          int const cj = (iglob / d3) % d2;
          int const ck =  iglob             % d3;
          auto const cLoc = cStack.unpack_global_index(iglob);
          if (cLoc(0) != ci || cLoc(1) != cj || cLoc(2) != ck ||
              cStack(cLoc(0),cLoc(1),cLoc(2)) != cStack.data()[iglob]) {
            Kokkos::atomic_add(&deviceErrors(),1);
          }

          int const fi = -2 +  iglob             % d1;
          int const fj =  3 + (iglob / d1)       % d2;
          int const fk =  7 + (iglob / (d1*d2))  % d3;
          auto const fLoc = fStack.unpack_global_index(iglob);
          if (fLoc(1) != fi || fLoc(2) != fj || fLoc(3) != fk ||
              fStack(fLoc(1),fLoc(2),fLoc(3)) != fStack.data()[iglob]) {
            Kokkos::atomic_add(&deviceErrors(),1);
          }
        }

        auto const cMin = yakl::intrinsics::minloc(cStack);
        auto const cMax = yakl::intrinsics::maxloc(cStack);
        auto const fMin = yakl::intrinsics::minloc(fStack);
        auto const fMax = yakl::intrinsics::maxloc(fStack);
        if (cMin(0) != 1 || cMin(1) != 0 || cMin(2) != 3 ||
            cMax(0) != 0 || cMax(1) != 2 || cMax(2) != 1 ||
            fMin(1) != -1 || fMin(2) != 3 || fMin(3) != 10 ||
            fMax(1) != -2 || fMax(2) != 5 || fMax(3) != 8) {
          Kokkos::atomic_add(&deviceErrors(),1);
        }
      });
      if (deviceErrors.hostRead() != 0) {
        die("ERROR: unpack_global_index, minloc, or maxloc failed on the device");
      }

      using yakl::intrinsics::minloc;
      using yakl::intrinsics::maxloc;
      auto const cHostMin = minloc(cHost);
      auto const cHostMax = maxloc(cHost);
      auto const cDeviceMin = minloc(cDevice);
      auto const cDeviceMax = maxloc(cDevice);
      auto const cStackMin = minloc(cStack);
      auto const cStackMax = maxloc(cStack);
      if (cHostMin(0) != 1 || cHostMin(1) != 0 || cHostMin(2) != 3 ||
          cDeviceMin(0) != 1 || cDeviceMin(1) != 0 || cDeviceMin(2) != 3 ||
          cStackMin(0) != 1 || cStackMin(1) != 0 || cStackMin(2) != 3) {
        die("ERROR: multidimensional C-style minloc returned an incorrect location");
      }
      if (cHostMax(0) != 0 || cHostMax(1) != 2 || cHostMax(2) != 1 ||
          cDeviceMax(0) != 0 || cDeviceMax(1) != 2 || cDeviceMax(2) != 1 ||
          cStackMax(0) != 0 || cStackMax(1) != 2 || cStackMax(2) != 1) {
        die("ERROR: multidimensional C-style maxloc returned an incorrect location");
      }

      auto const fHostMin = minloc(fHost);
      auto const fHostMax = maxloc(fHost);
      auto const fDeviceMin = minloc(fDevice);
      auto const fDeviceMax = maxloc(fDevice);
      auto const fStackMin = minloc(fStack);
      auto const fStackMax = maxloc(fStack);
      if (fHostMin(1) != -1 || fHostMin(2) != 3 || fHostMin(3) != 10 ||
          fDeviceMin(1) != -1 || fDeviceMin(2) != 3 || fDeviceMin(3) != 10 ||
          fStackMin(1) != -1 || fStackMin(2) != 3 || fStackMin(3) != 10) {
        die("ERROR: multidimensional Fortran-style minloc returned an incorrect location");
      }
      if (fHostMax(1) != -2 || fHostMax(2) != 5 || fHostMax(3) != 8 ||
          fDeviceMax(1) != -2 || fDeviceMax(2) != 5 || fDeviceMax(3) != 8 ||
          fStackMax(1) != -2 || fStackMax(2) != 5 || fStackMax(3) != 8) {
        die("ERROR: multidimensional Fortran-style maxloc returned an incorrect location");
      }

      // Tied extrema must select the first element in linear memory order for
      // host arrays, device arrays, and stack arrays alike.
      cHost (0,1,2) = -1000;
      cHost (0,0,3) =  1000;
      cStack(0,1,2) = -1000;
      cStack(0,0,3) =  1000;
      fHost (-2,4,7) = -2000;
      fHost (-1,3,7) =  2000;
      fStack(-2,4,7) = -2000;
      fStack(-1,3,7) =  2000;
      cHost.deep_copy_to(cDevice);
      fHost.deep_copy_to(fDevice);

      auto const cTiedHostMin = minloc(cHost);
      auto const cTiedHostMax = maxloc(cHost);
      auto const cTiedDeviceMin = minloc(cDevice);
      auto const cTiedDeviceMax = maxloc(cDevice);
      auto const cTiedStackMin = minloc(cStack);
      auto const cTiedStackMax = maxloc(cStack);
      if (cTiedHostMin(0) != 0 || cTiedHostMin(1) != 1 || cTiedHostMin(2) != 2 ||
          cTiedDeviceMin(0) != 0 || cTiedDeviceMin(1) != 1 || cTiedDeviceMin(2) != 2 ||
          cTiedStackMin(0) != 0 || cTiedStackMin(1) != 1 || cTiedStackMin(2) != 2) {
        die("ERROR: tied C-style minloc did not return the first location");
      }
      if (cTiedHostMax(0) != 0 || cTiedHostMax(1) != 0 || cTiedHostMax(2) != 3 ||
          cTiedDeviceMax(0) != 0 || cTiedDeviceMax(1) != 0 || cTiedDeviceMax(2) != 3 ||
          cTiedStackMax(0) != 0 || cTiedStackMax(1) != 0 || cTiedStackMax(2) != 3) {
        die("ERROR: tied C-style maxloc did not return the first location");
      }

      auto const fTiedHostMin = minloc(fHost);
      auto const fTiedHostMax = maxloc(fHost);
      auto const fTiedDeviceMin = minloc(fDevice);
      auto const fTiedDeviceMax = maxloc(fDevice);
      auto const fTiedStackMin = minloc(fStack);
      auto const fTiedStackMax = maxloc(fStack);
      if (fTiedHostMin(1) != -2 || fTiedHostMin(2) != 4 || fTiedHostMin(3) != 7 ||
          fTiedDeviceMin(1) != -2 || fTiedDeviceMin(2) != 4 || fTiedDeviceMin(3) != 7 ||
          fTiedStackMin(1) != -2 || fTiedStackMin(2) != 4 || fTiedStackMin(3) != 7) {
        die("ERROR: tied Fortran-style minloc did not return the first location");
      }
      if (fTiedHostMax(1) != -1 || fTiedHostMax(2) != 3 || fTiedHostMax(3) != 7 ||
          fTiedDeviceMax(1) != -1 || fTiedDeviceMax(2) != 3 || fTiedDeviceMax(3) != 7 ||
          fTiedStackMax(1) != -1 || fTiedStackMax(2) != 3 || fTiedStackMax(3) != 7) {
        die("ERROR: tied Fortran-style maxloc did not return the first location");
      }

      yakl::ScalarLiveOut<int> tiedStackErrors(0);
      yakl::parallel_for( "tied stack minloc and maxloc" , 1 , KOKKOS_LAMBDA (int) {
        auto const cMin = yakl::intrinsics::minloc(cStack);
        auto const cMax = yakl::intrinsics::maxloc(cStack);
        auto const fMin = yakl::intrinsics::minloc(fStack);
        auto const fMax = yakl::intrinsics::maxloc(fStack);
        if (cMin(0) != 0 || cMin(1) != 1 || cMin(2) != 2 ||
            cMax(0) != 0 || cMax(1) != 0 || cMax(2) != 3 ||
            fMin(1) != -2 || fMin(2) != 4 || fMin(3) != 7 ||
            fMax(1) != -1 || fMax(2) != 3 || fMax(3) != 7) {
          tiedStackErrors = 1;
        }
      });
      if (tiedStackErrors.hostRead() != 0) {
        die("ERROR: device execution returned incorrect tied stack minloc or maxloc");
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Tied minloc and maxloc across multiple reduction blocks
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    {
      int constexpr n = 4097;
      int constexpr lowerBound = -37;
      Array<int *,Kokkos::HostSpace> cHost("large tied C array",n);
      Array_F<int *,Kokkos::HostSpace> fHost("large tied Fortran array",{lowerBound,lowerBound+n-1});
      for (int i=0; i < n; i++) {
        cHost.data()[i] = 0;
        fHost.data()[i] = 0;
      }
      for (int i : {17,2048,4096}) {
        cHost.data()[i] = -2;
        fHost.data()[i] = -2;
      }
      for (int i : {9,3000,4000}) {
        cHost.data()[i] = 2;
        fHost.data()[i] = 2;
      }
      auto cDevice = cHost.createDeviceCopy();
      auto fDevice = fHost.createDeviceCopy();

      using yakl::intrinsics::minloc;
      using yakl::intrinsics::maxloc;
      if (minloc(cHost)(0) != 17 || minloc(cDevice)(0) != 17 ||
          maxloc(cHost)(0) !=  9 || maxloc(cDevice)(0) !=  9) {
        die("ERROR: C-style minloc or maxloc disagreed across reduction blocks");
      }
      if (minloc(fHost)(1) != lowerBound+17 || minloc(fDevice)(1) != lowerBound+17 ||
          maxloc(fHost)(1) != lowerBound+9  || maxloc(fDevice)(1) != lowerBound+9) {
        die("ERROR: Fortran-style minloc or maxloc disagreed across reduction blocks");
      }
    }

    ///////////////////////////////////////
    // Unallocated dynamic arrays; allocated cases are covered by the comprehensive host/device tests.
    ///////////////////////////////////////
    {
      using yakl::intrinsics::allocated;
      using yakl::intrinsics::associated;
      real_c_2d arr_c_no;
      real_f_2d arr_f_no;
      if (allocated(arr_c_no)) die("arr_c_no error allocated");
      if (allocated(arr_f_no)) die("arr_f_no error allocated");
      if (associated(arr_c_no)) die("arr_c_no error associated");
      if (associated(arr_f_no)) die("arr_f_no error associated");
    }


    ////////////////////////////////////////////////
    // product
    ////////////////////////////////////////////////
    {
      int constexpr n = 1024;
      using yakl::intrinsics::product;
      Array  <double *,yakl::DeviceSpace> d_arr("d_arr",n);
      Array  <double *,Kokkos::HostSpace> h_arr("h_arr",n);
      SArray  <double,n>                cs_arr;
      SArray_F<double,Bnds{1,n}>            fs_arr;
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
    // Large any/all reductions stress concurrent reducer updates and late dissenting values.
    ////////////////////////////////////////////////
    {
      using yakl::intrinsics::any;
      using yakl::intrinsics::all;

      // A large alternating input makes many device threads contribute both reduction outcomes. Repeating the
      // reductions increases the chance that a backend or implementation containing shared non-atomic writes fails.
      size_t constexpr raceSize = 1024*1024;
      int    constexpr repeats  = 32;
      bool_c_1d raceValues("raceValues",raceSize);
      yakl::parallel_for( raceSize , KOKKOS_LAMBDA (size_t i) {
        raceValues(i) = i%2 == 0;
      });
      for (int repeat=0; repeat < repeats; repeat++) {
        if (!any(raceValues)) die("any failed for large alternating device input");
        if ( all(raceValues)) die("all failed for large alternating device input");
      }

      // Exercise unanimous reductions and a dissenting value at the final index, which can be handled by a late block.
      raceValues = false;
      if ( any(raceValues)) die("any failed for large all-false device input");
      if ( all(raceValues)) die("all failed for large all-false device input");
      yakl::parallel_for( raceSize , KOKKOS_LAMBDA (size_t i) {
        raceValues(i) = i == raceSize-1;
      });
      if (!any(raceValues)) die("any failed for a lone true value in the final device block");
      if ( all(raceValues)) die("all failed for a lone true value in the final device block");

      raceValues = true;
      if (!any(raceValues)) die("any failed for large all-true device input");
      if (!all(raceValues)) die("all failed for large all-true device input");
      yakl::parallel_for( raceSize , KOKKOS_LAMBDA (size_t i) {
        raceValues(i) = i != raceSize-1;
      });
      if (!any(raceValues)) die("any failed for a lone false value in the final device block");
      if ( all(raceValues)) die("all failed for a lone false value in the final device block");
    }


    //////////////////////////////////////////////////////////
    // matmul_cr, matmul_rc, matinv, transpose
    //////////////////////////////////////////////////////////
    {
      using yakl::intrinsics::matmul_rc;
      using yakl::intrinsics::matmul_cr;
      using yakl::intrinsics::matinv;
      using yakl::intrinsics::transpose;
      SArray<real,3,3> A1_c;
      SArray<real,3,3> A2_c;
      SArray<real,3> b_c;

      SArray<real,3> A1_b_ref;
      SArray<real,3,3> A1_A2_ref;

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

      real adiff_A1_b_c  = 0;
      real adiff_A1_A2_c = 0;
      for (int i=0; i < 3; i++) {
        adiff_A1_b_c += abs( A1_b_c(i  ) - A1_b_ref(i) );
      }
      for (int j=0; j < 3; j++) {
        for (int i=0; i < 3; i++) {
          adiff_A1_A2_c += abs( A1_A2_c(j  ,i  ) - A1_A2_ref(j  ,i) );
        }
      }

      if (adiff_A1_b_c  >= 1.e-13) die("ERROR: incorrect adiff_A1_b_c  rc");
      if (adiff_A1_A2_c >= 1.e-13) die("ERROR: incorrect adiff_A1_A2_c rc");

      auto trans_A1_c = transpose( A1_c );
      auto trans_A2_c = transpose( A2_c );

      A1_b_c  = matmul_cr( trans_A1_c , b_c  );
      A1_A2_c = matmul_cr( trans_A1_c , trans_A2_c );

      A1_A2_c = transpose( A1_A2_c );

      adiff_A1_b_c  = 0;
      adiff_A1_A2_c = 0;
      for (int i=0; i < 3; i++) {
        adiff_A1_b_c += abs( A1_b_c(i  ) - A1_b_ref(i) );
      }
      for (int j=0; j < 3; j++) {
        for (int i=0; i < 3; i++) {
          adiff_A1_A2_c += abs( A1_A2_c(j  ,i  ) - A1_A2_ref(j  ,i) );
        }
      }

      if (adiff_A1_b_c  >= 1.e-13) die("ERROR: incorrect adiff_A1_b_c  cr");
      if (adiff_A1_A2_c >= 1.e-13) die("ERROR: incorrect adiff_A1_A2_c cr");


      A1_c(0,0) = 1;
      A1_c(0,1) = 0;
      A1_c(0,2) = 0;
      A1_c(1,0) = 1;
      A1_c(1,1) = 0.5;
      A1_c(1,2) = 0.25;
      A1_c(2,0) = 1;
      A1_c(2,1) = 1;
      A1_c(2,2) = 1;

      auto A1_inv_c = matinv( A1_c );
      auto identity_c = matmul_rc( A1_inv_c , A1_c );

      real adiff_inv_c = 0;
      for (int j=0; j < 3; j++) {
        for (int i=0; i < 3; i++) {
          if (i == j) {
            adiff_inv_c += abs( identity_c(j  ,i  ) - 1 );
          } else {
            adiff_inv_c += abs( identity_c(j  ,i  )     );
          }
        }
      }
      if (adiff_inv_c >= 1.e-13) die("ERROR: incorrect adiff_inv_c");
    }



    {
      using namespace yakl::componentwise;
      using yakl::intrinsics::sum;
      using yakl::intrinsics::count;
      int constexpr n = 1024;
      Array<float  *,yakl::DeviceSpace> a("a",n);
      Array<double *,yakl::DeviceSpace> b("b",n);
      Array<float  *,Kokkos::HostSpace> c("c",n);
      Array<double *,Kokkos::HostSpace> d("d",n);
      SArray  <float ,n>                e;
      SArray  <double,n>                f;
      SArray_F<float ,Bnds{1,n}>            g;
      SArray_F<double,Bnds{1,n}>            h;
      double                            i;
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

      if ( std::abs(sum(-a) - (-n))/n > 1.e-7 ) die("ERROR: unary negative failed");
      if ( std::abs(sum(+a) - (+n))/n > 1.e-7 ) die("ERROR: unary positive failed");

      bv = 0.1;  b = bv;
      if ( count( abs((sqrt (b    )) - (std::sqrt (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary sqrt  failed");
      if ( count( abs((cbrt (b    )) - (std::cbrt (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary cbrt  failed");
      if ( count( abs((pow  (b,0.2)) - (std::pow  (bv,0.2))) > 1.e-7 ) != 0 ) die("Error: unary pow   failed");
      if ( count( abs((sin  (b    )) - (std::sin  (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary sin   failed");
      if ( count( abs((cos  (b    )) - (std::cos  (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary cos   failed");
      if ( count( abs((tan  (b    )) - (std::tan  (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary tan   failed");
      if ( count( abs((asin (b    )) - (std::asin (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary asin  failed");
      if ( count( abs((acos (b    )) - (std::acos (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary acos  failed");
      if ( count( abs((atan (b    )) - (std::atan (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary atan  failed");
      if ( count( abs((exp  (b    )) - (std::exp  (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary exp   failed");
      if ( count( abs((log  (b    )) - (std::log  (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary log   failed");
      if ( count( abs((log10(b    )) - (std::log10(bv    ))) > 1.e-7 ) != 0 ) die("Error: unary log10 failed");
      if ( count( abs((log2 (b    )) - (std::log2 (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary log2  failed");
      if ( count( abs((floor(b    )) - (std::floor(bv    ))) > 1.e-7 ) != 0 ) die("Error: unary floor failed");
      if ( count( abs((ceil (b    )) - (std::ceil (bv    ))) > 1.e-7 ) != 0 ) die("Error: unary ceil  failed");
      if ( count( abs((round(b    )) - (std::round(bv    ))) > 1.e-7 ) != 0 ) die("Error: unary round failed");
      if ( count( abs((isnan(b    )) - (std::isnan(bv    ))) > 1.e-7 ) != 0 ) die("Error: unary isnan failed");
      if ( count( abs((isinf(b    )) - (std::isinf(bv    ))) > 1.e-7 ) != 0 ) die("Error: unary isinf failed");

    }


    yakl::timer_stop("main");

    test_host_device_intrinsics();

  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}
