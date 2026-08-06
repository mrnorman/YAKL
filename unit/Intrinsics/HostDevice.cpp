#include <cmath>
#include "YAKL.h"

namespace {

  KOKKOS_INLINE_FUNCTION bool close_enough(double actual, double expected, double tolerance = 1.e-12) {
    return std::abs(actual-expected) <= tolerance;
  }

  template <class V>
  KOKKOS_INLINE_FUNCTION int check_stack_componentwise(V const & a, V const & b) {
    using namespace yakl::componentwise;
    int errors = 0;
    int constexpr i = 2;
    double const x = a.data()[i];
    double const y = b.data()[i];

    errors += !close_enough((a+b).data()[i],x+y);
    errors += !close_enough((a-b).data()[i],x-y);
    errors += !close_enough((a*b).data()[i],x*y);
    errors += !close_enough((a/b).data()[i],x/y);
    errors += (a <  b).data()[i] != (x <  y);
    errors += (a >  b).data()[i] != (x >  y);
    errors += (a <= b).data()[i] != (x <= y);
    errors += (a >= b).data()[i] != (x >= y);
    errors += (a == b).data()[i] != (x == y);
    errors += (a != b).data()[i] != (x != y);
    errors += (a && b).data()[i] != (x && y);
    errors += (a || b).data()[i] != (x || y);

    // Instantiate both mixed scalar/SArray paths in device code as well.
    errors += !close_enough((a+2.).data()[i],x+2.);
    errors += !close_enough((2.+a).data()[i],2.+x);

    errors += (!a).data()[i] != (!x);
    errors += !close_enough((+a).data()[i],+x);
    errors += !close_enough((-a).data()[i],-x);
    errors += !close_enough(abs  (a).data()[i],std::abs  (x));
    errors += !close_enough(sqrt (a).data()[i],std::sqrt (x));
    errors += !close_enough(cbrt (a).data()[i],std::cbrt (x));
    errors += !close_enough(pow  (a,2.).data()[i],std::pow(x,2.));
    errors += !close_enough(sin  (a).data()[i],std::sin  (x));
    errors += !close_enough(cos  (a).data()[i],std::cos  (x));
    errors += !close_enough(tan  (a).data()[i],std::tan  (x));
    errors += !close_enough(asin (a).data()[i],std::asin (x));
    errors += !close_enough(acos (a).data()[i],std::acos (x));
    errors += !close_enough(atan (a).data()[i],std::atan (x));
    errors += !close_enough(exp  (a).data()[i],std::exp  (x));
    errors += !close_enough(log  (a).data()[i],std::log  (x));
    errors += !close_enough(log10(a).data()[i],std::log10(x));
    errors += !close_enough(log2 (a).data()[i],std::log2 (x));
    errors += !close_enough(floor(a).data()[i],std::floor(x));
    errors += !close_enough(ceil (a).data()[i],std::ceil (x));
    errors += !close_enough(round(a).data()[i],std::round(x));
    errors += isnan(a).data()[i] != std::isnan(x);
    errors += isinf(a).data()[i] != std::isinf(x);
    return errors;
  }

  template <class V, class M>
  KOKKOS_INLINE_FUNCTION int check_stack_intrinsics(V const & a, V const & sign_source, M const & mask,
                                                      int dim, int lower) {
    using namespace yakl::intrinsics;
    int errors = 0;
    auto const absolute = abs(a);
    auto const signed_values = sign(a,sign_source);
    auto const merged = merge(a,sign_source,mask);

    errors += !allocated(a) || !associated(a);
    errors += size(a) != 4 || size(a,dim) != 4;
    errors += shape(a).data()[0] != 4;
    errors += lbound(a).data()[0] != lower || lbound(a,dim) != lower;
    errors += ubound(a).data()[0] != lower+3 || ubound(a,dim) != lower+3;
    errors += epsilon(a) != std::numeric_limits<double>::epsilon();
    errors += tiny(a) != std::numeric_limits<double>::min();
    errors += huge(a) != std::numeric_limits<double>::max();
    errors += absolute.data()[0] != 2. || absolute.data()[1] != 1.;
    errors += signed_values.data()[0] != -2. || signed_values.data()[2] != -3.;
    errors += merged.data()[0] != -2. || merged.data()[1] != 1. ||
              merged.data()[2] != 3. || merged.data()[3] != 1.;
    errors += !any(mask) || all(mask) || count(mask) != 2;
    errors += sum(a) != 4. || product(a) != 24.;
    errors += minval(a) != -2. || maxval(a) != 4.;
    errors += minloc(a).data()[0] != lower || maxloc(a).data()[0] != lower+3;
    return errors;
  }

  KOKKOS_INLINE_FUNCTION int check_stack_linear_algebra() {
    using namespace yakl::intrinsics;
    yakl::SArray<double,2,2> matrix;
    yakl::SArray<double,2,2> identity;
    yakl::SArray<double,2> vector;
    matrix(0,0) = 4.; matrix(0,1) = 7.;
    matrix(1,0) = 2.; matrix(1,1) = 6.;
    identity(0,0) = 1.; identity(0,1) = 0.;
    identity(1,0) = 0.; identity(1,1) = 1.;
    vector(0) = 1.; vector(1) = 2.;

    int errors = 0;
    auto const matrix_vector = matmul_rc(matrix,vector);
    auto const matrix_matrix = matmul_rc(matrix,identity);
    auto const transposed = transpose(matrix);
    auto const column_vector = matmul_cr(transposed,vector);
    auto const column_matrix = transpose(matmul_cr(transposed,transpose(identity)));
    auto const inverse = matinv(matrix);
    auto const inverse_product = matmul_rc(inverse,matrix);
    errors += matrix_vector(0) != 18. || matrix_vector(1) != 14.;
    errors += column_vector(0) != 18. || column_vector(1) != 14.;
    errors += matrix_matrix(0,1) != 7. || column_matrix(1,0) != 2.;
    errors += transposed(0,1) != 2. || transposed(1,0) != 7.;
    errors += !close_enough(inverse_product(0,0),1.) || !close_enough(inverse_product(0,1),0.) ||
              !close_enough(inverse_product(1,0),0.) || !close_enough(inverse_product(1,1),1.);
    return errors;
  }

  KOKKOS_INLINE_FUNCTION int check_all_stack_routines() {
    yakl::SArray<double,4> c_a;
    yakl::SArray<double,4> c_b;
    yakl::SArray<double,4> c_sign;
    yakl::SArray<bool,4> c_mask;
    yakl::SArray_F<double,yakl::Bnds{-2,1}> f_a;
    yakl::SArray_F<double,yakl::Bnds{-2,1}> f_b;
    yakl::SArray_F<double,yakl::Bnds{-2,1}> f_sign;
    yakl::SArray_F<bool,yakl::Bnds{-2,1}> f_mask;
    for (int i=0; i < 4; i++) {
      double const component_value = 0.25 * (i+1);
      c_a.data()[i] = component_value;
      f_a.data()[i] = component_value;
      c_b.data()[i] = component_value+1.;
      f_b.data()[i] = component_value+1.;
      c_sign.data()[i] = i%2 == 0 ? -1. : 1.;
      f_sign.data()[i] = c_sign.data()[i];
      c_mask.data()[i] = i%2 == 0;
      f_mask.data()[i] = c_mask.data()[i];
    }

    int errors = check_stack_componentwise(c_a,c_b) + check_stack_componentwise(f_a,f_b);

    // Intrinsic reduction inputs include mixed signs and unique extrema.
    double constexpr values[4] = {-2.,-1.,3.,4.};
    for (int i=0; i < 4; i++) {
      c_a.data()[i] = values[i];
      f_a.data()[i] = values[i];
    }
    errors += check_stack_intrinsics(c_a,c_sign,c_mask,0,0);
    errors += check_stack_intrinsics(f_a,f_sign,f_mask,1,-2);
    errors += check_stack_linear_algebra();
    return errors;
  }

  template <class V>
  double sum_std_function(V const & host_values, double (*function)(double)) {
    double result = 0;
    for (size_t i=0; i < host_values.size(); i++) result += function(host_values.data()[i]);
    return result;
  }

  template <class V, class M>
  void check_dynamic_routines(V const & a, V const & b, V const & sign_source, M const & mask, int dim, int lower) {
    using namespace yakl::componentwise;
    using yakl::intrinsics::all;
    using yakl::intrinsics::allocated;
    using yakl::intrinsics::any;
    using yakl::intrinsics::associated;
    using yakl::intrinsics::count;
    using yakl::intrinsics::epsilon;
    using yakl::intrinsics::huge;
    using yakl::intrinsics::lbound;
    using yakl::intrinsics::maxloc;
    using yakl::intrinsics::maxval;
    using yakl::intrinsics::merge;
    using yakl::intrinsics::minloc;
    using yakl::intrinsics::minval;
    using yakl::intrinsics::product;
    using yakl::intrinsics::shape;
    using yakl::intrinsics::sign;
    using yakl::intrinsics::size;
    using yakl::intrinsics::sum;
    using yakl::intrinsics::tiny;
    using yakl::intrinsics::ubound;

    auto check_sum = [] (auto const & values, double expected) {
      if (!close_enough(yakl::intrinsics::sum(values),expected)) Kokkos::abort("dynamic componentwise result is incorrect");
    };

    if (!allocated(a) || !associated(a) || size(a) != 4 || size(a,dim) != 4 || shape(a).data()[0] != 4 ||
        lbound(a,dim) != lower || ubound(a,dim) != lower+3 || lbound(a).data()[0] != lower ||
        ubound(a).data()[0] != lower+3) {
      Kokkos::abort("dynamic Array metadata intrinsic failed");
    }
    if (epsilon(a) != std::numeric_limits<double>::epsilon() || tiny(a) != std::numeric_limits<double>::min() ||
        huge(a) != std::numeric_limits<double>::max()) {
      Kokkos::abort("dynamic Array numeric inquiry intrinsic failed");
    }
    check_sum(yakl::intrinsics::abs(a),2.5);
    check_sum(sign(a,sign_source),0.5);
    check_sum(merge(a,b,mask),4.5);
    if (!any(mask) || all(mask) || count(mask) != 2 || !close_enough(sum(a),2.5) ||
        !close_enough(product(a),0.09375) || minval(a) != 0.25 || maxval(a) != 1. ||
        minloc(a).data()[0] != lower || maxloc(a).data()[0] != lower+3) {
      Kokkos::abort("dynamic Array reduction intrinsic failed");
    }

    check_sum(a+b,9.);
    check_sum(a-b,-4.);
    check_sum(a*b,4.375);
    check_sum(a/b,0.2+1./3.+3./7.+0.5);
    check_sum(a+2.,10.5);
    check_sum(2.+a,10.5);
    if (count(a < b) != 4 || count(a > b) != 0 || count(a <= b) != 4 || count(a >= b) != 0 ||
        count(a == b) != 0 || count(a != b) != 4 || count(a && b) != 4 || count(a || b) != 4) {
      Kokkos::abort("dynamic Array binary componentwise operator failed");
    }

    check_sum(+a,2.5);
    check_sum(-a,-2.5);
    if (count(!a) != 0) Kokkos::abort("dynamic Array logical-not componentwise operator failed");
    check_sum(abs  (a),sum_std_function(a.createHostCopy(),static_cast<double (*)(double)>(std::abs)));
    check_sum(sqrt (a),sum_std_function(a.createHostCopy(),std::sqrt));
    check_sum(cbrt (a),sum_std_function(a.createHostCopy(),std::cbrt));
    check_sum(pow(a,2.),1.875);
    check_sum(sin  (a),sum_std_function(a.createHostCopy(),std::sin));
    check_sum(cos  (a),sum_std_function(a.createHostCopy(),std::cos));
    check_sum(tan  (a),sum_std_function(a.createHostCopy(),std::tan));
    check_sum(asin (a),sum_std_function(a.createHostCopy(),std::asin));
    check_sum(acos (a),sum_std_function(a.createHostCopy(),std::acos));
    check_sum(atan (a),sum_std_function(a.createHostCopy(),std::atan));
    check_sum(exp  (a),sum_std_function(a.createHostCopy(),std::exp));
    check_sum(log  (a),sum_std_function(a.createHostCopy(),std::log));
    check_sum(log10(a),sum_std_function(a.createHostCopy(),std::log10));
    check_sum(log2 (a),sum_std_function(a.createHostCopy(),std::log2));
    check_sum(floor(a),1.);
    check_sum(ceil (a),4.);
    check_sum(round(a),3.);
    if (count(isnan(a)) != 0 || count(isinf(a)) != 0) Kokkos::abort("dynamic Array classification operation failed");
  }

}

void test_host_device_intrinsics() {
  int const host_errors = check_all_stack_routines();
  if (host_errors != 0) Kokkos::abort("SArray intrinsic/componentwise routine failed on the host");

  yakl::ScalarLiveOut<int> device_errors(0);
  yakl::parallel_for("SArray host-device intrinsic coverage",1,KOKKOS_LAMBDA (int) {
    device_errors = check_all_stack_routines();
  });
  if (device_errors.hostRead() != 0) Kokkos::abort("SArray intrinsic/componentwise routine failed on the device");

  yakl::Array<double *,Kokkos::HostSpace> c_a("c_a",4);
  yakl::Array<double *,Kokkos::HostSpace> c_b("c_b",4);
  yakl::Array<double *,Kokkos::HostSpace> c_sign("c_sign",4);
  yakl::Array<bool *,Kokkos::HostSpace> c_mask("c_mask",4);
  yakl::Array_F<double *,Kokkos::HostSpace> f_a("f_a",{-2,1});
  yakl::Array_F<double *,Kokkos::HostSpace> f_b("f_b",{-2,1});
  yakl::Array_F<double *,Kokkos::HostSpace> f_sign("f_sign",{-2,1});
  yakl::Array_F<bool *,Kokkos::HostSpace> f_mask("f_mask",{-2,1});
  for (int i=0; i < 4; i++) {
    double const value = 0.25 * (i+1);
    c_a.data()[i] = value; f_a.data()[i] = value;
    c_b.data()[i] = value+1.; f_b.data()[i] = value+1.;
    c_sign.data()[i] = i%2 == 0 ? -1. : 1.; f_sign.data()[i] = c_sign.data()[i];
    c_mask.data()[i] = i%2 == 0; f_mask.data()[i] = c_mask.data()[i];
  }

  auto c_a_device = c_a.createDeviceCopy();
  auto c_b_device = c_b.createDeviceCopy();
  auto c_sign_device = c_sign.createDeviceCopy();
  auto c_mask_device = c_mask.createDeviceCopy();
  auto f_a_device = f_a.createDeviceCopy();
  auto f_b_device = f_b.createDeviceCopy();
  auto f_sign_device = f_sign.createDeviceCopy();
  auto f_mask_device = f_mask.createDeviceCopy();

  // Inquiry intrinsics do not allocate or launch nested kernels, so dynamic Array metadata should also be callable
  // directly from device code. Allocation-producing componentwise routines are instead tested below for both memory spaces.
  yakl::ScalarLiveOut<int> array_device_inquiry_errors(0);
  yakl::parallel_for("Array host-device inquiry intrinsic coverage",1,KOKKOS_LAMBDA (int) {
    using namespace yakl::intrinsics;
    int errors = 0;
    errors += !allocated(c_a_device) || !associated(c_a_device);
    errors += size(c_a_device) != 4 || size(c_a_device,0) != 4;
    errors += shape(c_a_device).data()[0] != 4;
    errors += lbound(c_a_device,0) != 0 || ubound(c_a_device,0) != 3;
    errors += !allocated(f_a_device) || !associated(f_a_device);
    errors += size(f_a_device) != 4 || size(f_a_device,1) != 4;
    errors += shape(f_a_device).data()[0] != 4;
    errors += lbound(f_a_device,1) != -2 || ubound(f_a_device,1) != 1;
    errors += epsilon(c_a_device) != std::numeric_limits<double>::epsilon();
    errors += tiny(f_a_device) != std::numeric_limits<double>::min();
    errors += huge(f_a_device) != std::numeric_limits<double>::max();
    array_device_inquiry_errors = errors;
  });
  if (array_device_inquiry_errors.hostRead() != 0) {
    Kokkos::abort("Array or Array_F inquiry intrinsic failed in device code");
  }

  check_dynamic_routines(c_a,c_b,c_sign,c_mask,0,0);
  check_dynamic_routines(c_a_device,c_b_device,c_sign_device,c_mask_device,0,0);
  check_dynamic_routines(f_a,f_b,f_sign,f_mask,1,-2);
  check_dynamic_routines(f_a_device,f_b_device,f_sign_device,f_mask_device,1,-2);
}
