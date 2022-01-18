
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;

typedef float real;

typedef Array<real,1,memHost,styleC> realHost1d;

typedef Array<real,1,memDevice,styleC> real1d;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


extern "C" void add(real *a_p, real *b_p, real *c_p, int &n) {
  realHost1d a_host("a_host",a_p,n);
  realHost1d b_host("b_host",b_p,n);
  realHost1d c_host("c_host",c_p,n);

  real1d a("a",n);
  real1d b("b",n);
  real1d c("c",n);
  
  parallel_for( n , YAKL_LAMBDA (int i) {
    b(i) = 1;
    c(i) = 2;
  });

  parallel_for( n , YAKL_LAMBDA (int i) {
    a(i) = b(i) + c(i);
  });

  a.deep_copy_to(a_host);
  b.deep_copy_to(b_host);
  c.deep_copy_to(c_host);
  yakl::fence();
}

