
#include <array>
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::Array_F;
using yakl::parallel_for;
using yakl::Bounds;
using yakl::SimpleBounds;
using yakl::COLON;

typedef Array  <size_t *,Kokkos::HostSpace> int1d;
typedef Array_F<size_t *,Kokkos::HostSpace> int1d_f;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main(int argc, char **argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    int constexpr n1 = 100;
    int constexpr n2 = 1024;
    int1d sum_a("sum_a",n1);
    int1d sum_b("sum_b",n1);
    int1d sum_c("sum_c",n1);
    #pragma omp parallel for
    for (int i1=0; i1 < n1; i1++) {
      int1d a("a",n2);
      int1d b("b",n2);
      int1d c("c",n2);
      auto copy_a = a;
      auto copy_b = b;
      auto copy_c = c;
      for (int i2=0; i2 < n2; i2++) {
        copy_a(i2) = i1 + 1;
        copy_b(i2) = i1 + 2;
        copy_c(i2) = i1 + i2;
      }
      auto copy_sum_a = sum_a;
      auto copy_sum_b = sum_b;
      auto copy_sum_c = sum_c;
      #ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS
      // Kokkos Threads kernels may only be launched by the master process, not by an OpenMP worker thread.
      size_t local_sum_a = 0;
      size_t local_sum_b = 0;
      size_t local_sum_c = 0;
      for (int i2=0; i2 < n2; i2++) {
        local_sum_a += a(i2);
        local_sum_b += b(i2);
        local_sum_c += c(i2);
      }
      copy_sum_a(i1) = local_sum_a;
      copy_sum_b(i1) = local_sum_b;
      copy_sum_c(i1) = local_sum_c;
      #else
      copy_sum_a(i1) = yakl::intrinsics::sum(a);
      copy_sum_b(i1) = yakl::intrinsics::sum(b);
      copy_sum_c(i1) = yakl::intrinsics::sum(c);
      #endif
    }
    #pragma omp parallel for
    for (int i1=0; i1 < n1; i1++) {
      int1d a("a",n2);
      int1d b("b",n2);
      int1d c("c",n2);
      for (int repeat=0; repeat < 48; repeat++) {
        a = int1d("a",n2);
        b = int1d("b",n2);
        c = int1d("c",n2);
      }
    }
    #pragma omp parallel for
    for (int i1=0; i1 < n1; i1++) {
      std::array<int1d,5> copies_a;
      std::array<int1d,5> copies_b;
      std::array<int1d,5> copies_c;
      for (int copy=0; copy < 5; copy++) {
        copies_a[copy] = sum_a;
        copies_b[copy] = sum_b;
        copies_c[copy] = sum_c;
      }
    }
  }
  {
    int constexpr n1 = 100;
    int constexpr n2 = 1024;
    int1d_f sum_a("sum_a",n1);
    int1d_f sum_b("sum_b",n1);
    int1d_f sum_c("sum_c",n1);
    #pragma omp parallel for
    for (int i1=1; i1 <= n1; i1++) {
      int1d_f a("a",n2);
      int1d_f b("b",n2);
      int1d_f c("c",n2);
      auto copy_a = a;
      auto copy_b = b;
      auto copy_c = c;
      for (int i2=1; i2 <= n2; i2++) {
        copy_a(i2) = i1 + 1;
        copy_b(i2) = i1 + 2;
        copy_c(i2) = i1 + i2;
      }
      auto copy_sum_a = sum_a;
      auto copy_sum_b = sum_b;
      auto copy_sum_c = sum_c;
      #ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS
      // Kokkos Threads kernels may only be launched by the master process, not by an OpenMP worker thread.
      size_t local_sum_a = 0;
      size_t local_sum_b = 0;
      size_t local_sum_c = 0;
      for (int i2=1; i2 <= n2; i2++) {
        local_sum_a += a(i2);
        local_sum_b += b(i2);
        local_sum_c += c(i2);
      }
      copy_sum_a(i1) = local_sum_a;
      copy_sum_b(i1) = local_sum_b;
      copy_sum_c(i1) = local_sum_c;
      #else
      copy_sum_a(i1) = yakl::intrinsics::sum(a);
      copy_sum_b(i1) = yakl::intrinsics::sum(b);
      copy_sum_c(i1) = yakl::intrinsics::sum(c);
      #endif
    }
    #pragma omp parallel for
    for (int i1=1; i1 <= n1; i1++) {
      int1d_f a("a",n2);
      int1d_f b("b",n2);
      int1d_f c("c",n2);
      for (int repeat=0; repeat < 48; repeat++) {
        a = int1d_f("a",n2);
        b = int1d_f("b",n2);
        c = int1d_f("c",n2);
      }
    }
    #pragma omp parallel for
    for (int i1=1; i1 <= n1; i1++) {
      std::array<int1d_f,5> copies_a;
      std::array<int1d_f,5> copies_b;
      std::array<int1d_f,5> copies_c;
      for (int copy=0; copy < 5; copy++) {
        copies_a[copy] = sum_a;
        copies_b[copy] = sum_b;
        copies_c[copy] = sum_c;
      }
    }
    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif

  return 0;
}
