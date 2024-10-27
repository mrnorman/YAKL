
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
using yakl::yakl_throw;
using yakl::intrinsics::sum;
using yakl::fence;

// If you define the CPP macro YAKL_ENABLE_STREAMS, then streams will be created and used.
// If you do not define this macro variable, then yakl::create_stream() will return the default stream,
// and everything will run there with no potential for kernel overlap.

int main() {
  yakl::init();
  {
    int n1 = 128;
    int n2 = 1024*1024;
    Array<int,2,memDevice,styleC> a("a",n1,n2);
    Array<int,2,memDevice,styleC> b("b",n1,n2);
    Array<int,2,memHost  ,styleC> d("d",n1,n2);

    // These will automatically call cudaStreamDestroy once they fall out of scope
    auto stream1 = yakl::create_stream();
    auto stream2 = yakl::create_stream();
    auto stream3 = yakl::create_stream();

    // Because operator= cannot take a stream argument, when YAKL_ENABLE_STREAMS is defined, this is surrounded
    // by fence() operations.
    a = 0;
    b = 0;

    // Launch in stream 1
    parallel_for( 128 , YAKL_LAMBDA (int i1) {
      for (int i2=0; i2 < n2; i2++) { a(i1,i2) += 1; }
    } , yakl::DefaultLaunchConfig().set_stream(stream1) );

    // Inform YAKL that stream2 must wait until the previous work in stream1 completes before running future work
    // in stream 2
    stream2.wait_on_event( yakl::record_event(stream1) );

    // Therefore, this kernel will not run until the previous kernel in stream1 completes
    parallel_for( 128 , YAKL_LAMBDA (int i1) {
      for (int i2=0; i2 < n2; i2++) { a(i1,i2) += 2; }
    } , yakl::DefaultLaunchConfig().set_stream(stream2) );

    // This will run in parallel with stream1 and stream2
    parallel_for( 128 , YAKL_LAMBDA (int i1) {
      for (int i2=0; i2 < n2; i2++) { b(i1,i2) += 3; }
    } , yakl::DefaultLaunchConfig().set_stream(stream3) );
    
    // Launch a sum intrinsic in stream2 to ensure the wait_on_event call succeeded
    // All YAKL routines that launch kernels or memory copies will take an optional stream parameter
    auto val1 = static_cast<double>(sum(a,stream2)) / static_cast<double>(n1*n2);
    std::cout << val1 << "\n";
    if ( abs(val1 - 3) >= 1.e-13 ) yakl_throw("ERROR: val1 is wrong");

    // Block on all streams
    fence();

    // This tells YAKL that Array "a" has a dependency on work in stream1. Therefore, once the data pointer for "a"
    // is deallocated (either with explicit deallocation or falling out of scope), at that point, the Array will record
    // events in all streams the Array object is dependent on. The deallocation will not actually happen until all
    // recorded events are completed. This command below is the only reason the array "c" further down does not alias
    // the pointer from array "a", because "a" isn't released from the pool until the kernel below in stream1 completes.
    // If "c" *had* aliased the pointer from "a", a wrong answer would occur because the two kernels below will run at
    // the same time. 
    // 
    // You can add as many stream dependencies as you want.
    // 
    // When the pool allocator is disabled, this is not necessary, and it will essentially no-op
    // 
    // YAKL only actually checks events on which array deallocation is dependent when a new allocation is requested from
    // the pool. Once the pool is destroyed, a yakl::fence() operation is performed, and all allocations dependent on events
    // are freed.
    // 
    // *** IMPORTANT ***
    a.add_stream_dependency(stream1);

    // Launch in stream1
    parallel_for( 128 , YAKL_LAMBDA (int i1) {
      for (int i2=0; i2 < n2; i2++) { a(i1,i2) = b(i1,i2) + 1; }
    } , yakl::DefaultLaunchConfig().set_stream(stream1) );

    // Deallocate "a" and allocate "c". Due to the add_stream_dependency call above, "c" does *not* alias "a"
    a.deallocate();
    Array<int,2,memDevice,styleC> c("c",n1,n2);

    // Launch in stream2 (runs in parallel with stream1)
    parallel_for( 128 , YAKL_LAMBDA (int i1) {
      for (int i2=0; i2 < n2; i2++) { c(i1,i2) = b(i1,i2) + 2; }
    } , yakl::DefaultLaunchConfig().set_stream(stream2) );

    // Copy to host array using stream2
    c.deep_copy_to(d,stream2);

    // Block host code until previous work in stream2 completes
    stream2.fence();

    auto val2 = static_cast<double>(sum(d)) / static_cast<double>(n1*n2);
    std::cout << val2 << "\n";
    if ( abs(val2 - 5) >= 1.e-13 ) yakl_throw("ERROR: val2 is wrong");

    // stream1.fence() would be the more straightforward way to do this, but this demonstrates that you can call
    // fence() on events as well as streams. This will record an event in stream1 and block host code until all previous
    // work in stream1 completes.
    // 
    // Also, there's no real reason why you would need this fence here.
    yakl::record_event(stream1).fence();

  }
  yakl::finalize();
  
  return 0;
}

