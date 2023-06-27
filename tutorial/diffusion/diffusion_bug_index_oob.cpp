
#include "YAKL.h"

// Declare a generic "real" fp type that can change based on user preference
typedef float real;
// Declare an Array with C-style (zero-based indexing, row-major index ordering) in GPU device memory
//                  datatype   # dimensions    memory space       Fortran or C style?
typedef yakl::Array<real      ,1              ,yakl::memDevice   ,yakl::styleC>        real1d;

int main() {
  // Initialize YAKL with default settings, allocate pool of data for quick array allocations and deallocations
  yakl::init();
  {
    int  nx   = 32;         // Number of grid points
    real xlen = 1.;         // Length of the domain
    real dx   = xlen / nx;  // Grid spacing
    real dt   = 0.25*dx*dx; // Time step size: Max stable dt is dt = 2^-2 * dx^2
    // Create the array to hold the data to solve: d_t(q) = d_xx(q)
    // It doesn't matter for a 1-D array, but with C-style arrays, the last dimension varies the fastest
    //           Label    dimension sizes
    real1d state("state" ,nx             );
    // Initialize the state to zero if you want (not really necessary in this case)
    // The following line actually launches a GPU kernel that assigns each element of "state" to zero
    // For sake of performance, this might not always be what you want to do. For instance, if you wanted to set
    //     multiple variables to zero, you may not want to launch a separate kernel for each variable because
    //     GPU kernels have launch overheads. Insted, you may want to explicitly create a parallel_for kernel
    //     and combine setting each variable to zero inside the same kernel to reduce launch overheads.
    //     But for initializetion and I/O routines, this overhead is likely not that important, and it's 
    //     convenient to be able to just use an equal sign sometimes.
    state = 0;
    // Use C++ "using" statement to place things in the yakl::c:: namespace in local scope so you don't have to
    //     type "yakl::c::"" over and over again. Makes the code a bit prettier.
    // We want things from the "yakl::c" namesapce instead of the "yakl::fortran" namespace because we're using
    //     C-style here.
    // C-style parallel_for and Bounds assume zero-based indexing in the loops.
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    // Launch a kernel to create a square wave data 
    // Anything inside a "parallel_for" is on the GPU. Anything outside of that is on the CPU.
    // You cannot access data in "state" outside a "parallel_for" because the data inside "state" was declared
    //     on the device, not on the host.
    // The following could use either "nx" or "Bounds<1>(nx)" to specify the loop bounds. In C-style, it loops
    //     from 0 --> nx-1. 
    // YAKL_LAMBDA (int i1 [, int i2, ...]) { // code }
    //     This will wrap the code in a C++ class object and pass it to parallel_for to launch using the
    //     appropriate backend.
    //            Kernel label   Loop bounds                loop indices
    parallel_for( "initialize"  ,nx           ,YAKL_LAMBDA (int i       ) {
      // Set the central half of the domain to one for a discontinuous jump from zero to one for initial data
      // I like to use "std::" before math functions like abs, min, max, ceil, sqrt, etc. I find that sometimes
      //     the compiler gives warnings or errors about data types when it's absent.
      if ( std::abs(i-nx/2) <= nx/4 ) {
        state(i) = 1;
      }
    });
    // Keep in mind that all kernels in YAKL are *asynchronous* by default. Therefore, this kernel will go
    //     return to the host *before* the GPU kernel is actually complete. If you want to wait for the
    //     kernel to complete before doing anything else on the host (most often for MPI or I/O), then you
    //     can use yakl::fence() to force the host to wait for the GPU kernel to finish before doing anything
    //     else. This has an overhead, though. Unless you need to wait, it's best not to.

    // Allocate an array to hold the initial state, and copy the data to that array.
    // In YAKL (and Kokkos), an equal sign between two arrays should be thought of as a pointer assignment.
    //     It does not copy the data but rather shares the pointer to the same data. So here, the RHS will
    //     allocate a new Array object and actually copy the contents of "state" into it. Then, the equal
    //     sign will share the data pointer between that temporary array and "state_init". Finally, the
    //     temporary created on the RHS falls out of scope, and state_init takes sole ownership of the
    //     data pointer. The following line is equivalent to the following:
    //       real1d state_init("state",nx);
    //       state.deep_copy_to(state_init);
    //     Unlike Fortran, an equal sign between two Array objects does *not* copy the data itself but rather
    //     shares the same data pointer such. So think of "=" between arrays in YAKL as "=>" in Fortran. To
    //     copy the data itself, you must use "deep_copy_to" or "create[Host|Device]Copy" instead.
    // The C++ "auto" keyword infers the type of the data from the return type of the RHS. This is done at
    //     compile time, so it's still "strongly" typed. In this case, it will be of type "real1d"
    auto state_init = state.createDeviceCopy();

    int num_iterations = 10;  // Number of time steps for diffusion

    // Start a timer so we can profile the cost
    yakl::timer_start("main_loop");

    // The following is a C "for" loop. Don't forget the semi-colons or bad things may happen. Also, zero-
    //     based indexing is the norm.
    for (int iter = 0; iter < num_iterations; iter++) {
      // Create an array to hold cell-edge fluxes
      //////////////////////////////////////////////////////////////////////////////////
      // PURPOSEFUL BUG: flux is declared with nx elements rather than nx+1. This will
      //                 lead to an index out of bounds error. I actually did this by
      //                 accident when developing this code, so it can definitely
      //                 happen.
      //////////////////////////////////////////////////////////////////////////////////
      real1d flux("flux",nx);
      // Launch a GPU kernel to compute fluxes at cell edges
      // Bounds<1>(nx) could just be "nx+1". They are interchangeable for single loops
      parallel_for( "Compute Fluxes" , Bounds<1>(nx+1) , YAKL_LAMBDA (int i) {
        // Compute indices that are periodic
        int ind_im1 = i-1;
        // This is an inline "if" with no curly braces, {}. Only one statement may be put afterward
        if (ind_im1 < 0) ind_im1 += nx;
        int ind_i = i;
        if (ind_i > nx-1) ind_i -= nx;
        flux(i) = -( state(ind_i) - state(ind_im1) ) / dx;
      });
      // Create an array to hold tendencies
      real1d tend("tend",nx);
      // Compute tendencies as flux divergence
      parallel_for( "Compute Tendencies" , nx , YAKL_LAMBDA (int i) {
        tend(i) = -(flux(i+1) - flux(i)) / dx;
      });
      // Apply tedencies (forward Euler)
      parallel_for( "Apply Tendencies" , nx , YAKL_LAMBDA (int i) {
        // The "+=" operator means add tend to the current value in "state(i)" and then overwrite state(i)
        //     with the result. It's identical to: state(i) = state(i) + tend(i);
        state(i) += dt * tend(i);
      });
      // Because of YAKL's pool allocator, creating "flux" and "tend" every time is totally fine.
      // You're free to allocate arrays only where they're needed rather than always creating them
      //     globally and keeping them persistent for all time. This can help reduce memory 
      //     requirements when using local arrays don't need global visibility / scope.
      // After this loop exits, "flux" and "tend" fall out of scope and are automatically deallocated.
    }

    // Stop the timer
    yakl::timer_stop("main_loop");

    // Print the initial and final values to stdout.
    // Even though state_init and state are valid only in GPU device memory, you can still "cout" them.
    // YAKL overloads operator<<, and if the Array is on the device, it'll automatically copy the data
    //     to the host for you and print it out. So feel free to call std::cout on any array in any
    //     memory space. When calling std::cout on a device Array, YAKL inserts a yakl::fence()
    //     to wait for the data to get to the host before printing.
    std::cout << "\n" << state_init << "\n";
    std::cout         << state      << "\n";

    // Check for mass conservation. YAKL has Fortran-like intrinsics like "sum", "minval", "maxloc", etc.
    // The following launches a GPU kernel to compute the sum of the mass.
    real mass_init  = yakl::intrinsics::sum( state_init );
    real mass_final = yakl::intrinsics::sum( state      );
    auto rel_mass_diff = (mass_final - mass_init) / mass_init;
    // Using "std::endl" is helpful because it automatically "flushes" output to stdout or stderr. If
    //     you use "\n", it'll print a newline but not flush the buffer. std::endl will flush.
    std::cout << "Relative Mass Difference: " << rel_mass_diff << "\n" << std::endl;

    // At the end of this block of code (after the "}" below), state_init and state fall out of scope.
    //     Just as in Fortran, once a YAKL Array falls out of scope, it is automatically deallocated.
    //     It's generally quite hard to get a memory leak with standard use of YAKL Arrays.
  }
  // All arrays must be deallocated before calling yakl::finalize() or you'll get a snarky warning
  //     message :)
  yakl::finalize();
}

