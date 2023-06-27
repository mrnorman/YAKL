
#include "YAKL.h"

typedef float real;
typedef yakl::Array<real      ,1,yakl::memDevice,yakl::styleC> real1d;
typedef yakl::Array<real const,1,yakl::memDevice,yakl::styleC> realConst1d;

// This routine doesn't change the array argument, so pass it as an array of const data
// YAKL automatically transforms it from non-const to const type, but for this to happen,
//     you must pass it by *value*, not by reference.
// By passing "a" as a realConst1d, we guarantee the routine cannot write to it.
real avg(realConst1d a) {
  return yakl::intrinsics::sum(a) / a.size();
}

int main() {
  yakl::init();
  {
    using yakl::intrinsics::sum; // So we can just say sum(arry) without the yakl::intrinsics::
    int n = 2;
    std::cout << "Allocating new array a\n";
    real1d a("a",n); // Allocate a with n elements
    std::cout << "Setting a to 5\n";
    a = 5;  // Launches kernel to assign "2" to each element of a
    std::cout << "avg(a) = " << avg(a) << "\n";
    std::cout << "Creating Array b that aliases a's data so that changes to one affect the other\n";
    real1d b = a;
    std::cout << "Setting b to 3\n";
    b = 3;
    std::cout << "avg(a) = " << avg(a) << "\n";
    std::cout << "avg(b) = " << avg(b) << "\n";
    // In C, adjacent strings are concatenated
    std::cout << "c = b.createDeviceCopy() creates a new Array with a new "
                 "allocated data buffer and copies b's contents to c."
                 "c has a different data buffer than a and b, so changes to"
                 " c will not affect changes to a or b";
    // "auto" means C++ compiler infers the type from the return type of the RHS. It is real1d
    //     in this case.
    auto c = b.createDeviceCopy();
    std::cout << "avg(b) = " << avg(b) << "\n";
    std::cout << "avg(c) = " << avg(c) << "\n";
    std::cout << "Setting c to 1\n";
    c = 1;
    std::cout << "avg(b) = " << avg(b) << "\n";
    std::cout << "avg(c) = " << avg(c) << "\n";
    std::cout << "c.deep_copy_to(a) will copy the contents of c's data buffer to a's. "
                 "a and c still have distinct data buffers, but the contents are now the same."
                 " Recall that a and b still share the same data buffer so changes to each affect "
                 "the other.\n";
    c.deep_copy_to(a);
    std::cout << "avg(a) = " << avg(a) << "\n";
    std::cout << "avg(b) = " << avg(b) << "\n";
    std::cout << "avg(c) = " << avg(c) << "\n";
    std::cout << "Overwriting variable b to be a newly allocated array. The old Array object stored "
                 "in variable, b, falls out of scope and is replaced by a new Array object that no "
                 "longer shares the data buffer of a.\n";
    b = real1d("b",n);
    std::cout << "Setting b to 7\n";
    b = 7;
    std::cout << "avg(a) = " << avg(a) << "\n";
    std::cout << "avg(b) = " << avg(b) << "\n";
    std::cout << "avg(c) = " << avg(c) << "\n";
  }
  yakl::finalize();
}


