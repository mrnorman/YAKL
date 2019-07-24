# YAKL: Yet Another Kernel Launcher
# Still Under Development

YAKL is designed to be similar to Kokkos but significantly simplified. While any datatype can be used with the kernel launcher, there are some important constaints on the objects used. CUDA lambdas requires all captures to be by value because the pointers to CPU data structures would be invalid on the GPU. Therefore, all assignments need to perform a shallow copy of the object, preserving the pointer to the underlying data. The Array.h class can serve as an example for how to do this with constructors, move and copy constructors, and operator= overloading. 

There is a lot of development left to do, but this serves as a basic prototype for now. Currently, the code automatically switches between CPU and CUDA based on whether nvcc is used as the compiler or not. To run the test on the GPU and CPU, use:

```
nvcc --expt-extended-lambda -x cu simpltest.cpp && cuda-memcheck ./a.out

g++ simpltest.cpp && valgrind ./a.out
```

