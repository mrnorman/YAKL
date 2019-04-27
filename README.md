# YAKL: Yet Another Kernel Launcher
A minimal overhead C++ kernel launcher for performance portability

YAKL is a minimally invasive library intended to allow a user to define kernels in the form of functors (possibly lambdas) without having to marry themselves to a specific data type. This library provides:

* Various launchers such as parallel_for or reduce_[operation], which can run on GPUs using CUDA or CPUs (serial or threaded), and potentially other architectures if they are added
* Wrappers for atomic accesses and synchronization
* Utility functions to extract multi-dimensional indices from a pool of flattened indices
* An optional multi-dimensional array class that uses CUDA Managed Memory if \_\_NVCC\_\_ is defined. Users do not need to use this in order to use the launchers. These are completely orthogonal options.

The user is expected to provide computation in the form of a kernel that acts on a single index in one dimension (already flattened / collapsed in the case of multiple dimensions). This kernel is expressed as a C++ functor and passed to a launcher. All data should be passed by parameter, and the function is expected to have a void return. A simple C++11 parameter pack approach is used to implement this in practice, which avoids the need for meta-templating or using custom data structures.
