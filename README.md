# KLaunch
A minimal overhead kernel launcher for performance portability

KLaunch is a minimally invasive library intended to allow a user to define kernels in the form of functors (possibly lambdas) without having to marry themselves to a specific data type. This library provides various launchers such as parallel_for or reduce_[operation], which can run on GPUs using CUDA or CPUs (serial or threaded), and potentially other architectures if they are added. It also provides wrappers for atomic accesses and synchronization.
