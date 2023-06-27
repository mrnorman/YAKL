To start with, please take a look at [`diffusion.cpp`](./diffusion.cpp) to familiarize yourself with the starting code and the comments describing what is going on.

## Regular compilation line

To start with, we'll compile with a vanilla `g++` compiler. You'll need to install the GNU C++ compiler to use this line. On Ubuntu, for instance, it will be `sudo apt-get install g++`. This will compile for a serial CPU target such that `parallel_for` kernels execute as simple serial for loops.

```bash
cd original
g++ -I../../../src -I../../../src/extensions -I../../../external diffusion.cpp -o diffusion
./diffusion
```

This should lead to output similar to the following:

```
Using memory pool. Initial size: 0.704102GB ;  Grow size: 0.704102GB.

For Array labeled: "Unlabeled: YAKL_DEBUG CPP macro not defined"
Number of Dimensions: 1
Total Number of Elements: 32
Dimension Sizes: 32, 
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 

For Array labeled: "Unlabeled: YAKL_DEBUG CPP macro not defined"
Number of Dimensions: 1
Total Number of Elements: 32
Dimension Sizes: 32, 
0.000402451 0.00130844 0.00590992 0.0206947 0.0576591 0.131588 0.251722 0.411901 0.588099 0.748278 0.868412 0.942341 0.979305 0.994091 0.998712 0.999798 0.99996 0.999798 0.998712 0.994091 0.979305 0.942341 0.868412 0.748278 0.588099 0.411901 0.251722 0.131588 0.0576591 0.0206947 0.00590992 0.00130844 

Relative Mass Difference: 0

Pool Memory High Water Mark:       640
Pool Memory High Water Efficiency: 8.46535e-07
```

The first line shows that the pool allocator is initializing. You'll likely have a different amount of memory allocated at the beginning. YAKL automatically allocates one eigth of the total system memory for the initial memory pool. So you can see that my system contains 5.6 GB of memory. If that seems odd, it is. I'm using a virtual machine at the moment. 

The second output is the information and data of the initial state array. The next output is the information and data of the final state array after diffusion has been performed. The output after that is the relative mass difference that should be machine precision (order `1.e-7` for single precision and order `1.e-14` for double precision). Finally, YAKL gives "high water mark" memory information to let you know how many bytes you used maximally and the proportion that was of the total pool size.

## Turning on Debugging to get array labels

You'll notice that the Array labels are "Unlabeled...". To enable storing and printing Array labels, we need to add `-DYAKL_DEBUG` to the compile line:

```bash
g++ -DYAKL_DEBUG -I../../../src -I../../../src/extensions -I../../../external diffusion.cpp -o diffusion
./diffusion
```

This time, you should see actual labels for the arrays when they are printed out.

## Turning on profiling

`diffusion.cpp` contains `yakl::timer_start` and `yakl::timer_stop` calls, but you'll notice there's no timer output. To enable this, we need to add `-DYAKL_PROFILE` so that timers are actually used.

```bash
g++ -DYAKL_PROFILE -I../../../src -I../../../src/extensions -I../../../external diffusion.cpp -o diffusion
./diffusion
```

This time, you'll see timer output at the end of the stdout:

```
******* Timers for thread 0 *******
________________________________________________________________________________________________________
Timer label                                       # calls     Total time     Min time       Max time
________________________________________________________________________________________________________
main_loop                                         1           8.137500e-05   8.137500e-05   8.137500e-05
________________________________________________________________________________________________________
The ~ character beginning a timer label indicates it has multiple parent timers.
Thus, those timers will likely not accumulate like you expect them to.
```

YAKL's internal timers behave similar to General Purpose Timer Library (GPTL) timers.

## Turning on YAKL's automated profiling

YAKL also has the ability to put timers around all `parallel_for` and data copy routines automatically. This is why it's important that you label your `parallel_for` calls so that you can tell quickly which ones are the most expensive when checking performance. To do this we'll specify `-DYAKL_AUTO_PROFILE` in the compile flags:


```bash
g++ -DYAKL_AUTO_PROFILE -I../../../src -I../../../src/extensions -I../../../external diffusion.cpp -o diffusion
./diffusion
```

Now, you'll see much more detailed timer output for every incremental device work that is called:
```
******* Timers for thread 0 *******
________________________________________________________________________________________________________
Timer label                                       # calls     Total time     Min time       Max time       
________________________________________________________________________________________________________
YAKL_internal_memset                              1           2.480000e-06   2.480000e-06   2.480000e-06   
initialize                                        1           1.285000e-06   1.285000e-06   1.285000e-06   
YAKL_internal_memcpy_device_to_device             1           6.210000e-07   6.210000e-07   6.210000e-07   
main_loop                                         1           1.240620e-04   1.240620e-04   1.240620e-04   
  Compute Fluxes                                  10          1.478800e-05   1.353000e-06   1.814000e-06   
  Compute Tendencies                              10          1.350400e-05   1.301000e-06   1.531000e-06   
  Apply Tendencies                                10          1.176000e-05   1.110000e-06   1.262000e-06   
YAKL_internal_memcpy_device_to_host               2           1.245000e-06   4.780000e-07   7.670000e-07   
________________________________________________________________________________________________________
The ~ character beginning a timer label indicates it has multiple parent timers.
Thus, those timers will likely not accumulate like you expect them to.
```

You can see nested timers for parent and child timers, how many times something was called, how long memory copies between host and device (just copies between host and host in this case since we're not on the GPU yet), and how long the kernel took that set `state` to zero. Further, like GPTL, the timers appear in the order they are first called to give you a sense of the program flow.
