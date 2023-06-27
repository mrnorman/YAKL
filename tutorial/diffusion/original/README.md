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

## Automated "printf debugging"

YAKL enables automated printf debugging by dumping out all internal YAKL actions like array creation, `parallel_for` kernel launch, and other issues.

If you only want the master MPI task's output in stdout, you can specify `-DYAKL_VERBOSE`. If you want one file per process to dump all actions for each MPI task, you can use `YAKL_VERBOSE_FILE`. This can make it easy to understand where a stall occurs if it occurs inside code that uses YAKL. Note that you need `-DYAKL_DEBUG` if you want array labels in array operations.


```bash
g++ -DYAKL_DEBUG -DYAKL_VERBOSE_FILE -I../../../src -I../../../src/extensions -I../../../external diffusion.cpp -o diffusion
./diffusion
```

Now, you'll have a bunch of output added to stdout. You'll also notice that you have a file called `yakl_verbose_output_task_0.log`:

```
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "state")
*** [YAKL_VERBOSE] Launching parallel_for labeled "YAKL_internal_memset" with 32 threads (label: "YAKL_internal_memset")
*** [YAKL_VERBOSE] Launching parallel_for labeled "initialize" with 32 threads (label: "initialize")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "state")
*** [YAKL_VERBOSE] Initiating device to device memcpy of 128 bytes from Array labeled "state" to Array labeled "state"
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 132 bytes (label: "flux")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Fluxes" with 33 threads (label: "Compute Fluxes")
*** [YAKL_VERBOSE] Allocating device, C-style, rank 1 Array of size 128 bytes (label: "tend")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Compute Tendencies" with 32 threads (label: "Compute Tendencies")
*** [YAKL_VERBOSE] Launching parallel_for labeled "Apply Tendencies" with 32 threads (label: "Apply Tendencies")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "tend")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "flux")
*** [YAKL_VERBOSE] Launching device reduction
*** [YAKL_VERBOSE] Launching device reduction
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "state")
*** [YAKL_VERBOSE] Deallocating device, C-style, rank 1 Array (label: "state")
*** [YAKL_VERBOSE] Destroying pool (label: "Gator: YAKL's primary memory pool")
```

If you were to have an error (segmentation fault, out-of-bounds index error, bus error, etc), the last output you see gives you information about where that error could have occurred. This is another reason it's important to provide meaningful labels to your arrays and `parallel_for` launches.

## Remove the pool allocator

The YAKL pool allocator makes frequent allocation and deallocation faster, **especially on GPU devices**. But even on the host, it typically makes a difference. Change `nx` to `1024*1024` in the code (you may want to comment out the "std::cout << state" lines as well). Then run with and without the pool allocator. The pool allocator can be disable with the shell environment variable `GATOR_DISABLE=1`.

```bash
g++ -DYAKL_PROFILE -O3 -I../../../src -I../../../src/extensions -I../../../external diffusion.cpp -o diffusion
# With pool allocator
./diffusion
# Without pool allocator (no need to recompile)
GATOR_DISABLE=1 ./diffusion
```

With the pool allocator, I got 3.058049e-02 seconds. Without the pool allocator, I got 6.319616e-02 seconds. So even on the host, the pool allocator can make a difference. 

## DEBUGGGING 1: Array creation with wrong number of dimensions

Now we'll go through some debugging exercises to get you used to the kinds of errors you'll encounter from time to time.

Here, we'll compile a file that creates an array with the wrong number of dimensions. We'll need `-DYAKL_DEBUG -g` in the compile flags from here on out to ensure we catch errors and have debug symbols.

```bash
g++ -DYAKL_DEBUG -g -I../../../src -I../../../src/extensions -I../../../external diffusion_bug_creation_dimension.cpp -o diffusion
```

Here, you should get a compile-time error as follows:

```
In file included from ../../../src/YAKL_Array.h:347,
                 from ../../../src/YAKL.h:68,
                 from diffusion_bug_creation_dimension.cpp:2:
../../../src/YAKL_CArray.h: In instantiation of ‘yakl::Array<T, rank, myMem, 1>::Array(const char*, yakl::index_t, yakl::index_t) [with T = float; int rank = 1; int myMem = 1; yakl::index_t = unsigned int]’:
diffusion_bug_creation_dimension.cpp:90:32:   required from here
../../../src/YAKL_CArray.h:126:27: error: static assertion failed: ERROR: Calling constructor with 2 bound on non-rank-2 array
  126 |       static_assert( rank == 2 , "ERROR: Calling constructor with 2 bound on non-rank-2 array" );
      |                      ~~~~~^~~~
../../../src/YAKL_CArray.h:126:27: note: ‘(1 == 2)’ evaluates to false
```

When you create an array, the number of dimensions you provide must equal the rank of the array you're declaring. We declared the array to have a rank of 1, but we provided two dimensions when creating it, leading to a compile-time error.

## DEBUGGGING 2: Array indexing with wrong number of dimensions

Here, we'll compile a file that indexes an array with the wrong number of dimensions.

```bash
g++ -DYAKL_DEBUG -g -I../../../src -I../../../src/extensions -I../../../external diffusion_bug_indexing_dimension.cpp -o diffusion
```

Here, you should get a compile-time error as follows:
```
In file included from ../../../src/YAKL_Array.h:347,
                 from ../../../src/YAKL.h:68,
                 from diffusion_bug_indexing_dimension.cpp:2:
../../../src/YAKL_CArray.h: In instantiation of ‘T& yakl::Array<T, rank, myMem, 1>::operator()(yakl::index_t, yakl::index_t) const [with T = float; int rank = 1; int myMem = 1; yakl::index_t = unsigned int]’:
diffusion_bug_indexing_dimension.cpp:100:27:   required from here
../../../src/YAKL_CArray.h:482:27: error: static assertion failed: ERROR: Indexing non-rank-2 array with 2 indices
  482 |       static_assert( rank == 2 , "ERROR: Indexing non-rank-2 array with 2 indices" );
      |                      ~~~~~^~~~
../../../src/YAKL_CArray.h:482:27: note: ‘(1 == 2)’ evaluates to false
```

When indexing an array, the number of indices must always match the rank you declared the array as. Note that Fortran compilers do not always enforce this. But YAKL always does. The rank of an array can never changes. You can call the `reshape()` member function of an Array object to create a new Array object with a different rank / shape that points to the same data. But that's creating a new Array object pointing to the same data, not changing the rank of an Array object.

## DEBUGGING 3: Array index out of bounds

Now we're going to index the flux array out of bounds on purpose by declaring flux with `nx` elements instead of `nx+1`. C++ doesn't allow this to be caught at compile time, so this will have to be caught at runtime instead.

```bash
g++ -DYAKL_DEBUG -g -I../../../src -I../../../src/extensions -I../../../external diffusion_bug_index_oob.cpp -o diffusion
./diffusion
```

You should get an error as follows:

```
Using memory pool. Initial size: 0.704102GB ;  Grow size: 0.704102GB.
INFORM: Automatically inserting fence() after every parallel_for
ERROR: For Array labeled: flux:
Index 1 of 1 is out of bounds.  Provided index: 32.  Upper Bound: 31
YAKL FATAL ERROR:
ERROR: Index out of bounds.
terminate called after throwing an instance of 'std::runtime_error'
  what():  ERROR: Index out of bounds.
Aborted (core dumped)
```

To find out the line where this failed, you can use `gdb`. You'll use `gdb [./executable]`, then type `run`. When it fails, type `bt` to get a backtrace of the error with the file and line number it occurred at.I always recommend running on the host for debugging first to catch these type of errors. You get more information on the host than you do on the GPU.

If you only have a core file, you can use `gdb [/path/to/executable] [/path/to/core]` and use `bt` to get a backgrace from there as well.

```
[imn@imn-virtual-machine:~/YAKL/tutorial/diffusion/original] 8-) gdb ./diffusion 
GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 12.1
Copyright (C) 2022 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./diffusion...
(gdb) run
Starting program: /home/oem/YAKL/tutorial/diffusion/original/diffusion 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Using memory pool. Initial size: 0.704102GB ;  Grow size: 0.704102GB.
INFORM: Automatically inserting fence() after every parallel_for
ERROR: For Array labeled: flux:
Index 1 of 1 is out of bounds.  Provided index: 32.  Upper Bound: 31
YAKL FATAL ERROR:
ERROR: Index out of bounds.
terminate called after throwing an instance of 'std::runtime_error'
  what():  ERROR: Index out of bounds.

Program received signal SIGABRT, Aborted.
__pthread_kill_implementation (no_tid=0, signo=6, threadid=140737352688576) at ./nptl/pthread_kill.c:44
44	./nptl/pthread_kill.c: No such file or directory.
(gdb) bt
#0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=140737352688576) at ./nptl/pthread_kill.c:44
#1  __pthread_kill_internal (signo=6, threadid=140737352688576) at ./nptl/pthread_kill.c:78
#2  __GI___pthread_kill (threadid=140737352688576, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
#3  0x00007ffff7842476 in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
#4  0x00007ffff78287f3 in __GI_abort () at ./stdlib/abort.c:79
#5  0x00007ffff7ca2bbe in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
#6  0x00007ffff7cae24c in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
#7  0x00007ffff7cae2b7 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
#8  0x00007ffff7cae518 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#9  0x0000555555559511 in yakl::yakl_throw (msg=0x5555555722be "ERROR: Index out of bounds.") at ../../../src/YAKL_error.h:19
#10 0x0000555555569309 in yakl::Array<float, 1, 1, 1>::ind_out_bounds<0> (this=0x7fffffffdcf8, ind=32) at ../../../src/YAKL_CArray.h:611
#11 0x0000555555565754 in yakl::Array<float, 1, 1, 1>::check (this=0x7fffffffdcf8, i0=32, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0) at ../../../src/YAKL_CArray.h:578
#12 0x0000555555561dd4 in yakl::Array<float, 1, 1, 1>::operator() (this=0x7fffffffdcf8, i0=32) at ../../../src/YAKL_CArray.h:474
#13 0x0000555555557a51 in operator() (__closure=0x7fffffffdcf0, i=32) at diffusion_bug_index_oob.cpp:103
#14 0x0000555555558b35 in yakl::c::parallel_for_cpu_serial<main()::<lambda(int)>, false, 1>(const yakl::c::Bounds<1, false> &, const struct {...} &, yakl::c::DoOuter<false>) (bounds=..., f=..., dummy=...)
    at ../../../src/YAKL_parallel_for_common.h:338
#15 0x0000555555558637 in yakl::c::parallel_for<main()::<lambda(int)>, 1, false>(const char *, const yakl::c::Bounds<1, false> &, const struct {...} &, yakl::LaunchConfig<128, false>) (
    str=0x555555571f7c "Compute Fluxes", bounds=..., f=..., config=...) at ../../../src/YAKL_parallel_for_common.h:536
#16 0x0000555555557f3f in main () at diffusion_bug_index_oob.cpp:96
(gdb) quit
A debugging session is active.

	Inferior 1 [process 29289] will be killed.

Quit anyway? (y or n) y
[imn@imn-virtual-machine:~/YAKL/tutorial/diffusion/original] 8-) 
```

From this, we can see that the error occurs at line 103 of `diffusion_bug_index_oob.cpp`. You'll see the entire stack and a lot of other output. My recommendation is to go from top to bottom in the output you get for the stack trace. The moment you no longer see `YAKL_` in the filename, you're now in **your own** code.

## DEBUGGING 4: Using uninitialized memory

One of the most nefarious, sneaky, and frustrating bugs you'll encounter is using uninitialized memory, leading to undefined code behavior and bugs the present at random places at random times. It can feel like a ghost in the machine. Here, we'll get introduced to the `valgrind` tool that detects situations like this. It also detects invalid memory address errors, but you likely won't find those often because of YAKL's index checking capabilities. Here, we'll simply delete the line that initialized the state to zero.

```
g++ -DYAKL_DEBUG -g -I../../../src -I../../../src/extensions -I../../../external diffusion_bug_read_uninitialized_memory.cpp -o diffusion
./diffusion
```

The thing is that you will likely be able to run this with no errors and even get the expected output because it's possible that the initial data is zeros already. But this **will** come back to bite you and probably at scale where it's nearly impossible to reproduce reliably. If we run this through `valgrind`, we'll find a different story and a **ton** of errors. Unfortunately, `valgrind` is very verbose, and not always that easy to interpret. But running cleanly without warnings or errors through `valgrind` is something that we should ensure for every code inside E3SM. It will save us money, time, and computing allocations.

valgrind is **very, very** slow to run, so please use the absolute smallest problem size possible. Also, you must turn off the pool allocator to properly identify memory bugs. The pool allocator will mask issues. This can be done with `export GATOR_DISABLE=1` or `setenv GATOR_DISABLE 1` depending on your shell.

```
export GATOR_DISABLE=1
valgrind ./diffusion >& valgrind_output.txt
```

In this, we'll see errors of the following form:

```
==29652== Conditional jump or move depends on uninitialised value(s)
==29652==    at 0x4B14D08: __printf_fp_l (printf_fp.c:396)
==29652==    by 0x4B309AC: __printf_fp_spec (vfprintf-internal.c:354)
==29652==    by 0x4B309AC: __vfprintf_internal (vfprintf-internal.c:1558)
==29652==    by 0x4B42519: __vsnprintf_internal (vsnprintf.c:114)
==29652==    by 0x496B1FF: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==29652==    by 0x499DBD2: std::ostreambuf_iterator<char, std::char_traits<char> > std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::_M_insert_float<double>(std::ostreambuf_iterat    or<char, std::char_traits<char> >, std::ios_base&, char, char, double) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==29652==    by 0x49ADCFD: std::ostream& std::ostream::_M_insert<double>(double) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==29652==    by 0x1161CB: yakl::operator<<(std::ostream&, yakl::Array<float, 1, 1, 1> const&) (YAKL_ArrayBase.h:364)
==29652==    by 0x10C133: main (diffusion_bug_read_uninitialized_memory.cpp:132)

...

==29652== Use of uninitialised value of size 8
==29652==    at 0x4B150DE: __printf_fp_l (printf_fp.c:438)
==29652==    by 0x4B309AC: __printf_fp_spec (vfprintf-internal.c:354)
==29652==    by 0x4B309AC: __vfprintf_internal (vfprintf-internal.c:1558)
==29652==    by 0x4B42519: __vsnprintf_internal (vsnprintf.c:114)
==29652==    by 0x496B1FF: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==29652==    by 0x499DBD2: std::ostreambuf_iterator<char, std::char_traits<char> > std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::_M_insert_float<double>(std::ostreambuf_iterat    or<char, std::char_traits<char> >, std::ios_base&, char, char, double) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==29652==    by 0x49ADCFD: std::ostream& std::ostream::_M_insert<double>(double) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==29652==    by 0x1161CB: yakl::operator<<(std::ostream&, yakl::Array<float, 1, 1, 1> const&) (YAKL_ArrayBase.h:364)
==29652==    by 0x10C161: main (diffusion_bug_read_uninitialized_memory.cpp:133)
```

`valgrind` has an interesting M.O. You often don't get a warning or error just by reading from uninitialized data, but once it changes program behavior (e.g., if-then-else branching), then you get the warning. So it can be tough at times to sleuth out the original problem and fix it. Here, we see program behavior changing due to uninitialized values at line 132, which is this line:
```
    std::cout << "\n" << state_init << "\n";
```
So, the entire simulation ran without valgrind complaining, but once we dump the data out, valgrind then complains. Again, this can be hard to sleuth out. So we'll add an option to valgrind once we find an error that traces the initial allocation of the offending data that's uninitialized. Warning this tracking increases the time valgrind takes substantially, so only use it once you've found an error you cannot figure out how to fix.

```
valgrind --track-origins=yes ./diffusion >& valgrind_output.txt
```

Now, we get the output:

```
==30013== Conditional jump or move depends on uninitialised value(s)
==30013==    at 0x4B14D08: __printf_fp_l (printf_fp.c:396)
==30013==    by 0x4B309AC: __printf_fp_spec (vfprintf-internal.c:354)
==30013==    by 0x4B309AC: __vfprintf_internal (vfprintf-internal.c:1558)
==30013==    by 0x4B42519: __vsnprintf_internal (vsnprintf.c:114)
==30013==    by 0x496B1FF: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==30013==    by 0x499DBD2: std::ostreambuf_iterator<char, std::char_traits<char> > std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::_M_insert_float<double>(std::ostreambuf_itera     tor<char, std::char_traits<char> >, std::ios_base&, char, char, double) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==30013==    by 0x49ADCFD: std::ostream& std::ostream::_M_insert<double>(double) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==30013==    by 0x1161CB: yakl::operator<<(std::ostream&, yakl::Array<float, 1, 1, 1> const&) (YAKL_ArrayBase.h:364)
==30013==    by 0x10C133: main (diffusion_bug_read_uninitialized_memory.cpp:132)
==30013==  Uninitialised value was created by a heap allocation
==30013==    at 0x4848899: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==30013==    by 0x110DA7: yakl::set_device_alloc_free(std::function<void* (unsigned long)>&, std::function<void (void*)>&)::{lambda(unsigned long)#1}::operator()(unsigned long) const (YAKL_allocators.h:134)
==30013==    by 0x12122B: void* std::__invoke_impl<void*, yakl::set_device_alloc_free(std::function<void* (unsigned long)>&, std::function<void (void*)>&)::{lambda(unsigned long)#1}&, unsigned long>(std::__     invoke_other, yakl::set_device_alloc_free(std::function<void* (unsigned long)>&, std::function<void (void*)>&)::{lambda(unsigned long)#1}&, unsigned long&&) (invoke.h:61)
==30013==    by 0x11EAA8: std::enable_if<is_invocable_r_v<void*, yakl::set_device_alloc_free(std::function<void* (unsigned long)>&, std::function<void (void*)>&)::{lambda(unsigned long)#1}&, unsigned long>,      void*>::type std::__invoke_r<void*, yakl::set_device_alloc_free(std::function<void* (unsigned long)>&, std::function<void (void*)>&)::{lambda(unsigned long)#1}&, unsigned long>(yakl::set_device_alloc_free(     std::function<void* (unsigned long)>&, std::function<void (void*)>&)::{lambda(unsigned long)#1}&, unsigned long&&) (invoke.h:114)
==30013==    by 0x11BB9C: std::_Function_handler<void* (unsigned long), yakl::set_device_alloc_free(std::function<void* (unsigned long)>&, std::function<void (void*)>&)::{lambda(unsigned long)#1}>::_M_invok     e(std::_Any_data const&, unsigned long&&) (std_function.h:290)
==30013==    by 0x112B0E: std::function<void* (unsigned long)>::operator()(unsigned long) const (std_function.h:590)
==30013==    by 0x110EEA: yakl::set_yakl_allocators_to_default()::{lambda(unsigned long, char const*)#3}::operator()(unsigned long, char const*) const (YAKL_allocators.h:178)
==30013==    by 0x12166F: void* std::__invoke_impl<void*, yakl::set_yakl_allocators_to_default()::{lambda(unsigned long, char const*)#3}&, unsigned long, char const*>(std::__invoke_other, yakl::set_yakl_all     ocators_to_default()::{lambda(unsigned long, char const*)#3}&, unsigned long&&, char const*&&) (invoke.h:61)
==30013==    by 0x11F0FE: std::enable_if<is_invocable_r_v<void*, yakl::set_yakl_allocators_to_default()::{lambda(unsigned long, char const*)#3}&, unsigned long, char const*>, void*>::type std::__invoke_r<vo     id*, yakl::set_yakl_allocators_to_default()::{lambda(unsigned long, char const*)#3}&, unsigned long, char const*>(yakl::set_yakl_allocators_to_default()::{lambda(unsigned long, char const*)#3}&, unsigned lo     ng&&, char const*&&) (invoke.h:114)
==30013==    by 0x11C00C: std::_Function_handler<void* (unsigned long, char const*), yakl::set_yakl_allocators_to_default()::{lambda(unsigned long, char const*)#3}>::_M_invoke(std::_Any_data const&, unsigne     d long&&, char const*&&) (std_function.h:290)
==30013==    by 0x1150A6: std::function<void* (unsigned long, char const*)>::operator()(unsigned long, char const*) const (std_function.h:590)
==30013==    by 0x111137: yakl::alloc_device(unsigned long, char const*) (YAKL_allocators.h:234)
``

So, here we see the uninitialized data was created "by a heap allocation", which means "malloc" (the C equivalent of Fortran's `allocate()` statement). So this clues us in that this is a YAKL Array. We see it's created in line 12. In this case, we never really see the line it's allocated. It's buried quite throughly in a bunch of C++ gobbledygook. But at least from the line it occurred, we know it has to do with state_init, which we know came from state, and we can then trace the initializeation of state to see how and where it happened. It's worth persevering with getting valgrind to run clean on your code.

It's worth noting that MPI and I/O libraries often throw false warnings through valgrind, so please keep that in mind. I regularly get valgrind errors MPI routines that are of no fault of my codes.

## DEBUGGING 5: Using an Array that isn't allocated

Here, we will "forget" to allocate a variable and then try to index it. You'll get a thrown exception, and you can track the line with `gdb` just as before:

```
>  g++ -g -DYAKL_DEBUG -I../../../src -I../../../src/extensions -I../../../external diffusion_bug_not_allocated.cpp -o diffusion
>  ./diffusion
INFORM: Automatically inserting fence() after every parallel_for
YAKL FATAL ERROR:
Error: Using operator() on an Array that isn't allocated
terminate called after throwing an instance of 'std::runtime_error'
  what():  Error: Using operator() on an Array that isn't allocated
Aborted (core dumped)
```



