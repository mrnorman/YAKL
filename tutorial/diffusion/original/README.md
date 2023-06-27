To start with, please take a look at [`diffusion.cpp`](./diffusion.cpp) to familiarize yourself with the starting code and the comments describing what is going on.

## Regular compilation line

To start with, we'll compile with a vanilla `g++` compiler. You'll need to install the GNU C++ compiler to use this line. On Ubuntu, for instance, it will be `sudo apt-get install g++`. 

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

Here, you'll see output of the array before the simulation, output after the simulation
