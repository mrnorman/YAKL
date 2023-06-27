## Diffusion YAKL Tutorial

This tutorial implements a diffusion simulation in 1-D using a mass conserving Finite-Volume approach to integrate the PDE:

### $\frac{\partial q}{\partial t} = \frac{\partial^2 q}{\partial^2 x}$

You can go through this tutorial hands-on on your laptop compiling to run on CPUs. If you have a GPU on your laptop, you can also compile and run on the GPU.

## Workflow
To start with, you need to clone this tutorial and `cd` into the correct directory:

```bash
git clone https://github.com/mrnorman/YAKL.git
cd YAKL/tutorial/diffusion
```

There are in-line comments in [`YAKL/tutorial/diffusion/original`](./original/diffusion.cpp) to guide you as to what's going on in the code. There are several things covered in this tutorial, and you can work through things in this order:
* [Basic compilation](./original)
