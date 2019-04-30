
#include "Array.h"
#include <iostream>
#include "YAKL.h"


class sumKernelClass {
public:
  _YAKL void operator() (ulong i, Array<float> &a, Array<float> &b, Array<float> &c) {
    c(i) = a(i) + b(i);
  }
};


class atomicsKernelCUDAClass : public yakl::Atomics<yakl::targetCUDA> {
public:
  _YAKL void operator() (ulong gInd, Array<float> &a, float &sum, float &min, float &max) {
    unsigned int nx = 128;
    unsigned int ny = 128;
    unsigned int nz = 16;
    unsigned int i, j, k;
    yakl::unpackIndices(gInd,nz,ny,nx,k,j,i);
    atomAdd(sum,a(k,j,i));
    // atomMin(min,a(k,j,i));
    // atomMax(max,a(k,j,i));
  }
};


class atomicsKernelCPUSerialClass : public yakl::Atomics<yakl::targetCPUSerial> {
public:
  _YAKL void operator() (ulong gInd, Array<float> &a, float &sum, float &min, float &max) {
    unsigned int nx = 128;
    unsigned int ny = 128;
    unsigned int nz = 16;
    unsigned int i, j, k;
    yakl::unpackIndices(gInd,nz,ny,nx,k,j,i);
    atomAdd(sum,a(k,j,i));
    // atomMin(min,a(k,j,i));
    // atomMax(max,a(k,j,i));
  }
};


void testLambdaSimple(yakl::Launcher &launcher) {
  int const n = 1024*1024;
  Array<float> a, b, c;
  a.setup(n);
  b.setup(n);
  c.setup(n);
  a = 2;
  b = 3;
  launcher.parallelFor( n , [] _YAKL (unsigned long i, Array<float> &a, Array<float> &b, Array<float> &c ) { c(i) = a(i) + b(i); } , a , b , c );
  launcher.synchronizeSelf();
  if ( (int) c.sum() == 5*1024*1024) {
    std::cout << "testLambdaSimple:  Pass!\n";
  } else {
    std::cout << "testLambdaSimple:  Fail!\n";
  }
}

void testSimpleSum(yakl::Launcher &launcher) {
  int const n = 1024*1024;
  Array<float> a, b, c;
  a.setup(n);
  b.setup(n);
  c.setup(n);
  a = 2;
  b = 3;
  sumKernelClass sumKernel;
  launcher.parallelFor( n , sumKernel , a , b , c );
  launcher.synchronizeSelf();
  if ( (int) c.sum() == 5*1024*1024) {
    std::cout << "testSimpleSum:  Pass!\n";
  } else {
    std::cout << "testSimpleSum:  Fail!\n";
  }
}

void testAtomicsCUDA(yakl::Launcher &launcher) {
  int const nx = 128;
  int const ny = 128;
  int const nz = 16;
  Array<float> a;
  float sum, max, min;
  a.setup(nz,ny,nx);
  for (int k=0; k<nz; k++) {
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        a(k,j,i) = k*ny*nx + j*nx + i;
      }
    }
  }
  sum = 0;
  max = -1.e30;
  min = 1.e30;
  atomicsKernelCUDAClass atomicsKernel;
  launcher.parallelFor( nx*ny*nz , atomicsKernel , a , sum, max, min );
  launcher.synchronizeSelf();
  std::cout << sum << " " << max << " " << min;
}

void testAtomicsCPUSerial(yakl::Launcher &launcher) {
  int const nx = 128;
  int const ny = 128;
  int const nz = 16;
  Array<float> a;
  float sum, max, min;
  a.setup(nz,ny,nx);
  for (int k=0; k<nz; k++) {
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        a(k,j,i) = k*ny*nx + j*nx + i;
      }
    }
  }
  sum = 0;
  max = -1.e30;
  min = 1.e30;
  atomicsKernelCPUSerialClass atomicsKernel;
  launcher.parallelFor( nx*ny*nz , atomicsKernel , a , sum, max, min );
  launcher.synchronizeSelf();
  std::cout << sum << " " << max << " " << min;
}


int main() {
  uint target;
  yakl::Launcher launcher;

  //////////////////////////////////////////////////////////////////////////////////////////
  // CUDA
  //////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "CUDA\n";

  target = yakl::targetCUDA;
  launcher.init(target);

  testLambdaSimple(launcher); 
  testSimpleSum(launcher); 
  testAtomicsCUDA(launcher);

  std::cout << "\n";

  //////////////////////////////////////////////////////////////////////////////////////////
  // CPU Serial
  //////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "CPU Serial\n";

  target = yakl::targetCPUSerial;
  launcher.init(target);

  testLambdaSimple(launcher); 
  testSimpleSum(launcher); 
  testAtomicsCPUSerial(launcher);

  std::cout << "\n";

}



