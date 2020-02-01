
#ifndef _YAKL_H_
#define _YAKL_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include "BuddyAllocator.h"

#ifdef __USE_CUDA__
  #define YAKL_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #include <cub/cub.cuh>
#elif defined(__USE_HIP__)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#else
  #define YAKL_LAMBDA [&]
  #define YAKL_INLINE inline
#endif

#ifdef _OPENMP45
#include <omp.h>
#endif

#ifdef _OPENACC
#include "openacc.h"
#endif


namespace yakl {

  // Memory space specifiers for YAKL Arrays
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;

  int constexpr COLON = std::numeric_limits<int>::min();
  int constexpr NOSPEC = std::numeric_limits<int>::min()+1;


  // Size of the buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
  int constexpr functorBufSize = 1024*128;
  // Buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
  extern void *functorBuffer;


  // Pool allocator object
  extern BuddyAllocator pool;

  // YAKL allocator and deallocator
  extern std::function<void *( size_t )> yaklAllocDevice;
  extern std::function<void ( void * )>  yaklFreeDevice;

  // YAKL allocator and deallocator
  extern std::function<void *( size_t )> yaklAllocHost;
  extern std::function<void ( void * )>  yaklFreeHost;



  template <int L, int U> class bnd {
  public:
    bnd() = delete;
    static constexpr int l() { return L; }
    static constexpr int u() { return U; }
  };



  // Block the CPU code until the device code and data transfers are all completed
  inline void fence() {
    #ifdef __USE_CUDA__
      cudaDeviceSynchronize();
    #endif
    #ifdef __USE_HIP__
      hipDeviceSynchronize();
    #endif
  }



  // Initialize the YAKL framework
  inline void init( size_t poolBytes = 0 ) {

    std::function<void *( size_t )> alloc;
    std::function<void( void * )>   dealloc;

    #if   defined(__USE_CUDA__)
      #if defined (__MANAGED__)
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMallocManaged(&ptr,bytes);
          cudaMemPrefetchAsync(ptr,bytes,0);
          #ifdef _OPENMP45
            omp_target_associate_ptr(ptr,ptr,bytes,0,0);
          #endif
          #ifdef _OPENACC
            acc_map_data(ptr,ptr,bytes);
          #endif
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
        };
      #else
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMalloc(&ptr,bytes);
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
        };
      #endif
    #elif defined(__USE_HIP__)
      #if defined (__MANAGED__)
        alloc   = [] ( size_t bytes ) -> void* { void *ptr; hipMallocHost(&ptr,bytes); return ptr; };
        dealloc = [] ( void *ptr    )          { hipFree(ptr); };
      #else
        alloc   = [] ( size_t bytes ) -> void* { void *ptr; hipMalloc(&ptr,bytes); return ptr; };
        dealloc = [] ( void *ptr    )          { hipFree(ptr); };
      #endif
    #else
      alloc   = ::malloc;
      dealloc = ::free;
    #endif

    // If bytes are specified, then initialize a pool allocator
    if ( poolBytes > 0 ) {
      std::cout << "Initializing the YAKL Pool Allocator with " << poolBytes << " bytes" << std::endl;

      pool = BuddyAllocator( poolBytes , 1024 , alloc , dealloc );

      yaklAllocDevice = [] (size_t bytes) -> void * { return pool.allocate( bytes ); };
      yaklFreeDevice  = [] (void *ptr)              { pool.free( ptr );              };

    } else { // poolBytes < 0
      std::cout << "Not using the YAKL Pool Allocator" << std::endl;

      yaklAllocDevice = alloc;
      yaklFreeDevice  = dealloc;

    } // poolBytes

    yaklAllocHost = [] (size_t bytes) -> void * { return malloc(bytes); };
    yaklFreeHost  = [] (void *ptr) { free(ptr); };

    #if defined(__USE_CUDA__)
      cudaMalloc(&functorBuffer,functorBufSize);
    #endif

    #if defined(__USE_HIP__)
      int id;
      hipGetDevice(&id);
      hipDeviceProp_t props;
      hipGetDeviceProperties(&props,id);
      std::cout << props.name << std::endl;
    #endif

    #if defined(__AUTO_FENCE__)
      std::cout << "WARNING: Automatically inserting fence() after every parallel_for" << std::endl;
    #endif

  } // 



  inline void finalize() {
    pool = BuddyAllocator();
    #if defined(__USE_CUDA__)
      cudaFree(functorBuffer);
    #endif
  }



  // Unpack 2D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int &i1, int &i2) {
    i1 = (iGlob/(n2))     ;
    i2 = (iGlob     ) % n2;
  }
  // Unpack 3D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int &i1, int &i2, int &i3) {
    i1 = (iGlob/(n3*n2))     ;
    i2 = (iGlob/(n3   )) % n2;
    i3 = (iGlob        ) % n3;
  }
  // Unpack 4D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int &i1, int &i2, int &i3, int &i4) {
    i1 = (iGlob/(n4*n3*n2))     ;
    i2 = (iGlob/(n4*n3   )) % n2;
    i3 = (iGlob/(n4      )) % n3;
    i4 = (iGlob           ) % n4;
  }
  // Unpack 5D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int &i1, int &i2, int &i3, int &i4, int &i5) {
    i1 = (iGlob/(n5*n4*n3*n2))     ;
    i2 = (iGlob/(n5*n4*n3   )) % n2;
    i3 = (iGlob/(n5*n4      )) % n3;
    i4 = (iGlob/(n5         )) % n4;
    i5 = (iGlob              ) % n5;
  }
  // Unpack 6D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6) {
    i1 = (iGlob/(n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n6*n5*n4      )) % n3;
    i4 = (iGlob/(n6*n5         )) % n4;
    i5 = (iGlob/(n6            )) % n5;
    i6 = (iGlob                 ) % n6;
  }
  // Unpack 7D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int n7, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7) {
    i1 = (iGlob/(n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n7*n6*n5         )) % n4;
    i5 = (iGlob/(n7*n6            )) % n5;
    i6 = (iGlob/(n7               )) % n6;
    i7 = (iGlob                    ) % n7;
  }
  // Unpack 8D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7, int &i8) {
    i1 = (iGlob/(n8*n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n8*n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n8*n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n8*n7*n6*n5         )) % n4;
    i5 = (iGlob/(n8*n7*n6            )) % n5;
    i6 = (iGlob/(n8*n7               )) % n6;
    i7 = (iGlob/(n8                  )) % n7;
    i8 = (iGlob                       ) % n8;
  }



  #ifdef __USE_CUDA__
    template <class F> __global__ void cudaKernelVal( int n1 , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }

    template <class F> __global__ void cudaKernelRef( int n1 , F const &f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }

    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , F const &f , int vectorSize = 128 ) {
      cudaKernelVal <<< (unsigned int) (n1-1)/vectorSize+1 , vectorSize >>> ( n1 , f );
    }

    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , F const &f , int vectorSize = 128 ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      cudaKernelRef <<< (unsigned int) (n1-1)/vectorSize+1 , vectorSize >>> ( n1 , *fp );
    }
  #endif



  #ifdef __USE_HIP__
    template <class F> __global__ void hipKernel( int n1 , F f ) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }

    template<class F> inline void parallel_for_hip( int n1 , F const &f , int vectorSize = 128 ) {
      hipLaunchKernelGGL( hipKernel , dim3((n1-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , f );
    }
  #endif



  template <class F> inline void parallel_for_cpu_serial( int n1 , F const &f ) {
    for (int i=0; i<n1; i++) {
      f(i);
    }
  }



  template <class F> inline void parallel_for( int n1 , F const &f , int vectorSize = 128 ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , f , vectorSize );
    #elif defined(__USE_HIP__)
      parallel_for_hip ( n1 , f , vectorSize );
    #else
      parallel_for_cpu_serial( n1 , f );
    #endif

    #if defined(__AUTO_FENCE__)
      fence();
    #endif
  }



  template <class F> inline void parallel_for( char const * str , int n1 , F const &f , int vectorSize = 128 ) {
    parallel_for( n1 , f , vectorSize );
  }



  template <class T, int myMem> class ParallelMin;
  template <class T, int myMem> class ParallelMax;
  template <class T, int myMem> class ParallelSum;

  #ifdef __USE_HIP__

    template <class T> class ParallelMin<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMin() { tmp = NULL; }
      ParallelMin(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelMin() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        hipcub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        tmp = yaklAllocDevice(nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP);
          yaklFreeDevice(tmp);
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        hipcub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        hipcub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };

    template <class T> class ParallelMax<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMax() { tmp = NULL; }
      ParallelMax(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelMax() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        hipcub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        tmp = yaklAllocDevice(nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP);
          yaklFreeDevice(tmp);
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        hipcub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        hipcub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };

    template <class T> class ParallelSum<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelSum() { tmp = NULL; }
      ParallelSum(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelSum() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        hipcub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        tmp = yaklAllocDevice(nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP);
          yaklFreeDevice(tmp);
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        hipcub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        hipcub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };

  #elif defined(__USE_CUDA__)

    template <class T> class ParallelMin<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMin() { tmp = NULL; }
      ParallelMin(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelMin() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        cub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        tmp = yaklAllocDevice(nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP);
          yaklFreeDevice(tmp);
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        cub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        cub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };

    template <class T> class ParallelMax<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMax() { tmp = NULL; }
      ParallelMax(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelMax() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        cub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        tmp = yaklAllocDevice(nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP);
          yaklFreeDevice(tmp);
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        cub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        cub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };

    template <class T> class ParallelSum<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelSum() { tmp = NULL; }
      ParallelSum(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelSum() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        cub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        tmp = yaklAllocDevice(nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP);
          yaklFreeDevice(tmp);
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        cub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        cub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };

  #else

    template <class T> class ParallelMin<T,memDevice> {
      int    nItems; // Number of items in the array that will be reduced
      public:
      ParallelMin() {}
      ParallelMin(int const nItems) {
        this->nItems = nItems;
      }
      ~ParallelMin() {
      }
      void setup(int nItems) { this->nItems = nItems; }
      T operator() (T *data) {
        T rslt = data[0];
        for (int i=1; i<nItems; i++) {
          rslt = data[i] < rslt ? data[i] : rslt;
        }
        return rslt;
      }
      void deviceReduce(T *data, T *rslt) {
        *(rslt) = data[0];
        for (int i=1; i<nItems; i++) {
          *(rslt) = data[i] < *(rslt) ? data[i] : rslt;
        }
      }
    };

    template <class T> class ParallelMax<T,memDevice> {
      int    nItems; // Number of items in the array that will be reduced
      public:
      ParallelMax() {}
      ParallelMax(int const nItems) {
        this->nItems = nItems;
      }
      ~ParallelMax() {
      }
      void setup(int nItems) { this->nItems = nItems; }
      T operator() (T *data) {
        T rslt = data[0];
        for (int i=1; i<nItems; i++) {
          rslt = data[i] > rslt ? data[i] : rslt;
        }
        return rslt;
      }
      void deviceReduce(T *data, T *rslt) {
        *(rslt) = data[0];
        for (int i=1; i<nItems; i++) {
          *(rslt) = data[i] > *(rslt) ? data[i] : rslt;
        }
      }
    };

    template <class T> class ParallelSum<T,memDevice> {
      int    nItems; // Number of items in the array that will be reduced
      public:
      ParallelSum() {}
      ParallelSum(int const nItems) {
        this->nItems = nItems;
      }
      ~ParallelSum() {
      }
      void setup(int nItems) { this->nItems = nItems; }
      T operator() (T *data) {
        T rslt = data[0];
        for (int i=1; i<nItems; i++) {
          rslt += data[i];
        }
        return rslt;
      }
      void deviceReduce(T *data, T *rslt) {
        *(rslt) = data[0];
        for (int i=1; i<nItems; i++) {
          *(rslt) += data[i];
        }
      }
    };

  #endif

  template <class T> class ParallelMin<T,memHost> {
    int    nItems; // Number of items in the array that will be reduced
    public:
    ParallelMin() {}
    ParallelMin(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMin() {
    }
    void setup(int nItems) { this->nItems = nItems; }
    T operator() (T *data) {
      T rslt = data[0];
      for (int i=1; i<nItems; i++) {
        rslt = data[i] < rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *rslt) {
      *(rslt) = data[0];
      for (int i=1; i<nItems; i++) {
        *(rslt) = data[i] < *(rslt) ? data[i] : rslt;
      }
    }
  };

  template <class T> class ParallelMax<T,memHost> {
    int    nItems; // Number of items in the array that will be reduced
    public:
    ParallelMax() {}
    ParallelMax(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMax() {
    }
    void setup(int nItems) { this->nItems = nItems; }
    T operator() (T *data) {
      T rslt = data[0];
      for (int i=1; i<nItems; i++) {
        rslt = data[i] > rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *rslt) {
      *(rslt) = data[0];
      for (int i=1; i<nItems; i++) {
        *(rslt) = data[i] > *(rslt) ? data[i] : rslt;
      }
    }
  };

  template <class T> class ParallelSum<T,memHost> {
    int    nItems; // Number of items in the array that will be reduced
    public:
    ParallelSum() {}
    ParallelSum(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelSum() {
    }
    void setup(int nItems) { this->nItems = nItems; }
    T operator() (T *data) {
      T rslt = data[0];
      for (int i=1; i<nItems; i++) {
        rslt += data[i];
      }
      return rslt;
    }
    void deviceReduce(T *data, T *rslt) {
      *(rslt) = data[0];
      for (int i=1; i<nItems; i++) {
        *(rslt) += data[i];
      }
    }
  };



  
  #ifdef __USE_CUDA__
    __device__ __forceinline__ void atomicMin(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMin(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      }
    }
    ////////////////////////////////////////////////////////////
    // CUDA has HW atomics for atomicAdd in float, double, int, unsigned int, and unsigned long long int
    ////////////////////////////////////////////////////////////
    __device__ __forceinline__ void atomicAdd(float &update , double value) {
      ::atomicAdd( &update , value );
    }
    #if __CUDA_ARCH__ >= 600
      __device__ __forceinline__ void atomicAdd(double &update , double value) {
        ::atomicAdd( &update , value );
      }
    #else
      __device__ __forceinline__ void atomicAdd(double &update , double value) {
        unsigned long long oldval, newval, readback;
        oldval = __double_as_longlong(update);
        newval = __double_as_longlong( __longlong_as_double(oldval) + value );
        while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
          oldval = readback;
          newval = __double_as_longlong( __longlong_as_double(oldval) + value );
        }
      }
    #endif
    __device__ __forceinline__ void atomicAdd(int &update , int value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd(unsigned int &update , unsigned int value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd(unsigned long long int &update , unsigned long long int value) {
      ::atomicAdd( &update , value );
    }

    ////////////////////////////////////////////////////////////
    // CUDA has HW atomics for atomicMin int, unsigned int, and unsigned long long int
    ////////////////////////////////////////////////////////////
    __device__ __forceinline__ void atomicMin(int &update , int value) {
      ::atomicMin( &update , value );
    }
    __device__ __forceinline__ void atomicMin(unsigned int &update , unsigned int value) {
      ::atomicMin( &update , value );
    }
    __device__ __forceinline__ void atomicMin(unsigned long long int &update , unsigned long long int value) {
      ::atomicMin( &update , value );
    }

    ////////////////////////////////////////////////////////////
    // CUDA has HW atomics for atomicMax int, unsigned int, and unsigned long long int
    ////////////////////////////////////////////////////////////
    __device__ __forceinline__ void atomicMax(int &update , int value) {
      ::atomicMax( &update , value );
    }
    __device__ __forceinline__ void atomicMax(unsigned int &update , unsigned int value) {
      ::atomicMax( &update , value );
    }
    __device__ __forceinline__ void atomicMax(unsigned long long int &update , unsigned long long int value) {
      ::atomicMax( &update , value );
    }
  #endif

  #ifdef __USE_HIP__
    __device__ __forceinline__ void atomicMin(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMin(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      }
    }
    //////////////////////////////////////////////////////////////////////
    // HIP has HW atomicAdd for float, but not for double
    // Software atomicAdd in double is probably going to be slow as hell
    //////////////////////////////////////////////////////////////////////
    __device__ __forceinline__ void atomicAdd(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) + value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) + value );
      }
    }
    __device__ __forceinline__ void atomicAdd(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) + value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) + value );
      }
    }
  #else
    template <class T> inline void atomicAdd(T &update, T value) {
      update += value;
    }
    template <class T> inline void atomicMin(T &update, T value) {
      update = update < value ? update : value;
    }
    template <class T> inline void atomicMax(T &update, T value) {
      update = update > value ? update : value;
    }
  #endif
  

}


#endif

