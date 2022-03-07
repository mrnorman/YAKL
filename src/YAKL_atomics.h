
#pragma once
// Included by YAKL.h

namespace yakl {

  // These are YAKL's atomic operations: atomicAdd, atomicMin, and atomicMax
  // Where possible, hardware atomics are used. Where that's not possible, CompareAndSwap (CAS)
  // implementations are used. 

  template <class T> inline void atomicAdd_host(T &update, T value) {
    update += value;
  }
  template <class T> inline void atomicMin_host(T &update, T value) {
    update = update < value ? update : value;
  }
  template <class T> inline void atomicMax_host(T &update, T value) {
    update = update > value ? update : value;
  }


  #ifdef YAKL_ARCH_CUDA


    __device__ __forceinline__ void atomicMin_device(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      }
    }
    __device__ __forceinline__ void atomicMin_device(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      }
    }
    __device__ __forceinline__ void atomicMin_device(int &update , int value) {
      ::atomicMin( &update , value );
    }
    __device__ __forceinline__ void atomicMin_device(unsigned int &update , unsigned int value) {
      ::atomicMin( &update , value );
    }
    __device__ __forceinline__ void atomicMin_device(unsigned long long int &update , unsigned long long int value) {
      #if __CUDA_ARCH__ >= 350
        ::atomicMin( &update , value );
      #else
        yakl_throw("ERROR: atomicMin not implemented for unsigned long long int for this CUDA architecture");
      #endif
    }
    template <class T> __host__ __device__ __forceinline__ void atomicMin(T &update , T value) {
      #if YAKL_CURRENTLY_ON_DEVICE()
        atomicMin_device(update,value);
      #else
        atomicMin_host  (update,value);
      #endif
    }

    __device__ __forceinline__ void atomicMax_device(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax_device(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      }
    }
    __device__ __forceinline__ void atomicMax_device(int &update , int value) {
      ::atomicMax( &update , value );
    }
    __device__ __forceinline__ void atomicMax_device(unsigned int &update , unsigned int value) {
      ::atomicMax( &update , value );
    }
    __device__ __forceinline__ void atomicMax_device(unsigned long long int &update , unsigned long long int value) {
      #if __CUDA_ARCH__ >= 350
        ::atomicMax( &update , value );
      #else
        yakl_throw("ERROR: atomicMin not implemented for unsigned long long int for this CUDA architecture");
      #endif
    }
    template <class T> __host__ __device__ __forceinline__ void atomicMax(T &update , T value) {
      #if YAKL_CURRENTLY_ON_DEVICE()
        atomicMax_device(update,value);
      #else
        atomicMax_host  (update,value);
      #endif
    }

    __device__ __forceinline__ void atomicAdd_device(float &update , float value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(double &update , double value) {
      #if __CUDA_ARCH__ >= 600
        ::atomicAdd( &update , value );
      #else
        unsigned long long oldval, newval, readback;
        oldval = __double_as_longlong(update);
        newval = __double_as_longlong( __longlong_as_double(oldval) + value );
        while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
          oldval = readback;
          newval = __double_as_longlong( __longlong_as_double(oldval) + value );
        }
      #endif
    }
    __device__ __forceinline__ void atomicAdd_device(int &update , int value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(unsigned int &update , unsigned int value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(unsigned long long int &update , unsigned long long int value) {
      ::atomicAdd( &update , value );
    }
    template <class T> __host__ __device__ __forceinline__ void atomicAdd(T &update , T value) {
      #if YAKL_CURRENTLY_ON_DEVICE()
        atomicAdd_device(update,value);
      #else
        atomicAdd_host  (update,value);
      #endif
    }


  #elif defined(YAKL_ARCH_SYCL)


    template <typename T, sycl::access::address_space addressSpace =
        sycl::access::address_space::global_space>
    using relaxed_atomic_ref =
          sycl::atomic_ref< T,
          sycl::memory_order::seq_cst,
          sycl::memory_scope::device,
          addressSpace>;

    template <typename T, sycl::access::address_space addressSpace =
        sycl::access::address_space::global_space>
    __inline__ __attribute__((always_inline)) void atomicMin(T &update , T value) {
      relaxed_atomic_ref<T, addressSpace>( update ).fetch_min( value );
    }

    template <typename T, sycl::access::address_space addressSpace =
        sycl::access::address_space::global_space>
    __inline__ __attribute__((always_inline)) void atomicMax(T &update , T value) {
      relaxed_atomic_ref<T, addressSpace>( update ).fetch_max( value );
    }

    template <typename T, sycl::access::address_space addressSpace =
        sycl::access::address_space::global_space>
    __inline__ __attribute__((always_inline)) void atomicAdd(T &update , T value) {
      relaxed_atomic_ref<T, addressSpace>( update ).fetch_add( value );
    }


  #elif defined(YAKL_ARCH_HIP)


    __device__ __forceinline__ void atomicMin_device(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMin_device(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      }
    }
    __device__ __forceinline__ void atomicMin_device(int &update , int value) {
      ::atomicMin( &update , value );
    }
    __device__ __forceinline__ void atomicMin_device(unsigned int &update , unsigned int value) {
      ::atomicMin( &update , value );
    }
    __device__ __forceinline__ void atomicMin_device(unsigned long long int &update , unsigned long long int value) {
      ::atomicMin( &update , value );
    }
    template <class T> __host__ __device__ __forceinline__ void atomicMin(T &update , T value) {
      #if YAKL_CURRENTLY_ON_DEVICE()
        atomicMin_device(update,value);
      #else
        atomicMin_host  (update,value);
      #endif
    }

    __device__ __forceinline__ void atomicMax_device(float &update , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(update);
      newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax_device(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      }
    }
    __device__ __forceinline__ void atomicMax_device(int &update , int value) {
      ::atomicMax( &update , value );
    }
    __device__ __forceinline__ void atomicMax_device(unsigned int &update , unsigned int value) {
      ::atomicMax( &update , value );
    }
    __device__ __forceinline__ void atomicMax_device(unsigned long long int &update , unsigned long long int value) {
      ::atomicMax( &update , value );
    }
    template <class T> __host__ __device__ __forceinline__ void atomicMax(T &update , T value) {
      #if YAKL_CURRENTLY_ON_DEVICE()
        atomicMax_device(update,value);
      #else
        atomicMax_host  (update,value);
      #endif
    }

    __device__ __forceinline__ void atomicAdd_device(float &update , float value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(double &update , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(update);
      newval = __double_as_longlong( __longlong_as_double(oldval) + value );
      while ( ( readback = atomicCAS( (unsigned long long *) &update , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) + value );
      }
    }
    __device__ __forceinline__ void atomicAdd_device(int &update , int value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(unsigned int &update , unsigned int value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(unsigned long long int &update , unsigned long long int value) {
      ::atomicAdd( &update , value );
    }
    template <class T> __host__ __device__ __forceinline__ void atomicAdd(T &update , T value) {
      #if YAKL_CURRENTLY_ON_DEVICE()
        atomicAdd_device(update,value);
      #else
        atomicAdd_host  (update,value);
      #endif
    }


  #elif defined(YAKL_ARCH_OPENMP45)


    template <class T> inline void atomicMin(T&update, T value) {
      #pragma omp critical
      { update = value < update ? value : update; }
    }

    template <class T> inline void atomicMax(T &update, T value) {
      #pragma omp critical
      { update = value > update ? value : update; }
    }

    template <class T> inline void atomicAdd(T &update, T value) {
      #pragma omp atomic update 
      update += value;
    }


  #elif defined(YAKL_ARCH_OPENMP)


    template <class T> inline void atomicMin(T &update, T value) {
      #pragma omp critical
      { update = value < update ? value : update; }
    }

    template <class T> inline void atomicMax(T &update, T value) {
      #pragma omp critical
      { update = value > update ? value : update; }
    }

    template <class T> inline void atomicAdd(T &update, T value) {
      #pragma omp atomic update 
      update += value;
    }


  #else


    template <class T> inline void atomicMin(T &update, T value) { atomicMin_host(update,value); }
    template <class T> inline void atomicMax(T &update, T value) { atomicMax_host(update,value); }
    template <class T> inline void atomicAdd(T &update, T value) { atomicAdd_host(update,value); }


  #endif

}


