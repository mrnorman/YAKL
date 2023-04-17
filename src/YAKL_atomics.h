/**
 * @file
 *
 * Defines atomic operations for each hardware backend.
 */

#pragma once
// Included by YAKL.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  // These are YAKL's atomic operations: atomicAdd, atomicMin, and atomicMax
  // Where possible, hardware atomics are used. Where that's not possible, CompareAndSwap (CAS)
  // implementations are used. 

  /** @private */
  template <class T> inline void atomicAdd_host(T &update, T value) {
    update += value;
  }
  /** @private */
  template <class T> inline void atomicMin_host(T &update, T value) {
    update = update < value ? update : value;
  }
  /** @private */
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
      #if ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350) ) || ( defined(__NVCOMPILER_CUDA_ARCH__) && (__NVCOMPILER_CUDA_ARCH__ >= 350) )
        ::atomicMin( &update , value );
      #else
        yakl_throw("ERROR: atomicMin not implemented for unsigned long long int for this CUDA architecture");
      #endif
    }
    template <class T> __host__ __device__ __forceinline__ void atomicMin(T &update , T value) {
      // This is safe from executing the command twice because with a CUDA backend, only one of these executes
      YAKL_EXECUTE_ON_DEVICE_ONLY( atomicMin_device(update,value); )
      YAKL_EXECUTE_ON_HOST_ONLY( atomicMin_host  (update,value); )
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
      #if ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350) ) || ( defined(__NVCOMPILER_CUDA_ARCH__) && (__NVCOMPILER_CUDA_ARCH__ >= 350) )
        ::atomicMax( &update , value );
      #else
        yakl_throw("ERROR: atomicMin not implemented for unsigned long long int for this CUDA architecture");
      #endif
    }
    template <class T> __host__ __device__ __forceinline__ void atomicMax(T &update , T value) {
      // This is safe from executing the command twice because with a CUDA backend, only one of these executes
      YAKL_EXECUTE_ON_DEVICE_ONLY( atomicMax_device(update,value); )
      YAKL_EXECUTE_ON_HOST_ONLY( atomicMax_host  (update,value); )
    }

    __device__ __forceinline__ void atomicAdd_device(float &update , float value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(double &update , double value) {
      #if ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600) ) || ( defined(__NVCOMPILER_CUDA_ARCH__) && (__NVCOMPILER_CUDA_ARCH__ >= 600) )
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
      // This is safe from executing the command twice because with a CUDA backend, only one of these executes
      YAKL_EXECUTE_ON_DEVICE_ONLY( atomicAdd_device(update,value); )
      YAKL_EXECUTE_ON_HOST_ONLY( atomicAdd_host  (update,value); )
    }


  #elif defined(YAKL_ARCH_SYCL)


    template <typename T, sycl::access::address_space addressSpace =
        sycl::access::address_space::global_space>
    using relaxed_atomic_ref =
          sycl::atomic_ref< T,
          sycl::memory_order::relaxed,
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
      // This is safe from executing the command twice because with a HIP backend, only one of these executes
      YAKL_EXECUTE_ON_DEVICE_ONLY( atomicMin_device(update,value); )
      YAKL_EXECUTE_ON_HOST_ONLY( atomicMin_host  (update,value); )
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
      // This is safe from executing the command twice because with a HIP backend, only one of these executes
      YAKL_EXECUTE_ON_DEVICE_ONLY( atomicMax_device(update,value); )
      YAKL_EXECUTE_ON_HOST_ONLY( atomicMax_host  (update,value); )
    }

    __device__ __forceinline__ void atomicAdd_device(float &update , float value) {
      ::atomicAdd( &update , value );
    }
    __device__ __forceinline__ void atomicAdd_device(double &update , double value) {
      ::atomicAdd( &update , value );
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
      // This is safe from executing the command twice because with a HIP backend, only one of these executes
      YAKL_EXECUTE_ON_DEVICE_ONLY( atomicAdd_device(update,value); )
      YAKL_EXECUTE_ON_HOST_ONLY( atomicAdd_host  (update,value); )
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


    /**
     * @brief `yakl::atomicMin(update,value)` atomically performs `update = min(update,value)`
     * 
     * Atomic instructions exist when multiple parallel threads are attempting to read-write to the same memory
     * location at the same time. Min, max, and add will read a memory location perform a local operation, and then
     * write a new value to that location. Atomic instructions ensure that the memory location has not changed between
     * reading the memory location and writing a new value to that location.
     *
     * Atomic min, max, and add are typically needed when you are writing to an array with fewer entries or dimensions than
     * the number of threads in the parallel_for() kernel launch. E.g.:
     * ```
     * parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
     *   yakl::atomicAdd( average_column(k) , data(k,j,i) / (ny*nx) );
     * });
     * ```
     * 
     * **IMPORTANT**: yakl::atomicAdd() is not bitwise deterministic for floating point (FP) numbers, meaning there
     * is no guarantee what order threads will perform FP addition. Since FP addition is not commutative, you cannot
     * guarantee bitwise reproducible results from one run to the next. To alleviate this, please use pass
     * yakl::DefaultLaunchConfigB4b to the parallel_for() launcher, and when you want to force bitwise reproducibility
     * define the CPP macro `YAKL_B4B`. yakl::atomicMin() and yakl::atomicMax() are both bitwise reproducible, so
     * do not worry about those. This is only for yakl::atomicAdd().
     */
    template <class T> YAKL_INLINE void atomicMin(T &update, T value) { atomicMin_host(update,value); }

    /**
     * @brief `yakl::atomicMax(update,value)` atomically performs `update = max(update,value)`
     * \copydetails atomicMin
     */
    template <class T> YAKL_INLINE void atomicMax(T &update, T value) { atomicMax_host(update,value); }

    /**
     * @brief `[NON_B4B] yakl::atomicAdd(update,value)` atomically performs `update += value)`
     * \copydetails atomicMin
     */
    template <class T> YAKL_INLINE void atomicAdd(T &update, T value) { atomicAdd_host(update,value); }


  #endif

}
__YAKL_NAMESPACE_WRAPPER_END__


