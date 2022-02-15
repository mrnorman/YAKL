
#pragma once


#ifdef YAKL_ARCH_CUDA


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
  __device__ __forceinline__ void atomicAdd(float &update , float value) {
    ::atomicAdd( &update , value );
  }
  __device__ __forceinline__ void atomicAdd(double &update , double value) {
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
    #if __CUDA_ARCH__ >= 350
      ::atomicMin( &update , value );
    #else
      yakl_throw("ERROR: atomicMin not implemented for unsigned long long int for this CUDA architecture");
    #endif
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
    #if __CUDA_ARCH__ >= 350
      ::atomicMax( &update , value );
    #else
      yakl_throw("ERROR: atomicMin not implemented for unsigned long long int for this CUDA architecture");
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
  __inline__ __attribute__((always_inline)) void atomicAdd(T &update , T value) {
    relaxed_atomic_ref<T, addressSpace>( update ).fetch_add( value );
  }

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


#elif defined(YAKL_ARCH_HIP)


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
    ::atomicAdd( &update , value );
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


#elif defined(YAKL_ARCH_OPENMP45)


  template <class T> inline void atomicAdd(T &update, T value) {
    #pragma omp atomic update 
    update += value;
  }
  template <class T> inline void atomicMin(T&update, T value) {
    #pragma omp critical
    {
      update = value < update ? value : update;
      //if (value < update){update = value;}

    }
    //T tmp;
    //#pragma omp atomic read
    //  tmp = update;
    //if (tmp > value) {
    //  #pragma omp atomic write
    //    update = value;
    //}
  }
  template <class T> inline void atomicMax(T &update, T value) {
    #pragma omp critical
    {
      update = value > update ? value : update;
      //if(value > update){update = value;}
    }
    //T tmp;
    //#pragma omp atomic read
    //  tmp = update;
    //if (tmp < value) {
    //  #pragma omp atomic write
    //    update = value;
    //}
  }


#elif defined(YAKL_ARCH_OPENMP)


  template <class T> inline void atomicAdd(T &update, T value) {
    #pragma omp atomic update 
    update += value;
  }
  template <class T> inline void atomicMin(T &update, T value) {
    #pragma omp critical
    {
      update = value < update ? value : update;
    }
  }
  template <class T> inline void atomicMax(T &update, T value) {
    #pragma omp critical
    {
      update = value > update ? value : update;
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
