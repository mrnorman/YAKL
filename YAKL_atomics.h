
#pragma once


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
    __device__ __forceinline__ void atomicAdd(float &update , float value) {
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
    #if __CUDA_ARCH__ >= 350
      __device__ __forceinline__ void atomicMin(unsigned long long int &update , unsigned long long int value) {
        ::atomicMin( &update , value );
      }
    #endif

    ////////////////////////////////////////////////////////////
    // CUDA has HW atomics for atomicMax int, unsigned int, and unsigned long long int
    ////////////////////////////////////////////////////////////
    __device__ __forceinline__ void atomicMax(int &update , int value) {
      ::atomicMax( &update , value );
    }
    __device__ __forceinline__ void atomicMax(unsigned int &update , unsigned int value) {
      ::atomicMax( &update , value );
    }
    #if __CUDA_ARCH__ >= 350
      __device__ __forceinline__ void atomicMax(unsigned long long int &update , unsigned long long int value) {
        ::atomicMax( &update , value );
      }
    #endif
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
