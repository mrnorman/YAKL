
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

  #elif defined(__USE_SYCL__)
    template <typename T, sycl::access::address_space addressSpace =
	      sycl::access::address_space::global_space>
    using relaxed_atomic_ref =
	sycl::ONEAPI::atomic_ref< T,
				  sycl::ONEAPI::memory_order::relaxed,
				  sycl::ONEAPI::memory_scope::device,
				  addressSpace>;


    template <typename T,
	      sycl::access::address_space addressSpace =
	      sycl::access::address_space::global_space>
    __inline__ __attribute__((always_inline)) void atomicAdd(T &update , T value) {
      relaxed_atomic_ref<T, addressSpace>( update ).fetch_add( value );
    }

    ////////////////////////////////////////////////////////////
    // SYCL's atomics for reals could be quite slow with different Intel hardware
    ////////////////////////////////////////////////////////////

    template <typename T=float, sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space>
    __inline__ __attribute__((always_inline)) void atomicAdd(float &addr , float operand) {
      static_assert(sizeof(float) == sizeof(int), "Mismatched type size");

      sycl::atomic<int, addressSpace> obj(
        (sycl::multi_ptr<int, addressSpace>(reinterpret_cast<int *>(&addr))));

      int old_value;
      float old_float_value;

      do {
        old_value = obj.load(sycl::ONEAPI::memory_order::relaxed);
        old_float_value = *reinterpret_cast<const float *>(&old_value);
        const float new_float_value = old_float_value + operand;
        const int new_value = *reinterpret_cast<const int *>(&new_float_value);
        if (obj.compare_exchange_strong(old_value, new_value, sycl::ONEAPI::memory_order::relaxed))
          break;
      } while (true);

      return;
    }

    template <typename T=double, sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space>
    __inline__ __attribute__((always_inline)) void atomicAdd(double &addr , double operand) {
      static_assert(sizeof(double) == sizeof(unsigned long long int),
                    "Mismatched type size");

      sycl::atomic<unsigned long long int, addressSpace> obj(
        (sycl::multi_ptr<unsigned long long int, addressSpace>(
          reinterpret_cast<unsigned long long int *>(&addr))));

      unsigned long long int old_value;
      double old_double_value;

      do {
        old_value = obj.load(sycl::ONEAPI::memory_order::relaxed);
        old_double_value = *reinterpret_cast<const double *>(&old_value);
        const double new_double_value = old_double_value + operand;
        const unsigned long long int new_value =
          *reinterpret_cast<const unsigned long long int *>(&new_double_value);

        if (obj.compare_exchange_strong(old_value, new_value, sycl::ONEAPI::memory_order::relaxed))
          break;
      } while (true);

      return;
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

  #elif defined(__USE_HIP__)
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
  #endif

  #ifdef __USE_OPENMP45__
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
