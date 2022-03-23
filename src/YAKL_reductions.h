
#pragma once
// Included by YAKL.h

namespace yakl {

  template <class T, int memSpace, int RED> class ParallelReduction;

  int constexpr YAKL_REDUCTION_MIN = 0;
  int constexpr YAKL_REDUCTION_MAX = 1;
  int constexpr YAKL_REDUCTION_SUM = 2;


  // It is highly recommended that the user use yakl::intrinsics::minval, yakl::intrinsics::maxval, and
  // yakl::intrinsics::sum instead of the classes in this file.


  // These are parallel reduction classes for host and device
  // They handle all memory allocation and deallocation internally to reduce the chance of memory errors
  // setup and constructor take an integer of the number of items needing to be reduced
  // operator() will perform a reduction on the device and return the result as a scalar value
  // after copying to the host
  // deviceReduce will perform a reduciton on the device and leave hte result on the device as a pointer
  // Array is passed to the device as a pointer, so it's not completely safe is the user makes an error
  // where the repested nItems differs from the actual number of items in the array.


  template <class T, int RED> class ParallelReduction<T,memHost,RED> {
    int  nItems; // Number of items in the array that will be reduced
    public:
    ParallelReduction() {}
    ParallelReduction(int const nItems) { this->nItems = nItems; }
    ~ParallelReduction() {}
    void setup(int nItems) { this->nItems = nItems; }
    T operator() (T *data) {
      T rslt = data[0];
      if constexpr        (RED == YAKL_REDUCTION_MIN) {
        for (int i=1; i<nItems; i++) { rslt = data[i] < rslt ? data[i] : rslt; }
      } else if constexpr (RED == YAKL_REDUCTION_MAX) {
        for (int i=1; i<nItems; i++) { rslt = data[i] > rslt ? data[i] : rslt; }
      } else if constexpr (RED == YAKL_REDUCTION_SUM) {
        for (int i=1; i<nItems; i++) { rslt += data[i]; }
      }
      return rslt;
    }
    void deviceReduce(T *data, T *rslt) {
      *(rslt) = data[0];
      if constexpr        (RED == YAKL_REDUCTION_MIN) {
        for (int i=1; i<nItems; i++) { *(rslt) = data[i] < *(rslt) ? data[i] : rslt; }
      } else if constexpr (RED == YAKL_REDUCTION_MAX) {
        for (int i=1; i<nItems; i++) { *(rslt) = data[i] > *(rslt) ? data[i] : rslt; }
      } else if constexpr (RED == YAKL_REDUCTION_SUM) {
        for (int i=1; i<nItems; i++) { *(rslt) += data[i]; }
      }
    }
  };


  #ifdef YAKL_ARCH_HIP


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelReduction() { tmp = NULL; }
      ParallelReduction(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelReduction() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          hipcub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          hipcub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          hipcub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        }
        tmp = yaklAllocDevice(nTmp,"");       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP,"");
          yaklFreeDevice(tmp,"");
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          hipcub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          hipcub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          hipcub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        }
        hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
        check_last_error();
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          hipcub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          hipcub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          hipcub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
        }
        #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
          fence();
        #endif
      }
    };


  #elif defined(YAKL_ARCH_CUDA)


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelReduction() { tmp = NULL; }
      ParallelReduction(int const nItems) { tmp = NULL; setup(nItems); }
      ~ParallelReduction() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          cub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          cub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          cub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        }
        tmp = yaklAllocDevice(nTmp,"");       // Allocate temporary storage
        this->nItems = nItems;
      }
      void finalize() {
        if (tmp != NULL) {
          yaklFreeDevice(rsltP,"");
          yaklFreeDevice(tmp,"");
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        T rslt;
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          cub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          cub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          cub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        }
        cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
        check_last_error();
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          cub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          cub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          cub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
        }
        #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
          fence();
        #endif
      }
    };


  #elif defined(YAKL_ARCH_SYCL)

    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelReduction() { rsltP = nullptr; }
      ParallelReduction(int const nItems) { rsltP = nullptr; setup(nItems); }
      ~ParallelReduction() { finalize(); }
      void setup(int const nItems) {
        finalize();
        rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
        this->nItems = nItems;
      }

      void finalize() {
        if(rsltP != nullptr) {
          yaklFreeDevice(rsltP,"");
        }
        rsltP = nullptr;
      }
      T operator() (T *data) {
        T rslt=0;
        sycl_default_stream().submit([&, nItems = this->nItems](sycl::handler &cgh) {
          if constexpr        (RED == YAKL_REDUCTION_MIN) {
            cgh.parallel_for(sycl::range<1>(nItems),
                             sycl::reduction(rsltP, sycl::minimum<>(),
                             sycl::property::reduction::initialize_to_identity{}),
                             [=] (sycl::id<1> idx, auto& min) { min.combine(data[idx]); });
          } else if constexpr (RED == YAKL_REDUCTION_MAX) {
            cgh.parallel_for(sycl::range<1>(nItems),
                             sycl::reduction(rsltP, sycl::maximum<>(),
                             sycl::property::reduction::initialize_to_identity{}),
                             [=] (sycl::id<1> idx, auto& max) { max.combine(data[idx]); });
          } else if constexpr (RED == YAKL_REDUCTION_SUM) {
            cgh.parallel_for(sycl::range<1>(nItems),
                             sycl::reduction(rsltP, std::plus<>(),
                             sycl::property::reduction::initialize_to_identity{}),
                             [=] (sycl::id<1> idx, auto& sum) { sum.combine(data[idx]); });
          }
        });
        sycl_default_stream().memcpy(&rslt,rsltP,sizeof(T)); // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        sycl_default_stream().submit([&, nItems = this->nItems](sycl::handler &cgh) {
          if constexpr        (RED == YAKL_REDUCTION_MIN) {
            cgh.parallel_for(sycl::range<1>(nItems),
                             sycl::reduction(devP, sycl::minimum<>(),
                             sycl::property::reduction::initialize_to_identity{}),
                             [=] (sycl::id<1> idx, auto& min) { min.combine(data[idx]); });
          } else if constexpr (RED == YAKL_REDUCTION_MAX) {
            cgh.parallel_for(get_reduction_range(nItems, devP),
                             sycl::reduction(devP, sycl::maximum<>(),
                             sycl::property::reduction::initialize_to_identity{}),
                             [=] (sycl::id<1> idx, auto& max) { max.combine(data[idx]); });
          } else if constexpr (RED == YAKL_REDUCTION_SUM) {
            cgh.parallel_for(sycl::range<1>(nItems),
                             sycl::reduction(rsltP, std::plus<T>(),
                             sycl::property::reduction::initialize_to_identity{}),
                             [=] (sycl::id<1> idx, auto& sum) { sum.combine(data[idx]); });
          }
        });
        #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
          fence();
        #endif
      }
    };



  #elif defined(YAKL_ARCH_OPENMP)


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      int  nItems; // Number of items in the array that will be reduced
      public:
      ParallelReduction() {}
      ParallelReduction(int const nItems) { this->nItems = nItems; }
      ~ParallelReduction() {}
      void setup(int nItems) { this->nItems = nItems; }
      T operator() (T *data) {
        T rslt = data[0];
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          #pragma omp parallel for reduction(min:rslt)
          for (int i=1; i<nItems; i++) { rslt = data[i] < rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          #pragma omp parallel for reduction(max:rslt)
          for (int i=1; i<nItems; i++) { rslt = data[i] > rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          #pragma omp parallel for reduction(+:rslt)
          for (int i=1; i<nItems; i++) { rslt += data[i]; }
        }
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        T rslt = data[0];
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          #pragma omp parallel for reduction(min:rslt)
          for (int i=1; i<nItems; i++) { rslt = data[i] < rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          #pragma omp parallel for reduction(max:rslt)
          for (int i=1; i<nItems; i++) { rslt = data[i] > rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          #pragma omp parallel for reduction(+:rslt)
          for (int i=1; i<nItems; i++) { rslt += data[i]; }
        }
        *devP = rslt;
      }
    };

    
  #else


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      int  nItems; // Number of items in the array that will be reduced
      public:
      ParallelReduction() {}
      ParallelReduction(int const nItems) { this->nItems = nItems; }
      ~ParallelReduction() {}
      void setup(int nItems) { this->nItems = nItems; }
      T operator() (T *data) {
        T rslt = data[0];
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          for (int i=1; i<nItems; i++) { rslt = data[i] < rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          for (int i=1; i<nItems; i++) { rslt = data[i] > rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          for (int i=1; i<nItems; i++) { rslt += data[i]; }
        }
        return rslt;
      }
      void deviceReduce(T *data, T *rslt) {
        *(rslt) = data[0];
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          for (int i=1; i<nItems; i++) { *(rslt) = data[i] < *(rslt) ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          for (int i=1; i<nItems; i++) { *(rslt) = data[i] > *(rslt) ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          for (int i=1; i<nItems; i++) { *(rslt) += data[i]; }
        }
      }
    };


  #endif


  template <class T, int memSpace> using ParallelMin = ParallelReduction<T,memSpace,YAKL_REDUCTION_MIN>;
  template <class T, int memSpace> using ParallelMax = ParallelReduction<T,memSpace,YAKL_REDUCTION_MAX>;
  template <class T, int memSpace> using ParallelSum = ParallelReduction<T,memSpace,YAKL_REDUCTION_SUM>;

}


