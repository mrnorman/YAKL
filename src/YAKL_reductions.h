
#pragma once


template <class T, int myMem> class ParallelMin;
template <class T, int myMem> class ParallelMax;
template <class T, int myMem> class ParallelSum;


#ifdef YAKL_ARCH_HIP


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
      rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
      // Get the amount of temporary storage needed (call with NULL storage pointer)
      hipcub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
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
      hipcub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
      hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
      check_last_error();
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      hipcub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
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
      rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
      // Get the amount of temporary storage needed (call with NULL storage pointer)
      hipcub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
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
      hipcub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
      hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
      check_last_error();
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      hipcub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
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
      rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
      // Get the amount of temporary storage needed (call with NULL storage pointer)
      hipcub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
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
      hipcub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
      hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
      check_last_error();
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      hipcub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
    }
  };


#elif defined(YAKL_ARCH_CUDA)


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
      rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
      // Get the amount of temporary storage needed (call with NULL storage pointer)
      cub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
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
      cub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
      cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
      check_last_error();
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      cub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
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
      rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
      // Get the amount of temporary storage needed (call with NULL storage pointer)
      cub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
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
      cub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
      cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
      check_last_error();
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      cub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
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
      rsltP = (T *) yaklAllocDevice(sizeof(T),""); // Allocate device pointer for result
      // Get the amount of temporary storage needed (call with NULL storage pointer)
      cub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
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
      cub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
      cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
      check_last_error();
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      cub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
    }
  };


#elif defined(YAKL_ARCH_SYCL)

  template <class T> class ParallelMin<T,memDevice> {
    int    nItems; // Number of items in the array that will be reduced
    T      *rsltP; // Device pointer for reduction result
    public:
    ParallelMin() { rsltP = nullptr; }
    ParallelMin(int const nItems) { rsltP = nullptr; setup(nItems); }
    ~ParallelMin() { finalize(); }
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
        cgh.parallel_for(sycl::range<1>(nItems),
                         sycl::reduction(rsltP, sycl::minimum<>(), sycl::property::reduction::initialize_to_identity{}),
                         [=](sycl::id<1> idx, auto& min) {
                           min.combine(data[idx]);
                         });
      });
      sycl_default_stream().memcpy(&rslt,rsltP,sizeof(T)); // Copy result to host
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      sycl_default_stream().submit([&, nItems = this->nItems](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(nItems),
                         sycl::reduction(devP, sycl::minimum<>(), sycl::property::reduction::initialize_to_identity{}),
                         [=](sycl::id<1> idx, auto& min) {
                           min.combine(data[idx]);
                         });
      });
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
    }
  };

  template <class T> class ParallelMax<T,memDevice> {
    int    nItems; // Number of items in the array that will be reduced
    T      *rsltP; // Device pointer for reduction result
    public:
    ParallelMax() { rsltP = nullptr; }
    ParallelMax(int const nItems) { rsltP = nullptr; setup(nItems); }
    ~ParallelMax() { finalize(); }
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
        cgh.parallel_for(sycl::range<1>(nItems),
                         sycl::reduction(rsltP, sycl::maximum<>(), sycl::property::reduction::initialize_to_identity{}),
                         [=](sycl::id<1> idx, auto& max) {
                           max.combine(data[idx]);
                         });
      });
      sycl_default_stream().memcpy(&rslt,rsltP,sizeof(T)); // Copy result to host
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      sycl_default_stream().submit([&, nItems = this->nItems](sycl::handler &cgh) {
        cgh.parallel_for(get_reduction_range(nItems, devP),
                         sycl::reduction(devP, sycl::maximum<>(), sycl::property::reduction::initialize_to_identity{}),
                         [=](sycl::id<1> idx, auto& max) {
                           max.combine(data[idx]);
                         });
      });
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
    }
  };

  template <class T> class ParallelSum<T,memDevice> {
    int    nItems; // Number of items in the array that will be reduced
    T      *rsltP; // Device pointer for reduction result
    public:
    ParallelSum() { rsltP = nullptr; }
    ParallelSum(int const nItems) { rsltP = nullptr; setup(nItems); }
    ~ParallelSum() { finalize(); }
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
        cgh.parallel_for(sycl::range<1>(nItems),
                         sycl::reduction(rsltP, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
                         [=](sycl::id<1> idx, auto& sum) {
			   sum.combine(data[idx]);
                         });
      });
      sycl_default_stream().memcpy(&rslt,rsltP,sizeof(T));
      fence();
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      sycl_default_stream().submit([&, nItems = this->nItems](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(nItems),
                         sycl::reduction(rsltP, std::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
                         [=](sycl::id<1> idx, auto& sum) {
			   sum.combine(data[idx]);
                         });
      });
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
    }
  };


#elif defined(YAKL_ARCH_OPENMP45)


  template <class T> class ParallelSum<T,memDevice> {
    int nItems;
    public:
    ParallelSum() {}
    ParallelSum(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelSum() { }
    T operator() (T *data) {
      T rslt = 0;
      #pragma omp target teams distribute parallel for simd reduction(+:rslt) is_device_ptr(data)
      for(int i=0; i<nItems; i++) {
        rslt += data[i];
      }
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      T rslt = 0;
      #pragma omp target teams distribute parallel for simd reduction(+:rslt) is_device_ptr(data)
      for (int i=0; i<nItems; i++) {
        rslt += data[i];
      }
      omp_target_memcpy(devP,&rslt,sizeof(T),0,0,omp_get_default_device(),omp_get_initial_device());
      #pragma omp taskwait
      check_last_error();
    }
  };  
  template <class T> class ParallelMin<T,memDevice> {
    int nItems;
    public:
    ParallelMin() {}
    ParallelMin(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMin() { }
    T operator() (T *data) {
      T rslt = std::numeric_limits<T>::max();
      #pragma omp target teams distribute parallel for simd reduction(min:rslt) is_device_ptr(data)
      for(int i=0; i<nItems; i++) {
        rslt = data[i] < rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      T rslt = std::numeric_limits<T>::max();
      #pragma omp target teams distribute parallel for simd reduction(min:rslt) is_device_ptr(data)
      for (int i=0; i<nItems; i++) {
        rslt = data[i] < rslt ? data[i] : rslt;
      }
      omp_target_memcpy(devP,&rslt,sizeof(T),0,0,omp_get_default_device(),omp_get_initial_device());
      #pragma omp taskwait
      check_last_error();
    }
  };  
  template <class T> class ParallelMax<T,memDevice> {
    int nItems;
    public:
    ParallelMax() {}
    ParallelMax(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMax() { }
    T operator() (T *data) {
      T rslt = std::numeric_limits<T>::lowest();
      #pragma omp target teams distribute parallel for simd reduction(max:rslt) is_device_ptr(data)
      for(int i=0; i<nItems; i++) {
        rslt = data[i] > rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      T rslt = std::numeric_limits<T>::lowest();
      #pragma omp target teams distribute parallel for simd reduction(max:rslt) is_device_ptr(data)
      for (int i=0; i<nItems; i++) {
        rslt = data[i] > rslt ? data[i] : rslt;
      }
      omp_target_memcpy(devP,&rslt,sizeof(T),0,0,omp_get_default_device(),omp_get_initial_device());
      #pragma omp taskwait
      check_last_error();
    }
  };  


#elif defined(YAKL_ARCH_OPENMP)


  template <class T> class ParallelSum<T,memDevice> {
    int nItems;
    public:
    ParallelSum() {}
    ParallelSum(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelSum() { }
    T operator() (T *data) {
      T rslt = 0;
      #pragma omp parallel for reduction(+:rslt)
      for(int i=0; i<nItems; i++) {
        rslt += data[i];
      }
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      T rslt = 0;
      #pragma omp parallel for reduction(+:rslt)
      for (int i=0; i<nItems; i++) {
        rslt += data[i];
      }
      *devP = rslt;
    }
  };  
  template <class T> class ParallelMin<T,memDevice> {
    int nItems;
    public:
    ParallelMin() {}
    ParallelMin(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMin() { }
    T operator() (T *data) {
      T rslt = std::numeric_limits<T>::max();
      #pragma omp parallel for reduction(min:rslt)
      for(int i=0; i<nItems; i++) {
        rslt = data[i] < rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      T rslt = std::numeric_limits<T>::max();
      #pragma omp parallel for reduction(min:rslt)
      for (int i=0; i<nItems; i++) {
        rslt = data[i] < rslt ? data[i] : rslt;
      }
      *devP = rslt;
    }
  };  
  template <class T> class ParallelMax<T,memDevice> {
    int nItems;
    public:
    ParallelMax() {}
    ParallelMax(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMax() { }
    T operator() (T *data) {
      T rslt = std::numeric_limits<T>::lowest();
      #pragma omp parallel for reduction(max:rslt)
      for(int i=0; i<nItems; i++) {
        rslt = data[i] > rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      T rslt = std::numeric_limits<T>::lowest();
      #pragma omp parallel for reduction(max:rslt)
      for (int i=0; i<nItems; i++) {
        rslt = data[i] > rslt ? data[i] : rslt;
      }
      *devP = rslt;
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
