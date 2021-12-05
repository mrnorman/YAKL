
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


  static inline size_t get_wg_size_for_reduction(size_t bytes_per_wi) {
    // The best work-group size depends on implementation details
    // We make the following assumptions, which aren't specific to DPC++:
    // - Bigger work-groups are better
    // - An implementation may reserve 1 element per work-item in shared memory
    // In practice, DPC++ seems to limit itself to 1/2 of this
    const size_t max_size = sycl_default_stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    const size_t local_mem = sycl_default_stream->get_device().get_info<sycl::info::device::local_mem_size>();
    return std::min(local_mem / bytes_per_wi, max_size) / 2;
  }

  static inline size_t round_up(size_t N, size_t multiple) { return ((N + multiple - 1) / multiple) * multiple; }

  template <class T>
  static inline sycl::nd_range<1> get_reduction_range(size_t N, T reductionVars) {
    size_t bytes_per_wi = sizeof( std::remove_pointer_t<T> );
    size_t L = get_wg_size_for_reduction(bytes_per_wi);
    size_t G = round_up(N, L);
    return sycl::nd_range<1>{G, L};
  }

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
      sycl_default_stream->submit([&, nItems = this->nItems](sycl::handler &cgh) {
          cgh.parallel_for(get_reduction_range(nItems, rsltP),
                           sycl::reduction(rsltP, sycl::minimum<>(), sycl::property::reduction::initialize_to_identity{}),
                           [=](sycl::nd_item<1> idx, auto& min) {
                             const int i = idx.get_global_linear_id();
                             if (i < nItems) {
                               min.combine(data[i]);
                             }
                           });
        }).wait();
      sycl_default_stream->memcpy(&rslt,rsltP,sizeof(T)).wait(); // Copy result to host
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      sycl_default_stream->submit([&, nItems = this->nItems](sycl::handler &cgh) {
          cgh.parallel_for(get_reduction_range(nItems, devP),
                           sycl::reduction(devP, sycl::minimum<>(), sycl::property::reduction::initialize_to_identity{}),
                           [=](sycl::nd_item<1> idx, auto& min) {
                             const int i = idx.get_global_linear_id();
                             if (i < nItems) {
                               min.combine(data[i]);
                             }
                           });
        }).wait();
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
      sycl_default_stream->submit([&, nItems = this->nItems](sycl::handler &cgh) {
          cgh.parallel_for(get_reduction_range(nItems, rsltP),
                           sycl::reduction(rsltP, sycl::maximum<>(), sycl::property::reduction::initialize_to_identity{}),
                           [=](sycl::nd_item<1> idx, auto& max) {
                             const int i = idx.get_global_linear_id();
                             if (i < nItems) {
                               max.combine(data[i]);
                             }
                           });
        }).wait();
      sycl_default_stream->memcpy(&rslt,rsltP,sizeof(T)).wait(); // Copy result to host
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      sycl_default_stream->submit([&, nItems = this->nItems](sycl::handler &cgh) {
          cgh.parallel_for(get_reduction_range(nItems, devP),
                           sycl::reduction(devP, sycl::maximum<>(), sycl::property::reduction::initialize_to_identity{}),
                           [=](sycl::nd_item<1> idx, auto& max) {
                             const int i = idx.get_global_linear_id();
                             if (i < nItems) {
                               max.combine(data[i]);
                             }
                           });
        }).wait();
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
      sycl_default_stream->submit([&, nItems = this->nItems](sycl::handler &cgh) {
          cgh.parallel_for(get_reduction_range(nItems, rsltP),
                           sycl::reduction(rsltP, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
                           [=](sycl::nd_item<1> idx, auto& sum) {
                             const int i = idx.get_global_linear_id();
                             if (i < nItems) {
                               sum.combine(data[i]);
                             }
                           });
        }).wait();

      sycl_default_stream->memcpy(&rslt,rsltP,sizeof(T)).wait(); // Copy result to host
      return rslt;
    }
    void deviceReduce(T *data, T *devP) {
      sycl_default_stream->submit([&, nItems = this->nItems](sycl::handler &cgh) {
          cgh.parallel_for(get_reduction_range(nItems, rsltP),
                           sycl::reduction(rsltP, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
                           [=](sycl::nd_item<1> idx, auto& sum) {
                             const int i = idx.get_global_linear_id();
                             if (i < nItems) {
                               sum.combine(data[i]);
                             }
                           });
        }).wait();
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
