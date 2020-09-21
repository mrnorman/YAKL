
#pragma once


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
