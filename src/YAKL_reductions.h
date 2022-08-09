
#pragma once
// Included by YAKL.h

namespace yakl {

  template <class T, int memSpace, int RED> class ParallelReduction;

  int constexpr YAKL_REDUCTION_MIN  = 0;
  int constexpr YAKL_REDUCTION_MAX  = 1;
  int constexpr YAKL_REDUCTION_SUM  = 2;
  int constexpr YAKL_REDUCTION_PROD = 3;


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
      #ifdef YAKL_VERBOSE
        verbose_inform("Launching host reduction");
      #endif
      if constexpr        (RED == YAKL_REDUCTION_MIN) {
        for (int i=1; i<nItems; i++) { rslt = data[i] < rslt ? data[i] : rslt; }
      } else if constexpr (RED == YAKL_REDUCTION_MAX) {
        for (int i=1; i<nItems; i++) { rslt = data[i] > rslt ? data[i] : rslt; }
      } else if constexpr (RED == YAKL_REDUCTION_SUM) {
        for (int i=1; i<nItems; i++) { rslt += data[i]; }
      } else if constexpr (RED == YAKL_REDUCTION_PROD) {
        for (int i=1; i<nItems; i++) { rslt *= data[i]; }
      }
      return rslt;
    }
  };


  #ifdef YAKL_ARCH_HIP


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      Stream stream;
      public:
      ParallelReduction() { tmp = NULL; }
      ParallelReduction(int const nItems, Stream stream = Stream()) { tmp = NULL; setup(nItems,stream); }
      ~ParallelReduction() { finalize(); }
      void setup(int const nItems, Stream stream = Stream()) {
        #ifdef YAKL_AUTO_PROFILE
          timer_start("YAKL_internal_reduction_setup");
        #endif
        finalize();
        #ifdef YAKL_VERBOSE
          verbose_inform(std::string("Allocating device reduction scalar of size ")+std::to_string(sizeof(T))+std::string(" bytes"));
        #endif
        rsltP = (T *) alloc_device(sizeof(T),"Parallel Reduction Result"); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          hipcub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          hipcub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          hipcub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_PROD) {
          hipcub::DeviceReduce::Reduce(tmp, nTmp, rsltP, rsltP, nItems, YAKL_LAMBDA (T a,T b)->T {return a*b;} , (T) 1 );
        }
        #ifdef YAKL_VERBOSE
          verbose_inform("Allocating device reduction temporary storage of size "+std::to_string(nTmp)+std::string(" bytes"));
        #endif
        tmp = alloc_device(nTmp,"Parallel Reduction Temporary");       // Allocate temporary storage
        this->nItems = nItems;
        #ifdef YAKL_AUTO_PROFILE
          timer_stop("YAKL_internal_reduction_setup");
        #endif
        this->stream = stream;
      }
      void finalize() {
        if (tmp != NULL) {
          #ifdef YAKL_VERBOSE
            verbose_inform("Deallocating device reduction scalar");
          #endif
          free_device(rsltP,"Parallel Reduction Result");
          #ifdef YAKL_VERBOSE
            verbose_inform("Deallocating device reduction temporary storage");
          #endif
          free_device(tmp,"Parallel Reduction Temporary");
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        #ifdef YAKL_AUTO_PROFILE
          timer_start("YAKL_internal_reduction_apply");
        #endif
        T rslt;
        #ifdef YAKL_VERBOSE
          verbose_inform("Launching device reduction");
        #endif
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          hipcub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , stream.get_real_stream() ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          hipcub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , stream.get_real_stream() ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          hipcub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , stream.get_real_stream() ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_PROD) {
          hipcub::DeviceReduce::Reduce(tmp, nTmp, data, rsltP, nItems, YAKL_LAMBDA (T a,T b)->T {return a*b;} , (T) 1, stream.get_real_stream() );
        }
        #ifdef YAKL_VERBOSE
          verbose_inform("Initiating device to host memcpy of reduction scalar value of size "+std::to_string(sizeof(T))+std::string(" bytes"));
        #endif
        memcpy_device_to_host(&rslt , rsltP , 1 , stream);
        check_last_error();
        if (stream.is_default_stream()) { fence(); }
        else                            { stream.fence(); }
        #ifdef YAKL_AUTO_PROFILE
          timer_stop("YAKL_internal_reduction_apply");
        #endif
        return rslt;
      }
    };


  #elif defined(YAKL_ARCH_CUDA)


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      Stream stream;
      public:
      ParallelReduction() { tmp = NULL; }
      ParallelReduction(int const nItems, Stream stream = Stream()) { tmp = NULL; setup(nItems,stream); }
      ~ParallelReduction() { finalize(); }
      void setup(int const nItems, Stream stream = Stream()) {
        #ifdef YAKL_AUTO_PROFILE
          timer_start("YAKL_internal_reduction_setup");
        #endif
        finalize();
        #ifdef YAKL_VERBOSE
          verbose_inform(std::string("Allocating device reduction scalar of size ")+std::to_string(sizeof(T))+std::string(" bytes"));
        #endif
        rsltP = (T *) alloc_device(sizeof(T),"Parallel Reduction Result"); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          cub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          cub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          cub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        } else if constexpr (RED == YAKL_REDUCTION_PROD) {
          cub::DeviceReduce::Reduce(tmp, nTmp, rsltP, rsltP, nItems, YAKL_LAMBDA (T a,T b)->T {return a*b;} , (T) 1 );
        }
        #ifdef YAKL_VERBOSE
          verbose_inform("Allocating device reduction temporary storage of size "+std::to_string(nTmp)+std::string(" bytes"));
        #endif
        tmp = alloc_device(nTmp,"Parallel Reduction Temporary");       // Allocate temporary storage
        this->nItems = nItems;
        #ifdef YAKL_AUTO_PROFILE
          timer_stop("YAKL_internal_reduction_setup");
        #endif
        this->stream = stream;
      }
      void finalize() {
        if (tmp != NULL) {
          #ifdef YAKL_VERBOSE
            verbose_inform("Deallocating device reduction scalar");
          #endif
          free_device(rsltP,"Parallel Reduction Result");
          #ifdef YAKL_VERBOSE
            verbose_inform("Deallocating device reduction temporary storage");
          #endif
          free_device(tmp,"Parallel Reduction Temporary");
        }
        tmp = NULL;
      }
      T operator() (T *data) {
        #ifdef YAKL_AUTO_PROFILE
          timer_start("YAKL_internal_reduction_apply");
        #endif
        T rslt;
        #ifdef YAKL_VERBOSE
          verbose_inform("Launching device reduction");
        #endif
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          cub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , stream.get_real_stream() ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          cub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , stream.get_real_stream() ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          cub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , stream.get_real_stream() ); // Compute the reduction
        } else if constexpr (RED == YAKL_REDUCTION_PROD) {
          cub::DeviceReduce::Reduce(tmp, nTmp, data ,rsltP, nItems, YAKL_LAMBDA (T a,T b)->T {return a*b;} , (T) 1, stream.get_real_stream() );
        }
        #ifdef YAKL_VERBOSE
          verbose_inform("Initiating device to host memcpy of reduction scalar value of size "+std::to_string(sizeof(T))+std::string(" bytes"));
        #endif
        memcpy_device_to_host(&rslt , rsltP , 1 , stream);
        check_last_error();
        if (stream.is_default_stream()) { fence(); }
        else                            { stream.fence(); }
        #ifdef YAKL_AUTO_PROFILE
          timer_stop("YAKL_internal_reduction_apply");
        #endif
        return rslt;
      }
    };


  #elif defined(YAKL_ARCH_SYCL)

    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelReduction() { rsltP = nullptr; }
      ParallelReduction(int const nItems, Stream stream = Stream()) { rsltP = nullptr; setup(nItems); }
      ~ParallelReduction() { finalize(); }
      void setup(int const nItems, Stream stream = Stream()) {
        #ifdef YAKL_AUTO_PROFILE
          timer_start("YAKL_internal_reduction_setup");
        #endif
        finalize();
        #ifdef YAKL_VERBOSE
          verbose_inform(std::string("Allocating device reduction scalar of size ")+std::to_string(sizeof(T))+std::string(" bytes"));
        #endif
        rsltP = (T *) alloc_device(sizeof(T),"Parallel Reduction Result"); // Allocate device pointer for result
        this->nItems = nItems;
        #ifdef YAKL_AUTO_PROFILE
          timer_stop("YAKL_internal_reduction_setup");
        #endif
      }

      void finalize() {
        if(rsltP != nullptr) {
          #ifdef YAKL_VERBOSE
            verbose_inform("Deallocating device reduction scalar");
          #endif
          free_device(rsltP,"Parallel Reduction Result");
        }
        rsltP = nullptr;
      }
      T operator() (T *data) {
        #ifdef YAKL_AUTO_PROFILE
          timer_start("YAKL_internal_reduction_apply");
        #endif
        T rslt=0;
        #ifdef YAKL_VERBOSE
          verbose_inform("Launching device reduction");
        #endif
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
          } else if constexpr (RED == YAKL_REDUCTION_PROD) {
            cgh.parallel_for(sycl::range<1>(nItems),
                             sycl::reduction(rsltP, std::multiplies<>(),
                             sycl::property::reduction::initialize_to_identity{}),
                             [=] (sycl::id<1> idx, auto& prod) { prod.combine(data[idx]); });
          }
        });
        #ifdef YAKL_VERBOSE
          verbose_inform("Initiating device to host memcpy of reduction scalar value of size "+std::to_string(sizeof(T))+std::string(" bytes"));
        #endif
        memcpy_device_to_host(&rslt , rsltP , 1 );
        fence();
        #ifdef YAKL_AUTO_PROFILE
          timer_stop("YAKL_internal_reduction_apply");
        #endif
        return rslt;
      }
    };



  #elif defined(YAKL_ARCH_OPENMP)


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      int  nItems; // Number of items in the array that will be reduced
      public:
      ParallelReduction() {}
      ParallelReduction(int const nItems, Stream stream = Stream()) { this->nItems = nItems; }
      ~ParallelReduction() {}
      void setup(int nItems, Stream stream = Stream()) { this->nItems = nItems; }
      T operator() (T *data) {
        #ifdef YAKL_AUTO_PROFILE
          timer_start("YAKL_internal_reduction_apply");
        #endif
        T rslt = data[0];
        #ifdef YAKL_VERBOSE
          verbose_inform("Launching device reduction");
        #endif
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          #pragma omp parallel for reduction(min:rslt)
          for (int i=1; i<nItems; i++) { rslt = data[i] < rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          #pragma omp parallel for reduction(max:rslt)
          for (int i=1; i<nItems; i++) { rslt = data[i] > rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          #pragma omp parallel for reduction(+:rslt)
          for (int i=1; i<nItems; i++) { rslt += data[i]; }
        } else if constexpr (RED == YAKL_REDUCTION_PROD) {
          #pragma omp parallel for reduction(*:rslt)
          for (int i=1; i<nItems; i++) { rslt *= data[i]; }
        }
        #ifdef YAKL_AUTO_PROFILE
          timer_stop("YAKL_internal_reduction_apply");
        #endif
        return rslt;
      }
    };

    
  #else


    template <class T, int RED> class ParallelReduction<T,memDevice,RED> {
      int  nItems; // Number of items in the array that will be reduced
      public:
      ParallelReduction() {}
      ParallelReduction(int const nItems, Stream stream = Stream()) { this->nItems = nItems; }
      ~ParallelReduction() {}
      void setup(int nItems, Stream stream = Stream()) { this->nItems = nItems; }
      T operator() (T *data) {
        T rslt = data[0];
        #ifdef YAKL_VERBOSE
          verbose_inform("Launching device reduction");
        #endif
        if constexpr        (RED == YAKL_REDUCTION_MIN) {
          for (int i=1; i<nItems; i++) { rslt = data[i] < rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_MAX) {
          for (int i=1; i<nItems; i++) { rslt = data[i] > rslt ? data[i] : rslt; }
        } else if constexpr (RED == YAKL_REDUCTION_SUM) {
          for (int i=1; i<nItems; i++) { rslt += data[i]; }
        } else if constexpr (RED == YAKL_REDUCTION_PROD) {
          for (int i=1; i<nItems; i++) { rslt *= data[i]; }
        }
        return rslt;
      }
    };


  #endif


  template <class T, int memSpace> using ParallelMin  = ParallelReduction<T,memSpace,YAKL_REDUCTION_MIN >;
  template <class T, int memSpace> using ParallelMax  = ParallelReduction<T,memSpace,YAKL_REDUCTION_MAX >;
  template <class T, int memSpace> using ParallelSum  = ParallelReduction<T,memSpace,YAKL_REDUCTION_SUM >;
  template <class T, int memSpace> using ParallelProd = ParallelReduction<T,memSpace,YAKL_REDUCTION_PROD>;

}


