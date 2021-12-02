
#pragma once

#include "YAKL_error.h"

namespace yakl {

  #ifdef YAKL_ARCH_SYCL
  auto asyncHandler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cerr << "Caught asynchronous SYCL exception:" << std::endl
        << e.what() << std::endl
        << "Exception caught at file:" << __FILE__
        << ", line:" << __LINE__ << std::endl;
      }
    }
  };

  class dev_mgr {
  public:
    sycl::queue &default_queue() {
      check_id(DEFAULT_DEVICE_ID);
      return *_queues[DEFAULT_DEVICE_ID];
    }
    unsigned int device_count() { return _devs.size(); }

    /// Returns the instance of device manager singleton.
    static dev_mgr &instance() {
      static dev_mgr d_m;
      return d_m;
    }
    dev_mgr(const dev_mgr &) = delete;
    dev_mgr &operator=(const dev_mgr &) = delete;
    dev_mgr(dev_mgr &&) = delete;
    dev_mgr &operator=(dev_mgr &&) = delete;

  private:
    dev_mgr() {
      sycl::platform platform(sycl::gpu_selector{});
      auto gpu_devs = platform.get_devices(sycl::info::device_type::gpu);
      for (auto &dev : gpu_devs) {
        if (dev.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
          auto subDevs = dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
          for (auto &sub_dev : subDevs) {
            _devs.push_back(std::make_shared<sycl::device>(sub_dev));
            _ctxts.push_back(std::make_shared<sycl::context>(sub_dev));
            _queues.push_back(std::make_shared<sycl::queue>(*(_ctxts.back()),
                                                            sub_dev,
                                                            asyncHandler,
                                                            sycl::property_list{sycl::property::queue::in_order{}}));
          }
        } else {
          _devs.push_back(std::make_shared<sycl::device>(dev));
          _ctxts.push_back(std::make_shared<sycl::context>(dev));
          _queues.push_back(std::make_shared<sycl::queue>(*(_ctxts.back()),
                                                          dev,
                                                          asyncHandler,
                                                          sycl::property_list{sycl::property::queue::in_order{}}));
        }
      }
    }
    void check_id(unsigned int id) const {
      if (id >= _devs.size()) {
        yakl_throw("ERROR: invalid SYCL device id");
      }
    }
    std::vector<std::shared_ptr<sycl::device>> _devs;
    std::vector<std::shared_ptr<sycl::context>> _ctxts;
    std::vector<std::shared_ptr<sycl::queue>> _queues;

    const unsigned int DEFAULT_DEVICE_ID = 0;
  };

  static inline sycl::queue &sycl_default_stream() {
    return dev_mgr::instance().default_queue();
  }
#endif // YAKL_ARCH_SYCL

  inline void set_alloc_free(std::function<void *( size_t )> &alloc , std::function<void ( void * )> &dealloc) {
    #if   defined(YAKL_ARCH_CUDA)
      #if defined (YAKL_MANAGED_MEMORY)
        alloc   = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          cudaMallocManaged(&ptr,bytes);
          check_last_error();
          cudaMemPrefetchAsync(ptr,bytes,0);
          check_last_error();
          #ifdef _OPENMP45
            omp_target_associate_ptr(ptr,ptr,bytes,0,0);
          #endif
          #ifdef _OPENACC
            acc_map_data(ptr,ptr,bytes);
          #endif
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
          check_last_error();
        };
      #else
        alloc   = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          cudaMalloc(&ptr,bytes);
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
          check_last_error();
        };
      #endif
    #elif defined(YAKL_ARCH_HIP)
      #if defined (YAKL_MANAGED_MEMORY)
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          hipMallocHost(&ptr,bytes);
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          hipFree(ptr);
          check_last_error();
        };
      #else
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          hipMalloc(&ptr,bytes);
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          hipFree(ptr);
          check_last_error();
        };
      #endif
    #elif defined (YAKL_ARCH_SYCL)
      #if defined (YAKL_MANAGED_MEMORY)
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          yakl::sycl_default_stream().wait();
          void *ptr = sycl::malloc_shared(bytes,yakl::sycl_default_stream());
          yakl::sycl_default_stream().memset(ptr, 0, bytes);
          yakl::sycl_default_stream().wait();
          check_last_error();
          yakl::sycl_default_stream().prefetch(ptr,bytes);
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          yakl::sycl_default_stream().wait();
          sycl::free(ptr, yakl::sycl_default_stream());
          yakl::sycl_default_stream().wait();
          check_last_error();
        };
      #else
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          yakl::sycl_default_stream().wait();
          void *ptr = sycl::malloc_device(bytes,yakl::sycl_default_stream());
          yakl::sycl_default_stream().memset(ptr, 0, bytes);
          yakl::sycl_default_stream().wait();
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          yakl::sycl_default_stream().wait();
          sycl::free(ptr, yakl::sycl_default_stream());
          yakl::sycl_default_stream().wait();
          check_last_error();
          // ptr = nullptr;
        };
      #endif
    #elif defined(YAKL_ARCH_OPENMP45)
      alloc = [] ( size_t bytes ) -> void* {
        if (bytes == 0) return nullptr;
        void *ptr;
        int device;
        device = omp_get_default_device();
        ptr = omp_target_alloc(bytes,device);
        //check does nothing
        check_last_error();
        return ptr;
      };
      dealloc = [] (void *ptr) {
        int device;
        device = omp_get_default_device();
        omp_target_free(ptr,device);
        //check does nothing
        check_last_error();
      };
    #else
      alloc   = [] ( size_t bytes ) -> void* { if (bytes == 0) return nullptr; return ::malloc(bytes); };
      dealloc = [] ( void *ptr ) { ::free(ptr); };
    #endif
  }

}
