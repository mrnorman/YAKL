
#pragma once

#include "YAKL_alloc_free.h"

  // Initialize the YAKL framework
  inline void init() {
    yakl_mtx.lock();

    if ( ! isInitialized() ) {
      #if defined(YAKL_PROFILE) || defined(YAKL_AUTO_PROFILE)
        if (yakl_masterproc()) std::cout << "Using YAKL Timers via GPTL\n";
        GPTLinitialize();
      #endif
      bool use_pool = true;

      #ifdef YAKL_ARCH_SYCL
        auto asyncHandler = [&](sycl::exception_list eL) {
          for (auto& e : eL) {
            try {
              std::rethrow_exception(e);
            } catch (sycl::exception& e) {
              std::cout << e.what() << std::endl;
              std::cout << "fail" << std::endl;
              std::terminate();
            }
          }
        };

        sycl_default_stream = sycl::queue(sycl::gpu_selector{}, asyncHandler,
                                          sycl::property_list{sycl::property::queue::in_order{}});
        if (yakl_masterproc()) std::cout << "Running on "
                                         << sycl_default_stream.get_device().get_info<sycl::info::device::name>()
                                         << "\n";
      #endif

      yakl_is_initialized = true;

      // Check for pool allocator env var
      char * env = std::getenv("GATOR_DISABLE");
      if ( env != nullptr ) {
        std::string resp(env);
        if (resp == "yes" || resp == "YES" || resp == "1" || resp == "true" || resp == "TRUE" || resp == "T") {
          use_pool = false;
        }
      }

      if (use_pool) {

        std::function<void *( size_t )> alloc;
        std::function<void ( void * )>  dealloc;
        yakl::set_alloc_free(alloc , dealloc);
        pool.init(alloc,dealloc);

        yaklAllocDeviceFunc = [] (size_t bytes , char const *label) -> void * {
          return pool.allocate( bytes , label );
        };
        yaklFreeDeviceFunc  = [] (void *ptr , char const *label)              {
          pool.free( ptr , label );
        };

      } else {

        std::function<void *( size_t)> alloc;
        std::function<void ( void *)>  dealloc;
        set_alloc_free(alloc , dealloc);
        yaklAllocDeviceFunc = [=] (size_t bytes , char const *label) -> void * { return alloc(bytes); };
        yaklFreeDeviceFunc  = [=] (void *ptr , char const *label)              { dealloc(ptr); };
      }

      yaklAllocHostFunc = [] (size_t bytes , char const *label) -> void * { return malloc(bytes); };
      yaklFreeHostFunc  = [] (void *ptr , char const *label) { free(ptr); };

      #ifdef YAKL_ARCH_CUDA
        cudaMalloc(&functorBuffer,functorBufSize);
      #endif
      #ifdef YAKL_ARCH_SYCL
        functorBuffer = sycl::malloc_device(functorBufSize, sycl_default_stream);
        sycl_default_stream.memset(functorBuffer, 0, functorBufSize).wait();
      #endif

      #if defined(YAKL_ARCH_HIP)
        int id;
        hipGetDevice(&id);
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props,id);
        if (yakl_masterproc()) std::cout << props.name << std::endl;
      #endif

      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        if (yakl_masterproc()) std::cout << "INFORM: Automatically inserting fence() after every parallel_for" << std::endl;
      #endif
    } else {
      std::cerr << "WARNING: Calling yakl::initialize() when YAKL is already initialized. ";
    }

    yakl_mtx.unlock();
  } //

#pragma once

#include "YAKL_error.h"

namespace yakl {

#ifdef YAKL_ARCH_SYCL
  extern sycl::queue sycl_default_stream;
#endif

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
          void *ptr = sycl::malloc_shared(bytes,sycl_default_stream);
          sycl_default_stream.memset(ptr, 0, bytes).wait();
          check_last_error();
          sycl_default_stream.prefetch(ptr,bytes).wait();
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          sycl::free(ptr, sycl_default_stream);
          check_last_error();
        };
      #else
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr = sycl::malloc_device(bytes,sycl_default_stream);
          sycl_default_stream.memset(ptr, 0, bytes).wait();
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          sycl::free(ptr, sycl_default_stream);
          check_last_error();
          ptr = nullptr;
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

