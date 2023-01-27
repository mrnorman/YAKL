
#pragma once
// Included by YAKL.h

namespace yakl {
  // This exists to create a queue for SYCL kernels and data transfers
  #ifdef YAKL_ARCH_SYCL
    // Triggered when asynchronous exceptions arise from the SYCL runtime
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

    // Singleton pattern to initialize a SYCL queue upon first instantiation
    class dev_mgr {
    public:
      sycl::queue &default_queue() {
        return *(_queues.get());
      }

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
        sycl::device dev(sycl::gpu_selector{});
        _devs = std::make_shared<sycl::device>(dev);
        _queues = std::make_shared<sycl::queue>(dev,
                                                asyncHandler,
                                                sycl::property_list{sycl::property::queue::in_order{}});
      }
      std::shared_ptr<sycl::device> _devs;
      std::shared_ptr<sycl::queue> _queues;
    };

    static inline sycl::queue &sycl_default_stream() {
      return dev_mgr::instance().default_queue();
    }

    #ifdef SYCL_DEVICE_COPYABLE
      template <class F>
      class SYCL_Functor_Wrapper {

        union UnionWrapper {
          F functor;
          UnionWrapper(){};
          UnionWrapper(F const & f) { std::memcpy( &functor , &f , sizeof(F) ); }
          UnionWrapper           (UnionWrapper const  & rhs) { std::memcpy( &functor , &rhs.functor , sizeof(F) ); }
          UnionWrapper           (UnionWrapper       && rhs) { std::memcpy( &functor , &rhs.functor , sizeof(F) ); }
          UnionWrapper &operator=(UnionWrapper const  & rhs) { std::memcpy( &functor , &rhs.functor , sizeof(F) ); return *this; }
          UnionWrapper &operator=(UnionWrapper       && rhs) { std::memcpy( &functor , &rhs.functor , sizeof(F) ); return *this; }
          void operator=(F const & f) { std::memcpy(&functor, &f, sizeof(F)); }
          ~UnionWrapper() { }
        };

        UnionWrapper union_wrapper;

       public:
        SYCL_Functor_Wrapper(F const & functor) { union_wrapper = functor; }
        F const & get_functor() const { return union_wrapper.functor; }
      };
    #endif

  #endif // YAKL_ARCH_SYCL
}


#if defined(YAKL_ARCH_SYCL) && defined(SYCL_DEVICE_COPYABLE)
template <typename F>
struct sycl::is_device_copyable<yakl::SYCL_Functor_Wrapper<F>> : std::true_type {};

template <typename F>
struct sycl::is_device_copyable<yakl::SYCL_Functor_Wrapper<F> const> : std::true_type {};
#endif

