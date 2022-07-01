
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
  #endif // YAKL_ARCH_SYCL
}
