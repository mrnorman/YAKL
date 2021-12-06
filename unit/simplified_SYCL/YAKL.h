
#pragma once

#include <iostream>
#include <iomanip>
#include <time.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include "stdlib.h"

#define YAKL_LAMBDA [=]
#define YAKL_INLINE inline
#define YAKL_DEVICE inline
#define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
#include <CL/sycl.hpp>

namespace yakl {

  typedef unsigned int index_t;

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
      if (gpu_devs[0].get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
	auto subDevs = gpu_devs[0].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
	_queues.push_back(std::make_shared<sycl::queue>(subDevs[0],
							asyncHandler,
							sycl::property_list{sycl::property::queue::in_order{}}));
      } else {
	_queues.push_back(std::make_shared<sycl::queue>(gpu_devs[0],
							asyncHandler,
							sycl::property_list{sycl::property::queue::in_order{}}));
      }
    }

    std::vector<std::shared_ptr<sycl::queue>> _queues;

    const unsigned int DEFAULT_DEVICE_ID = 0;
  };

  static inline sycl::queue &sycl_default_stream() {
    return dev_mgr::instance().default_queue();
  }

  // Memory space specifiers for YAKL Arrays
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr memStack  = 3;
  int constexpr memDefault = memDevice;


  int constexpr styleC       = 1;
  int constexpr styleFortran = 2;
  int constexpr styleDefault = styleC;


  extern bool yakl_is_initialized;

  inline void init() {
    yakl_is_initialized = true;
  }

  inline void finalize() {
    yakl_is_initialized = false;
  }


#include "YAKL_parallel_for_c.h"

#include "Array.h"

}



