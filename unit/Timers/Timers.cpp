#include "YAKL.h"

void fail(std::string const &message) {
  Kokkos::abort(message.c_str());
}

int main() {
  Kokkos::initialize();
  yakl::init(yakl::InitConfig().set_pool_enabled(false));

  yakl::timer_start("manual timer");
  yakl::timer_stop ("manual timer");

  if constexpr (yakl::yakl_profile) {
    if (yakl::timer_get_count("manual timer") != 1 ||
        yakl::timer_get_accumulated_duration("manual timer") < 0 ||
        yakl::timer_get_last_duration("manual timer") < 0 ||
        yakl::timer_get_min_duration("manual timer") < 0 ||
        yakl::timer_get_max_duration("manual timer") < 0) {
      fail("enabled public timer wrappers returned invalid results");
    }
  } else {
    if (yakl::timer_get_count("manual timer") != 0 || ! yakl::get_yakl_instance().timer.timers.empty()) {
      fail("disabled public timer wrappers recorded a timer");
    }
  }

  yakl::parallel_for("automatic timer",1,KOKKOS_LAMBDA (size_t) {});
  if constexpr (yakl::yakl_auto_profile) {
    if (yakl::timer_get_count("automatic timer") != 1) fail("automatic parallel_for timer was not recorded");
  } else if constexpr (yakl::yakl_profile) {
    if (yakl::get_yakl_instance().timer.get_timer_id("automatic timer",false) != -1) {
      fail("automatic parallel_for timer was recorded without YAKL_AUTO_PROFILE");
    }
  }

  yakl::finalize();
  Kokkos::finalize();
  return 0;
}
