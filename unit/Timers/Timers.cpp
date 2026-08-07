#include <atomic>
#include <chrono>
#include <sstream>
#include <thread>
#include <vector>
#include "YAKL.h"

void fail(std::string const &message) {
  Kokkos::abort(message.c_str());
}

int main(int argc, char **argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  yakl::init(yakl::InitConfig().set_pool_enabled(false));

  static_assert(yakl::Toney::Clock::is_steady);

  // Each recursive invocation needs an independent start time. The pause before the inner invocation distinguishes
  // the outer duration from the incorrect behavior that reused the inner invocation's start time.
  {
    yakl::Toney timer;
    timer.start("recursive");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    timer.start("recursive");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    timer.stop("recursive");
    timer.stop("recursive");

    int const timerIndex = timer.get_timer_id("recursive");
    if (timer.get_count("recursive") != 2 || timer.get_last_duration("recursive") < 0.015 ||
        timer.get_accumulated_duration("recursive") < timer.get_last_duration("recursive") ||
        ! timer.timers[timerIndex].child_indices.empty()) {
      fail("recursive same-label timing did not retain independent invocation state");
    }
  }

  // Stress shared records, recursive labels, nesting, queries, and reports from concurrent host threads.
  {
    yakl::Toney timer;
    timer.start("threaded");
    timer.start("thread child");
    timer.stop("thread child");
    timer.stop("threaded");

    int constexpr numThreads = 8;
    int constexpr numIterations = 500;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<bool> reportFailed{false};
    std::vector<std::thread> workers;
    workers.reserve(numThreads);
    for (int thread=0; thread < numThreads; thread++) {
      workers.emplace_back([&]() {
        ready.fetch_add(1,std::memory_order_release);
        while (! go.load(std::memory_order_acquire)) std::this_thread::yield();
        for (int iteration=0; iteration < numIterations; iteration++) {
          timer.start("threaded");
          timer.start("threaded");
          timer.start("thread child");
          if (iteration % 25 == 0 && timer.get_count("threaded") == 0) reportFailed.store(true);
          timer.stop("thread child");
          timer.stop("threaded");
          timer.stop("threaded");
        }
      });
    }
    std::thread reporter([&]() {
      ready.fetch_add(1,std::memory_order_release);
      while (! go.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int report=0; report < 16; report++) {
        std::ostringstream stream;
        timer.print(stream);
        if (stream.str().find("threaded") == std::string::npos) reportFailed.store(true);
      }
    });
    while (ready.load(std::memory_order_acquire) != numThreads+1) std::this_thread::yield();
    go.store(true,std::memory_order_release);
    for (auto &worker : workers) worker.join();
    reporter.join();

    size_t constexpr expectedThreaded = 1 + 2*numThreads*numIterations;
    size_t constexpr expectedChildren = 1 + numThreads*numIterations;
    int const threadedIndex = timer.get_timer_id("threaded");
    int const childIndex = timer.get_timer_id("thread child");
    bool activeStackRemains = false;
    for (auto const &entry : timer.active_stacks) activeStackRemains = activeStackRemains || ! entry.second.empty();
    if (reportFailed.load() || timer.get_count("threaded") != expectedThreaded ||
        timer.get_count("thread child") != expectedChildren || timer.timers[threadedIndex].child_indices.size() != 1 ||
        timer.timers[threadedIndex].child_indices[0] != childIndex || activeStackRemains) {
      fail("concurrent timer operations corrupted counts, nesting, queries, or reports");
    }
  }

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
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif
  return 0;
}
