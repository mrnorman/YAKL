#include "YAKL.h"

extern "C" void gatorInit();
extern "C" void gatorFinalize();
extern "C" void *gatorAllocate(size_t bytes);
extern "C" void gatorDeallocate(void *ptr);

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  std::string const scenario = argv[1];

  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  if (scenario == "kokkos_yakl") yakl::init(yakl::InitConfig().set_pool_enabled(false));

  gatorInit();
  if (!Kokkos::is_initialized() || !yakl::get_yakl_instance().is_initialized()) {
    Kokkos::abort("ERROR: gatorInit did not preserve or initialize its dependencies");
  }

  int constexpr numThreads = 8;
  int constexpr numCycles  = 1000;
  std::vector<std::thread> threads;
  for (int thread=0; thread < numThreads; thread++) {
    threads.emplace_back([thread] {
      for (int cycle=0; cycle < numCycles; cycle++) {
        void *ptr = gatorAllocate(static_cast<size_t>((cycle+thread)%257)+1);
        if ((cycle+thread)%11 == 0) std::this_thread::yield();
        gatorDeallocate(ptr);
      }
    });
  }
  for (auto &thread : threads) thread.join();

  gatorFinalize();

  if (!Kokkos::is_initialized()) Kokkos::abort("ERROR: gatorFinalize finalized application-owned Kokkos");
  if (scenario == "kokkos") {
    if (yakl::get_yakl_instance().is_initialized()) Kokkos::abort("ERROR: gatorFinalize did not finalize owned YAKL");
  } else if (scenario == "kokkos_yakl") {
    if (!yakl::get_yakl_instance().is_initialized()) Kokkos::abort("ERROR: gatorFinalize finalized application-owned YAKL");
    yakl::finalize();
  } else {
    Kokkos::abort("ERROR: unknown ownership scenario");
  }
  Kokkos::finalize();
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif
  return 0;
}
