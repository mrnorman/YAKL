#include <cstdio>
#include <fstream>
#include <sstream>
#include "YAKL.h"

void fail(std::string const & message) {
  Kokkos::abort(message.c_str());
}

int main(int argc, char ** argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();

  int rank = 0;
  #ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  #endif
  std::string const filename = "yakl_auto_printf."+std::to_string(rank)+".out";
  std::remove(filename.c_str());

  if constexpr (yakl::yakl_auto_printf) {
    yakl::init(yakl::InitConfig().set_pool_enabled(false));
    {
      yakl::Array<int *,yakl::DeviceSpace> default_constructed;
      yakl::Array<int *,yakl::DeviceSpace> left("left",4);
      yakl::Array<int *,yakl::DeviceSpace> right("right",4);
      yakl::Array<int[4],yakl::DeviceSpace> static_extent("static extent");
      yakl::Array<int *,yakl::DeviceSpace> view_allocated(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,"view allocated"),4);
      yakl::Array<int *,Kokkos::HostSpace> host("host",4);
      yakl::Array_F<int *,yakl::DeviceSpace> fortran("fortran",4);
      {
        yakl::Array<int *,yakl::DeviceSpace> paired("paired",1);
        auto alias = paired;
      }
      {
        yakl::Array_F<int *,yakl::DeviceSpace> paired_f("paired F",1);
        auto alias_f = paired_f;
      }
      left = 1;
      right = 2;
      yakl::parallel_for("labeled kernel",4,KOKKOS_LAMBDA (int i) { left(i) += right(i); });
      yakl::parallel_for("",0,KOKKOS_LAMBDA (int) {});
      auto const sum = yakl::intrinsics::sum(left);
      using yakl::componentwise::operator+;
      auto result = left + right;
      if (sum != 12 || yakl::intrinsics::sum(result) != 20) fail("YAKL_AUTO_PRINTF operations returned bad results");

      std::ifstream input(filename);
      std::stringstream contents;
      contents << input.rdbuf();
      std::string const output = contents.str();
      auto const kernel_begin   = output.find("BEGIN yakl::parallel_for: labeled kernel rank=1 extents=4 iterations=4");
      auto const kernel_end     = output.find("END yakl::parallel_for: labeled kernel rank=1 extents=4 iterations=4");
      auto const intrinsic_begin = output.find("BEGIN yakl::intrinsics::sum: left rank=1 extents=4 elements=4");
      auto const intrinsic_end   = output.find("END yakl::intrinsics::sum: left rank=1 extents=4 elements=4");
      auto const component_begin = output.find("BEGIN yakl::componentwise::binary: left, right rank=1 extents=4 elements=4");
      auto const component_end   = output.find("END yakl::componentwise::binary: left, right rank=1 extents=4 elements=4");
      auto const paired_alloc   = output.find("ALLOC yakl::Array: paired rank=1 extents=1 elements=1 bytes=4 space=device");
      auto const paired_free    = output.find("FREE yakl::Array: paired rank=1 extents=1 elements=1 bytes=4 space=device");
      auto const paired_f_alloc = output.find("ALLOC yakl::Array_F: paired F rank=1 extents=1 bounds=1:1");
      auto const paired_f_free  = output.find("FREE yakl::Array_F: paired F rank=1 extents=1 bounds=1:1");
      if (output.find("START rank="+std::to_string(rank)+" size=") == std::string::npos ||
          output.find(" backend=") == std::string::npos || output.find(" time=") == std::string::npos ||
          output.find("ALLOC yakl::Array: left rank=1 extents=4 elements=4 bytes=16 space=device") == std::string::npos ||
          output.find("ALLOC yakl::Array: static extent rank=1 extents=4") == std::string::npos ||
          output.find("ALLOC yakl::Array: view allocated rank=1 extents=4") == std::string::npos ||
          output.find("ALLOC yakl::Array: host rank=1 extents=4 elements=4 bytes=16 space=host") == std::string::npos ||
          output.find("ALLOC yakl::Array_F: fortran rank=1 extents=4 bounds=1:4") == std::string::npos ||
          paired_alloc == std::string::npos || paired_free == std::string::npos || paired_alloc >= paired_free ||
          output.find("FREE yakl::Array: paired",paired_free+1) != std::string::npos ||
          paired_f_alloc == std::string::npos || paired_f_free == std::string::npos || paired_f_alloc >= paired_f_free ||
          output.find("FREE yakl::Array_F: paired F",paired_f_free+1) != std::string::npos ||
          output.find("BEGIN yakl::Array::operator=scalar: left\n") == std::string::npos ||
          output.find("END yakl::Array::operator=scalar: left\n") == std::string::npos ||
          kernel_begin == std::string::npos || kernel_end == std::string::npos || kernel_begin >= kernel_end ||
          output.find("BEGIN yakl::parallel_for rank=1 iterations=0") == std::string::npos ||
          output.find("END yakl::parallel_for rank=1 iterations=0") == std::string::npos ||
          intrinsic_begin == std::string::npos || intrinsic_end == std::string::npos || intrinsic_begin >= intrinsic_end ||
          component_begin == std::string::npos || component_end == std::string::npos || component_begin >= component_end) {
        fail("YAKL_AUTO_PRINTF output is missing expected flushed operation lines");
      }
    }
    yakl::finalize();

    {
      std::ofstream stale(filename,std::ios::out | std::ios::trunc);
      stale << "stale" << std::endl;
    }
    yakl::init(yakl::InitConfig().set_pool_enabled(false));
    std::ifstream truncated(filename);
    std::stringstream restarted_contents;
    restarted_contents << truncated.rdbuf();
    auto const restarted_output = restarted_contents.str();
    if (restarted_output.find("stale") != std::string::npos || restarted_output.find("START rank=") == std::string::npos) {
      fail("YAKL_AUTO_PRINTF did not truncate and restart its output on re-init");
    }
    yakl::finalize();
    std::ifstream finalized(filename);
    std::stringstream finalized_contents;
    finalized_contents << finalized.rdbuf();
    if (finalized_contents.str().find("NORMAL_END\n") == std::string::npos) {
      fail("YAKL_AUTO_PRINTF output is missing the normal finalization marker");
    }
    std::remove(filename.c_str());
  } else {
    yakl::init(yakl::InitConfig().set_pool_enabled(false));
    yakl::parallel_for("disabled",1,KOKKOS_LAMBDA (int) {});
    yakl::finalize();
    std::ifstream input(filename);
    if (input.good()) fail("YAKL_AUTO_PRINTF-disabled build created an output file");
  }

  Kokkos::finalize();
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif
  return 0;
}
