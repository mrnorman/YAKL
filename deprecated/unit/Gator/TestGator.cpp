
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;

typedef float real;

typedef Array<real,1,memDevice,styleC> real1d;


void die(std::string msg) {
  yakl::yakl_throw(msg.c_str());
}


int main() {
  yakl::init();
  {
    ///////////////////////////////////////////////////////////
    // Test zero size allocation
    ///////////////////////////////////////////////////////////
    real1d zerovar("zerovar",0);
    if (zerovar.data() != nullptr) { die("zero-sized allocation did not return nullptr"); }

    ///////////////////////////////////////////////////////////
    // Test pool growth
    ///////////////////////////////////////////////////////////
    real1d large1("large1",1024*1024*1024/4);
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
    real1d large2("large2",1024*1024*1024/4);
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
    real1d large3("large3",1024*1024*1024/4);
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
    large3 = real1d();
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
    large3 = real1d("large3",1024*1024*1024/4);
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
    large3 = real1d();
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
    real1d small4("small1",1024*1024);
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
    real1d large4("large4",1024*1024*1024/4);
    std::cout << "*** High Water Mark: "           << yakl::get_pool().get_high_water_mark() << "\n";
    std::cout << "*** Number of pools: "           << yakl::get_pool().get_num_pools() << "\n";
    std::cout << "*** Total pool capacity: "       << yakl::get_pool().get_pool_capacity() << "\n";
    std::cout << "*** Bytes currently allocated: " << yakl::get_pool().get_bytes_currently_allocated() << "\n";
    std::cout << "*** Pool efficiency: "           << yakl::get_pool().get_pool_space_efficiency() << "\n";
    std::cout << "*** Number of Allocations: "     << yakl::get_pool().get_num_allocs() << "\n\n";
  }
  yakl::finalize();

  // yakl::init( yakl::InitConfig().set_pool_enabled(false) );
  // {
  //   std::cout << "Pool not enabled here" << std::endl;
  // }
  // yakl::finalize();

  // yakl::init( yakl::InitConfig().set_pool_enabled(true).set_pool_initial_mb(17).set_pool_grow_mb(10) );
  // {
  //   real1d large1("large1",17*1024*1024/4);
  //   real1d small1("large1",1*1024/4);
  //   // real1d large2("large2",11*1024*1024/4); // This will cause an error
  // }
  // yakl::finalize();

}

