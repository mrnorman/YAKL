
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
  std::cerr << msg << std::endl;
  exit(-1);
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
    real1d large2("large2",1024*1024*1024/4);
    real1d large3("large3",1024*1024*1024/4);
    large3 = real1d();
    large3 = real1d("large3",1024*1024*1024/4);
    large3 = real1d();
    real1d small4("small1",1024*1024);
    real1d large4("large4",1024*1024*1024/4);


  }
  yakl::finalize();

  yakl::init( yakl::InitConfig().set_pool_enabled(false) );
  {
    std::cout << "Pool not enabled here" << std::endl;
  }
  yakl::finalize();

  yakl::init( yakl::InitConfig().set_pool_enabled(true).set_pool_initial_mb(17).set_pool_grow_mb(10) );
  {
    real1d large1("large1",17*1024*1024/4);
    real1d small1("large1",1*1024/4);
    // real1d large2("large2",11*1024*1024/4); // This will cause an error
  }
  yakl::finalize();
}

