
#include <iostream>
#include "YAKL.h"
#include "YAKL_pnetcdf.h"


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main(int argc , char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    // Write so that d1 is always the fastest varying and ordered from there
    int constexpr nx = 128;
    int constexpr ny = 128;
    int constexpr nz = 128;
    yakl::Array<double ***,yakl::DeviceSpace> arr("a0",nz,ny,nx);
    arr = 2;
    auto arr_read = arr.createDeviceObject();

    std::string file_name = "testyMcTestFace.nc";

    // This block is the writing phase
    {
      yakl::SimplePNetCDF nc(MPI_COMM_WORLD);
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info,"romio_no_indep_rw"   ,"true"   );
      MPI_Info_set(info,"nc_header_align_size","1048576");
      MPI_Info_set(info,"nc_var_align_size"   ,"1048576");
      nc.create(file_name , NC_CLOBBER | NC_64BIT_DATA , info );
      nc.create_dim("x",nx);
      nc.create_dim("y",ny);
      nc.create_dim("z",nz);
      nc.create_var<double>( "arr" , {"z","y","x"} );
      nc.enddef();
      nc.write_all(arr,"arr",std::vector<MPI_Offset>({0,0,0}));
      nc.close();
    }

    // This block is the reading phase
    {
      yakl::SimplePNetCDF nc(MPI_COMM_WORLD);
      nc.open(file_name,NC_NOWRITE);
      nc.read_all(arr_read,"arr",std::vector<MPI_Offset>({0,0,0})); // Read dry density
      nc.close();
      using yakl::componentwise::operator-;
      using yakl::componentwise::operator>;
      using yakl::intrinsics::count;
      if ( count( (arr_read-2) > 0 ) > 0 ) Kokkos::abort("ERROR: Incorrect data in read");
    }

  }
  yakl::finalize();
  Kokkos::finalize(); 
  MPI_Finalize();
  return 0;
}

