
#include <array>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include "YAKL.h"
#include "YAKL_pnetcdf.h"


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


template <class T>
concept HasPNetCDFOverloads = requires(int ncid, int varid, MPI_Offset const *start, MPI_Offset const *count,
                                       T const *input, T *output) {
  yakl::pnetcdf_put_var(ncid,varid,input);
  yakl::pnetcdf_put_var1(ncid,varid,input);
  yakl::pnetcdf_put_vara(ncid,varid,start,count,input);
  yakl::pnetcdf_put_vara_all(ncid,varid,start,count,input);
  yakl::pnetcdf_get_var(ncid,varid,output);
  yakl::pnetcdf_get_var1(ncid,varid,output);
  yakl::pnetcdf_get_vara(ncid,varid,start,count,output);
  yakl::pnetcdf_get_vara_all(ncid,varid,start,count,output);
};


static_assert(HasPNetCDFOverloads<long>);
static_assert(HasPNetCDFOverloads<unsigned long>);
static_assert(HasPNetCDFOverloads<char>);
static_assert(HasPNetCDFOverloads<bool>);


template <class T>
void test_pnetcdf_type(yakl::SimplePNetCDF &nc, std::string const &name, std::array<T,4> const &expected,
                       T scalar_expected) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  nc.create_var<T>(name+"_scalar",{});
  nc.create_var<T>(name+"_whole",{"type_element"});
  nc.create_var<T>(name+"_collective",{"type_element"});
  nc.enddef();

  yakl::Array<T *,Kokkos::HostSpace> input("input",expected.size());
  yakl::Array<T *,Kokkos::HostSpace> whole_output("whole_output",expected.size());
  yakl::Array<T *,Kokkos::HostSpace> collective_output("collective_output",expected.size());
  for (size_t i=0; i < expected.size(); i++) input(i) = expected[i];

  nc.begin_indep_data();
  if (rank == 0) {
    nc.write(scalar_expected,name+"_scalar");
    nc.write(input,name+"_whole");
    T scalar_output{};
    nc.read(scalar_output,name+"_scalar");
    nc.read(whole_output,name+"_whole");
    if (scalar_output != scalar_expected) die("ERROR: Incorrect PNetCDF scalar value for "+name);
    for (size_t i=0; i < expected.size(); i++) {
      if (whole_output(i) != expected[i]) die("ERROR: Incorrect PNetCDF whole-variable value for "+name);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  nc.end_indep_data();

  nc.write_all(input,name+"_collective",{0});
  nc.read_all(collective_output,name+"_collective",{0});
  for (size_t i=0; i < expected.size(); i++) {
    if (collective_output(i) != expected[i]) die("ERROR: Incorrect PNetCDF collective value for "+name);
  }
}


int main(int argc , char **argv) {
  static_assert(!std::is_copy_constructible_v<yakl::SimplePNetCDF>);
  static_assert(!std::is_copy_assignable_v<yakl::SimplePNetCDF>);
  static_assert( std::is_nothrow_move_constructible_v<yakl::SimplePNetCDF>);
  static_assert( std::is_nothrow_destructible_v<yakl::SimplePNetCDF>);

  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
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
      yakl::SimplePNetCDF original(MPI_COMM_WORLD);
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info,"romio_no_indep_rw"   ,"true"   );
      MPI_Info_set(info,"nc_header_align_size","1048576");
      MPI_Info_set(info,"nc_var_align_size"   ,"1048576");
      original.create(file_name , NC_CLOBBER | NC_64BIT_DATA , info );
      yakl::SimplePNetCDF moved(std::move(original));
      yakl::SimplePNetCDF nc(MPI_COMM_WORLD);
      nc = std::move(moved);
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

    // Exercise every PNetCDF path advertised by the arithmetic templates for the previously missing types.
    {
      long const wide_long = std::numeric_limits<long>::digits > 32 ? (static_cast<long>(1) << 40) : 100000L;
      unsigned long const wide_ulong = std::numeric_limits<unsigned long>::digits > 32 ?
                                       (static_cast<unsigned long>(1) << 40) : 100000UL;
      yakl::SimplePNetCDF nc(MPI_COMM_WORLD);
      nc.create("pnetcdf_arithmetic_types.nc",NC_CLOBBER | NC_64BIT_DATA);
      nc.create_dim("type_element",4);
      test_pnetcdf_type(nc,"long",{-wide_long,-1L,0L,wide_long},-wide_long);
      test_pnetcdf_type(nc,"unsigned_long",{0UL,1UL,wide_ulong,wide_ulong+7},wide_ulong+7);
      test_pnetcdf_type(nc,"char",{'\0','A','z',static_cast<char>(127)},'z');
      test_pnetcdf_type(nc,"bool",{false,true,true,false},true);
      nc.close();
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  MPI_Finalize();
  return 0;
}
