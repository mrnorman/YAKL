
#include <array>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
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
void write_attribute_type(yakl::SimplePNetCDF &nc, std::string const &name, T scalar, std::vector<T> const &values) {
  nc.writeGlobalAttribute(scalar,name+"_global");
  nc.writeVariableAttribute(values,"attribute_variable",name+"_variable");
}


template <class T>
void check_attribute_type(yakl::SimplePNetCDF const &nc, std::string const &name, T scalar,
                          std::vector<T> const &values) {
  if (nc.template readGlobalAttribute<T>(name+"_global") != scalar) {
    die("ERROR: incorrect PNetCDF global attribute for "+name);
  }
  if (nc.template readVariableAttribute<std::vector<T>>("attribute_variable",name+"_variable") != values) {
    die("ERROR: incorrect PNetCDF variable attribute for "+name);
  }
}


void test_attributes(MPI_Comm comm) {
  long const wide_long = std::numeric_limits<long>::digits > 32 ? (static_cast<long>(1) << 40) : 100000L;
  unsigned long const wide_ulong = std::numeric_limits<unsigned long>::digits > 32 ?
                                     (static_cast<unsigned long>(1) << 40) : 100000UL;
  {
    yakl::SimplePNetCDF nc(comm);
    nc.create("pnetcdf_attributes.nc",NC_CLOBBER | NC_64BIT_DATA);
    nc.create_var<int>("attribute_variable",{});
    write_attribute_type(nc,"char",'Q',{'a','b','c'});
    write_attribute_type(nc,"signed_char",static_cast<signed char>(-7),{static_cast<signed char>(-2),
                                                                       static_cast<signed char>(3)});
    write_attribute_type(nc,"unsigned_char",static_cast<unsigned char>(201),{static_cast<unsigned char>(1),
                                                                              static_cast<unsigned char>(250)});
    write_attribute_type(nc,"short",static_cast<short>(-1234),{static_cast<short>(-3),static_cast<short>(7)});
    write_attribute_type(nc,"unsigned_short",static_cast<unsigned short>(60000),
                         {static_cast<unsigned short>(2),static_cast<unsigned short>(50000)});
    write_attribute_type(nc,"int",-123456,{-8,0,19});
    write_attribute_type(nc,"unsigned_int",4000000000U,{0U,17U,3000000000U});
    write_attribute_type(nc,"long",-wide_long,{-wide_long,0L,wide_long});
    write_attribute_type(nc,"unsigned_long",wide_ulong,{0UL,wide_ulong,wide_ulong+5});
    write_attribute_type(nc,"long_long",-(static_cast<long long>(1) << 50),
                         {-(static_cast<long long>(1) << 48),0LL,static_cast<long long>(1) << 48});
    write_attribute_type(nc,"unsigned_long_long",static_cast<unsigned long long>(1) << 60,
                         {0ULL,static_cast<unsigned long long>(1) << 55});
    write_attribute_type(nc,"float",1.25f,{-2.5f,0.f,7.75f});
    write_attribute_type(nc,"double",-3.5,{-9.25,0.,11.5});
    write_attribute_type(nc,"bool",true,{false,true,true,false});
    nc.writeGlobalAttribute<std::string>("YAKL PNetCDF attributes","title");
    nc.writeVariableAttribute<std::string>("temperature","attribute_variable","description");
    nc.writeGlobalAttribute(std::vector<int>{2,4,8},"int_vector_global");
    nc.writeVariableAttribute(17,"attribute_variable","int_scalar_variable");
    nc.close();
  }

  {
    yakl::SimplePNetCDF nc(comm);
    nc.open("pnetcdf_attributes.nc",NC_NOWRITE);
    check_attribute_type(nc,"char",'Q',{'a','b','c'});
    check_attribute_type(nc,"signed_char",static_cast<signed char>(-7),{static_cast<signed char>(-2),
                                                                       static_cast<signed char>(3)});
    check_attribute_type(nc,"unsigned_char",static_cast<unsigned char>(201),{static_cast<unsigned char>(1),
                                                                              static_cast<unsigned char>(250)});
    check_attribute_type(nc,"short",static_cast<short>(-1234),{static_cast<short>(-3),static_cast<short>(7)});
    check_attribute_type(nc,"unsigned_short",static_cast<unsigned short>(60000),
                         {static_cast<unsigned short>(2),static_cast<unsigned short>(50000)});
    check_attribute_type(nc,"int",-123456,{-8,0,19});
    check_attribute_type(nc,"unsigned_int",4000000000U,{0U,17U,3000000000U});
    check_attribute_type(nc,"long",-wide_long,{-wide_long,0L,wide_long});
    check_attribute_type(nc,"unsigned_long",wide_ulong,{0UL,wide_ulong,wide_ulong+5});
    check_attribute_type(nc,"long_long",-(static_cast<long long>(1) << 50),
                         {-(static_cast<long long>(1) << 48),0LL,static_cast<long long>(1) << 48});
    check_attribute_type(nc,"unsigned_long_long",static_cast<unsigned long long>(1) << 60,
                         {0ULL,static_cast<unsigned long long>(1) << 55});
    check_attribute_type(nc,"float",1.25f,{-2.5f,0.f,7.75f});
    check_attribute_type(nc,"double",-3.5,{-9.25,0.,11.5});
    check_attribute_type(nc,"bool",true,{false,true,true,false});
    if (nc.readGlobalAttribute<std::string>("title") != "YAKL PNetCDF attributes" ||
        nc.readVariableAttribute<std::string>("attribute_variable","description") != "temperature") {
      die("ERROR: incorrect PNetCDF string attribute");
    }
    std::vector<int> global_vector;
    int variable_scalar = 0;
    nc.readGlobalAttribute(global_vector,"int_vector_global");
    nc.readVariableAttribute(variable_scalar,"attribute_variable","int_scalar_variable");
    if (global_vector != std::vector<int>{2,4,8} || variable_scalar != 17) {
      die("ERROR: incorrect PNetCDF output-reference attribute read");
    }
    nc.close();
  }
}


void test_dimension_order(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm,&rank);
  int constexpr fastest = 2;
  int constexpr middle  = 3;
  int constexpr slowest = 4;
  yakl::Array  <int ***,Kokkos::HostSpace> c_input("C input",slowest,middle,fastest);
  yakl::Array_F<int ***,Kokkos::HostSpace> f_input("F input",fastest,middle,slowest);
  for (int k=0; k < slowest; k++) {
    for (int j=0; j < middle; j++) {
      for (int i=0; i < fastest; i++) {
        int const value = 100*k + 10*j + i;
        c_input(k,j,i) = value;
        f_input(i+1,j+1,k+1) = value;
      }
    }
  }

  yakl::SimplePNetCDF nc(comm);
  nc.create("pnetcdf_dimension_order.nc",NC_CLOBBER | NC_64BIT_DATA);
  nc.create_dim("slowest",slowest);
  nc.create_dim("middle" ,middle );
  nc.create_dim("fastest",fastest);
  for (auto const &name : {"c_independent","f_independent","c_collective","f_collective"}) {
    nc.create_var<int>(name,{"slowest","middle","fastest"});
  }
  nc.enddef();

  yakl::Array  <int ***,Kokkos::HostSpace> c_independent("C independent output",slowest,middle,fastest);
  yakl::Array_F<int ***,Kokkos::HostSpace> f_independent("F independent output",fastest,middle,slowest);
  nc.begin_indep_data();
  if (rank == 0) {
    nc.write(c_input,"c_independent");
    nc.write(f_input,"f_independent");
    nc.read(c_independent,"f_independent");
    nc.read(f_independent,"c_independent");
  }
  MPI_Barrier(comm);
  nc.end_indep_data();

  auto c_device_input = c_input.createDeviceCopy();
  auto f_device_input = f_input.createDeviceCopy();
  auto c_device_output = c_input.createDeviceObject();
  auto f_device_output = f_input.createDeviceObject();
  nc.write_all(c_device_input,"c_collective",{0,0,0});
  nc.write_all(f_device_input,"f_collective",{0,0,0});
  nc.read_all(c_device_output,"f_collective",{0,0,0});
  nc.read_all(f_device_output,"c_collective",{0,0,0});
  auto c_collective = c_device_output.createHostCopy();
  auto f_collective = f_device_output.createHostCopy();
  nc.close();

  if (rank == 0) {
    for (int k=0; k < slowest; k++) {
      for (int j=0; j < middle; j++) {
        for (int i=0; i < fastest; i++) {
          int const expected = 100*k + 10*j + i;
          if (c_independent(k,j,i) != expected || f_independent(i+1,j+1,k+1) != expected ||
              c_collective(k,j,i) != expected || f_collective(i+1,j+1,k+1) != expected) {
            die("ERROR: PNetCDF Array/Array_F dimension order or data order is incorrect");
          }
        }
      }
    }
  }
}


void test_dimension_mismatch_failure(MPI_Comm comm) {
  yakl::SimplePNetCDF nc(comm);
  nc.create("pnetcdf_dimension_mismatch.nc",NC_CLOBBER | NC_64BIT_DATA);
  nc.create_dim("slowest",4);
  nc.create_dim("middle" ,3);
  nc.create_dim("fastest",2);
  nc.create_var<int>("values",{"slowest","middle","fastest"});
  nc.enddef();
  yakl::Array_F<int ***,Kokkos::HostSpace> wrong_order("wrong order",4,3,2);
  nc.read_all(wrong_order,"values",{0,0,0});
}


void test_second_unlimited_dimension_failure(MPI_Comm comm) {
  yakl::SimplePNetCDF nc(comm);
  nc.create("pnetcdf_unlimited_dimension.nc",NC_CLOBBER | NC_64BIT_DATA);
  nc.create_dim("record",NC_UNLIMITED);
  bool exception_thrown = false;
  try {
    nc.create_dim("second_record",NC_UNLIMITED);
  } catch (std::runtime_error const &) {
    exception_thrown = true;
  }
  if (!exception_thrown) die("ERROR: SimplePNetCDF accepted a second unlimited dimension");
  nc.close();
}


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
  if (argc > 1 && std::string(argv[1]) == "offset_overflow") {
    (void) yakl::pnetcdf_checked_mpi_offset(std::numeric_limits<size_t>::max());
    yakl::finalize();
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
  }
  {
    if (argc > 1 && std::string(argv[1]) == "dimension_mismatch") {
      test_dimension_mismatch_failure(MPI_COMM_WORLD);
    }
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

    test_dimension_order(MPI_COMM_WORLD);
    test_attributes(MPI_COMM_WORLD);
    test_second_unlimited_dimension_failure(MPI_COMM_WORLD);

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  MPI_Finalize();
  return 0;
}
