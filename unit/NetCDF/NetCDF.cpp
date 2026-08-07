
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include "YAKL.h"
#include "YAKL_netcdf.h"

using yakl::Array;
using yakl::Array_F;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


template <class T>
void write_attribute_type(yakl::SimpleNetCDF &nc, std::string const &name, T scalar, std::vector<T> const &values) {
  nc.writeGlobalAttribute(scalar,name+"_global");
  nc.writeVariableAttribute(values,"attribute_variable",name+"_variable");
}


template <class T>
void check_attribute_type(yakl::SimpleNetCDF const &nc, std::string const &name, T scalar,
                          std::vector<T> const &values) {
  if (nc.template readGlobalAttribute<T>(name+"_global") != scalar) {
    die("ERROR: incorrect netCDF global attribute for "+name);
  }
  if (nc.template readVariableAttribute<std::vector<T>>("attribute_variable",name+"_variable") != values) {
    die("ERROR: incorrect netCDF variable attribute for "+name);
  }
}


void test_attributes() {
  long const wide_long = std::numeric_limits<long>::digits > 32 ? (static_cast<long>(1) << 40) : 100000L;
  unsigned long const wide_ulong = std::numeric_limits<unsigned long>::digits > 32 ?
                                     (static_cast<unsigned long>(1) << 40) : 100000UL;
  {
    yakl::SimpleNetCDF nc;
    nc.create("netcdf_attributes.nc");
    nc.write(0,"attribute_variable");
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
    nc.writeGlobalAttribute<std::string>("YAKL netCDF attributes","title");
    nc.writeVariableAttribute<std::string>("temperature","attribute_variable","description");
    nc.writeGlobalAttribute(std::vector<int>{2,4,8},"int_vector_global");
    nc.writeVariableAttribute(17,"attribute_variable","int_scalar_variable");
  }

  {
    yakl::SimpleNetCDF nc;
    nc.open("netcdf_attributes.nc");
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
    if (nc.readGlobalAttribute<std::string>("title") != "YAKL netCDF attributes" ||
        nc.readVariableAttribute<std::string>("attribute_variable","description") != "temperature") {
      die("ERROR: incorrect netCDF string attribute");
    }
    std::vector<int> global_vector;
    int variable_scalar = 0;
    nc.readGlobalAttribute(global_vector,"int_vector_global");
    nc.readVariableAttribute(variable_scalar,"attribute_variable","int_scalar_variable");
    if (global_vector != std::vector<int>{2,4,8} || variable_scalar != 17) {
      die("ERROR: incorrect netCDF output-reference attribute read");
    }
  }
}


void test_dimension_order() {
  int constexpr fastest = 2;
  int constexpr middle  = 3;
  int constexpr slowest = 4;
  Array  <int ***,Kokkos::HostSpace> c_input("C input",slowest,middle,fastest);
  Array_F<int ***,Kokkos::HostSpace> f_input("F input",fastest,middle,slowest);
  for (int k=0; k < slowest; k++) {
    for (int j=0; j < middle; j++) {
      for (int i=0; i < fastest; i++) {
        int const value = 100*k + 10*j + i;
        c_input(k,j,i) = value;
        f_input(i+1,j+1,k+1) = value;
      }
    }
  }

  {
    yakl::SimpleNetCDF nc;
    nc.create("netcdf_dimension_order.nc");
    nc.write(c_input,"from_c",{"slowest","middle","fastest"});
    nc.write(f_input,"from_f",{"fastest","middle","slowest"});
  }

  Array  <int ***,Kokkos::HostSpace> c_from_f("C read from F",slowest,middle,fastest);
  Array_F<int ***,Kokkos::HostSpace> f_from_c("F read from C",fastest,middle,slowest);
  auto c_from_f_device = c_from_f.createDeviceObject();
  auto f_from_c_device = f_from_c.createDeviceObject();
  {
    yakl::SimpleNetCDF nc;
    nc.open("netcdf_dimension_order.nc");
    nc.read(c_from_f_device,"from_f");
    nc.read(f_from_c_device,"from_c");
  }
  c_from_f_device.deep_copy_to(c_from_f);
  f_from_c_device.deep_copy_to(f_from_c);
  for (int k=0; k < slowest; k++) {
    for (int j=0; j < middle; j++) {
      for (int i=0; i < fastest; i++) {
        int const expected = 100*k + 10*j + i;
        if (c_from_f(k,j,i) != expected || f_from_c(i+1,j+1,k+1) != expected) {
          die("ERROR: netCDF Array/Array_F dimension order or data order is incorrect");
        }
      }
    }
  }
}


void test_dimension_mismatch_failure() {
  Array<int ***,Kokkos::HostSpace> input("input",4,3,2);
  {
    yakl::SimpleNetCDF nc;
    nc.create("netcdf_dimension_mismatch.nc");
    nc.write(input,"values",{"slowest","middle","fastest"});
  }
  Array_F<int ***,Kokkos::HostSpace> wrong_order("wrong order",4,3,2);
  yakl::SimpleNetCDF nc;
  nc.open("netcdf_dimension_mismatch.nc");
  nc.read(wrong_order,"values");
}


void test_second_unlimited_dimension_failure() {
  yakl::SimpleNetCDF nc;
  nc.create("netcdf_unlimited_dimension.nc");
  nc.createDim("record");
  bool exception_thrown = false;
  try {
    nc.createDim("second_record");
  } catch (std::runtime_error const &) {
    exception_thrown = true;
  }
  if (!exception_thrown || nc.dimExists("second_record")) {
    die("ERROR: SimpleNetCDF accepted a second unlimited dimension");
  }
}


void test_unsigned_long_and_move_ownership() {
  static_assert(!std::is_copy_constructible_v<yakl::SimpleNetCDF>);
  static_assert(!std::is_copy_assignable_v<yakl::SimpleNetCDF>);
  static_assert( std::is_nothrow_move_constructible_v<yakl::SimpleNetCDF>);
  static_assert( std::is_nothrow_move_assignable_v<yakl::SimpleNetCDF>);

  std::string const fileName = "unsigned_long.nc";
  unsigned long base = 17;
  if constexpr (sizeof(unsigned long) > sizeof(unsigned int)) {
    base = static_cast<unsigned long>(std::numeric_limits<unsigned int>::max()) + 37UL;
  }

  Array<unsigned long *,Kokkos::HostSpace> values("unsigned long values",4);
  for (size_t i=0; i < values.size(); i++) values(i) = base + static_cast<unsigned long>(11*i);
  unsigned long const scalar = base + 101UL;

  {
    yakl::SimpleNetCDF original;
    original.create(fileName);
    yakl::SimpleNetCDF moved(std::move(original));
    yakl::SimpleNetCDF nc;
    nc = std::move(moved);
    nc.write(values,"values",{"nvalues"});
    nc.write(scalar,"scalar");
    nc.write1(values,"record_values",{"nvalues"},0,"record");
    nc.write1(scalar,"record_scalar",0,"record");
  }

  {
    yakl::SimpleNetCDF original;
    original.open(fileName);
    yakl::SimpleNetCDF nc(std::move(original));
    Array<unsigned long *,Kokkos::HostSpace> valuesRead("unsigned long values read",4);
    Array<unsigned long **,Kokkos::HostSpace> recordValues("unsigned long record values read",1,4);
    Array<unsigned long *,Kokkos::HostSpace> recordScalar("unsigned long record scalar read",1);
    unsigned long scalarRead = 0;
    nc.read(valuesRead,"values");
    nc.read(scalarRead,"scalar");
    nc.read(recordValues,"record_values");
    nc.read(recordScalar,"record_scalar");
    for (size_t i=0; i < values.size(); i++) {
      if (valuesRead(i) != values(i) || recordValues(0,i) != values(i)) {
        die("ERROR: unsigned long netCDF array data was corrupted");
      }
    }
    if (scalarRead != scalar || recordScalar(0) != scalar) {
      die("ERROR: unsigned long netCDF scalar data was corrupted");
    }
  }
}


int main(int argc, char **argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  yakl::init();
  {
    if (argc > 1 && std::string(argv[1]) == "dimension_mismatch") {
      test_dimension_mismatch_failure();
    }
    yakl::timer_start("main");
    // Write so that d1 is always the fastest varying and ordered from there
    int constexpr d1 = 2;
    int constexpr d2 = 3;
    int constexpr d3 = 4;
    int constexpr d4 = 5;
    int constexpr d5 = 6;
    int constexpr d6 = 7;
    int constexpr d7 = 8;
    int constexpr d8 = 9;

    std::string file_name = "testyMcTestFace.nc";

    int         unlim_dim_ind  = 0;
    std::string unlim_dim_name = "snuffaluffagus";
    int         num_entries    = 10;

    // This block is the writing phase
    {
      yakl::SimpleNetCDF nc;

      nc.create( file_name ); // Default mode is overwrite when creating

      // Create the varaibles
      Array  <signed        char *       ,yakl::DeviceSpace> a0   ("a0"   ,d1);
      Array_F<unsigned      char **      ,yakl::DeviceSpace> a1   ("a1"   ,d1,d2);
      Array  <             short ***     ,yakl::DeviceSpace> a2   ("a2"   ,d3,d2,d1);
      Array_F<unsigned     short ****    ,yakl::DeviceSpace> a3   ("a3"   ,d1,d2,d3,d4);
      Array  <               int *****   ,yakl::DeviceSpace> a4   ("a4"   ,d5,d4,d3,d2,d1);
      Array_F<unsigned       int ******  ,Kokkos::HostSpace> a5   ("a5"   ,d1,d2,d3,d4,d5,d6);
      Array  <         long long ******* ,Kokkos::HostSpace> a6   ("a6"   ,d7,d6,d5,d4,d3,d2,d1);
      Array_F<unsigned long long ********,Kokkos::HostSpace> a7   ("a7"   ,d1,d2,d3,d4,d5,d6,d7,d8);
      Array  <             float ***     ,Kokkos::HostSpace> a8   ("a8"   ,d3,d2,d1);
      Array_F<            double ***     ,Kokkos::HostSpace> a9   ("a9"   ,d1,d2,d3);
      Array  <              char **      ,Kokkos::HostSpace> text ("text" ,4,10);
      Array  <               int *       ,Kokkos::HostSpace> bool8("bool8",d8);
      float s0 = 1;
      int   s1 = 2;

      // Assign values
      a0 = 0;
      a1 = 1;
      a2 = 2;
      a3 = 3;
      a4 = 4;
      a5 = 5;
      a6 = 6;
      a7 = 7;
      a8 = 8;
      a9 = 9;
      bool8 = 0;
      bool8(1) = 1;
      bool8(4) = 1;
      text(0,0)='I';
      text(1,0)='n';  text(1,1)='e';  text(1,2)='e';  text(1,3)='d';
      text(2,0)='m';  text(2,1)='o';
      text(3,0)='g';  text(3,1)='a';  text(3,2)='s';  text(3,3)='e';  text(3,4)='s';

      // Write entire arrays to file
      nc.write( a0 , "a0" , {"d1"} );
      nc.write( a1 , "a1" , {"d1","d2"} );
      nc.write( a2 , "a2" , {"d3","d2","d1"} );
      nc.write( a3 , "a3" , {"d1","d2","d3","d4"} );
      nc.write( a4 , "a4" , {"d5","d4","d3","d2","d1"} );
      nc.write( a5 , "a5" , {"d1","d2","d3","d4","d5","d6"} );
      nc.write( a6 , "a6" , {"d7","d6","d5","d4","d3","d2","d1"} );
      nc.write( a7 , "a7" , {"d1","d2","d3","d4","d5","d6","d7","d8"} );
      nc.write( a8 , "a8" , {"d3","d2","d1"} );
      nc.write( a9 , "a9" , {"d1","d2","d3"} );
      nc.write( bool8 , "bool8" , {"d8"} );
      nc.write( s0  , "s0" );
      nc.write( s1  , "s1" );
      nc.write( text , "text" , {"four","ten"} );
    
      // Create a dimension that isn't used
      nc.createDim( "nobody_likes_me" , 100 );

      // Write arrays as entries into an unlimited index
      nc.write1( a0 , "a0_unlim" , {"d1"}                                    , unlim_dim_ind , unlim_dim_name );
      nc.write1( a1 , "a1_unlim" , {"d1","d2"}                               , unlim_dim_ind , unlim_dim_name );
      nc.write1( a2 , "a2_unlim" , {"d3","d2","d1"}                          , unlim_dim_ind , unlim_dim_name );
      nc.write1( a3 , "a3_unlim" , {"d1","d2","d3","d4"}                     , unlim_dim_ind , unlim_dim_name );
      nc.write1( a4 , "a4_unlim" , {"d5","d4","d3","d2","d1"}                , unlim_dim_ind , unlim_dim_name );
      nc.write1( a5 , "a5_unlim" , {"d1","d2","d3","d4","d5","d6"}           , unlim_dim_ind , unlim_dim_name );
      nc.write1( a6 , "a6_unlim" , {"d7","d6","d5","d4","d3","d2","d1"}      , unlim_dim_ind , unlim_dim_name );
      // nc.write1( a7 , "a7_unlim" , {"d1","d2","d3","d4","d5","d6","d7","d8"} , unlim_dim_ind , unlim_dim_name );
      nc.write1( a8 , "a8_unlim" , {"d3","d2","d1"}                          , unlim_dim_ind , unlim_dim_name );
      nc.write1( a9 , "a9_unlim" , {"d1","d2","d3"}                          , unlim_dim_ind , unlim_dim_name );

      // Write scalars as entries into an unlimited index
      nc.write1( s0 , "s0_unlim"                                             , unlim_dim_ind , unlim_dim_name );
      nc.write1( s1 , "s1_unlim"                                             , unlim_dim_ind , unlim_dim_name );

      nc.close();

      // Write the rest of the entries in the unlimited index
      for (int i=1; i < num_entries; i++) {
        nc.open( file_name , yakl::NETCDF_MODE_WRITE );
        unlim_dim_ind = nc.getDimSize( unlim_dim_name );

        // Write arrays as entries into an unlimited index
        nc.write1( a0 , "a0_unlim" , {"d1"}                                    , unlim_dim_ind , unlim_dim_name );
        nc.write1( a1 , "a1_unlim" , {"d1","d2"}                               , unlim_dim_ind , unlim_dim_name );
        nc.write1( a2 , "a2_unlim" , {"d3","d2","d1"}                          , unlim_dim_ind , unlim_dim_name );
        nc.write1( a3 , "a3_unlim" , {"d1","d2","d3","d4"}                     , unlim_dim_ind , unlim_dim_name );
        nc.write1( a4 , "a4_unlim" , {"d5","d4","d3","d2","d1"}                , unlim_dim_ind , unlim_dim_name );
        nc.write1( a5 , "a5_unlim" , {"d1","d2","d3","d4","d5","d6"}           , unlim_dim_ind , unlim_dim_name );
        nc.write1( a6 , "a6_unlim" , {"d7","d6","d5","d4","d3","d2","d1"}      , unlim_dim_ind , unlim_dim_name );
        // nc.write1( a7 , "a7_unlim" , {"d1","d2","d3","d4","d5","d6","d7","d8"} , unlim_dim_ind , unlim_dim_name );
        nc.write1( a8 , "a8_unlim" , {"d3","d2","d1"}                          , unlim_dim_ind , unlim_dim_name );
        nc.write1( a9 , "a9_unlim" , {"d1","d2","d3"}                          , unlim_dim_ind , unlim_dim_name );

        // Write scalars as entries into an unlimited index
        nc.write1( s0 , "s0_unlim"                                             , unlim_dim_ind , unlim_dim_name );
        nc.write1( s1 , "s1_unlim"                                             , unlim_dim_ind , unlim_dim_name );

        nc.close();
      }
    }

    // This block is the reading phase
    {
      yakl::SimpleNetCDF nc;
      nc.open( file_name ); // Default mode is read when opening

      if ( nc.dimExists("chicken_liver")) die("ERROR: chicken_liver is not a dimension");
      if (!nc.dimExists("d1"))            die("ERROR: d1 is a dimension");
      if ( nc.varExists("small_colonel")) die("ERROR: small_colonel is not a variable");
      if (!nc.varExists("s1_unlim"))      die("ERROR: s1_unlim is a variable");
      if ( nc.getDimSize(unlim_dim_name) != num_entries) die("ERROR: unlim dim size should be 10");
      if ( nc.getDimSize("nobody_likes_me") != 100) die("ERROR: nobody_likes_me size should be 100");

      // We're going to permute the memory space and Array style to ensure it's written and read correctly
      // If the dimensions are off, there will be an error thrown from YAKL_netcdf
      Array  <signed        char *       ,yakl::DeviceSpace> a0("a0",d1);
      Array  <unsigned      char **      ,Kokkos::HostSpace> a1("a1",d2,d1);
      Array  <             short ***     ,yakl::DeviceSpace> a2("a2",d3,d2,d1);
      Array  <unsigned     short ****    ,Kokkos::HostSpace> a3("a3",d4,d3,d2,d1);
      Array  <               int *****   ,yakl::DeviceSpace> a4("a4",d5,d4,d3,d2,d1);
      Array_F<unsigned       int ******  ,Kokkos::HostSpace> a5("a5",d1,d2,d3,d4,d5,d6);
      Array_F<         long long ******* ,yakl::DeviceSpace> a6("a6",d1,d2,d3,d4,d5,d6,d7);
      Array_F<unsigned long long ********,Kokkos::HostSpace> a7("a7",d1,d2,d3,d4,d5,d6,d7,d8);
      Array_F<             float ***     ,yakl::DeviceSpace> a8("a8",d1,d2,d3);
      Array_F<            double ***     ,Kokkos::HostSpace> a9("a9",d1,d2,d3);
      Array  <signed        char **      ,yakl::DeviceSpace> a0_unlim("a0_unlim",num_entries,d1);
      Array  <unsigned      char ***     ,Kokkos::HostSpace> a1_unlim("a1_unlim",num_entries,d2,d1);
      Array  <             short ****    ,yakl::DeviceSpace> a2_unlim("a2_unlim",num_entries,d3,d2,d1);
      Array  <unsigned     short *****   ,Kokkos::HostSpace> a3_unlim("a3_unlim",num_entries,d4,d3,d2,d1);
      Array  <               int ******  ,yakl::DeviceSpace> a4_unlim("a4_unlim",num_entries,d5,d4,d3,d2,d1);
      Array_F<unsigned       int ******* ,Kokkos::HostSpace> a5_unlim("a5_unlim",d1,d2,d3,d4,d5,d6,num_entries);
      Array_F<         long long ********,yakl::DeviceSpace> a6_unlim("a6_unlim",d1,d2,d3,d4,d5,d6,d7,num_entries);
      Array_F<             float ****    ,yakl::DeviceSpace> a8_unlim("a8_unlim",d1,d2,d3,num_entries);
      Array_F<            double ****    ,Kokkos::HostSpace> a9_unlim("a9_unlim",d1,d2,d3,num_entries);
      Array  <              char **      ,Kokkos::HostSpace> text("text",4,10);
      Array  <              bool *       ,Kokkos::HostSpace> bool8("bool8",d8);
      Array  <             float *       ,Kokkos::HostSpace> s0_unlim("s0_unlim",num_entries);
      Array  <             int   *       ,yakl::DeviceSpace> s1_unlim("s1_unlim",num_entries);
      float s0;
      int   s1;

      nc.read( a0 , "a0" );
      nc.read( a1 , "a1" );
      nc.read( a2 , "a2" );
      nc.read( a3 , "a3" );
      nc.read( a4 , "a4" );
      nc.read( a5 , "a5" );
      nc.read( a6 , "a6" );
      nc.read( a7 , "a7" );
      nc.read( a8 , "a8" );
      nc.read( a9 , "a9" );
      nc.read( s0 , "s0" );
      nc.read( s1 , "s1" );
      nc.read( bool8 , "bool8");
      nc.read( text , "text" );
      nc.read( a0_unlim , "a0_unlim" );
      nc.read( a1_unlim , "a1_unlim" );
      nc.read( a2_unlim , "a2_unlim" );
      nc.read( a3_unlim , "a3_unlim" );
      nc.read( a4_unlim , "a4_unlim" );
      nc.read( a5_unlim , "a5_unlim" );
      nc.read( a6_unlim , "a6_unlim" );
      nc.read( a8_unlim , "a8_unlim" );
      nc.read( a9_unlim , "a9_unlim" );
      nc.read( s0_unlim , "s0_unlim" );
      nc.read( s1_unlim , "s1_unlim" );

      nc.close();

      using yakl::intrinsics::sum;
      using yakl::intrinsics::size;

      if ( sum(a0) / size(a0) != 0 ) die("ERROR: avg of a0 should be 0");
      if ( sum(a1) / size(a1) != 1 ) die("ERROR: avg of a1 should be 1");
      if ( sum(a2) / size(a2) != 2 ) die("ERROR: avg of a2 should be 2");
      if ( sum(a3) / size(a3) != 3 ) die("ERROR: avg of a3 should be 3");
      if ( sum(a4) / size(a4) != 4 ) die("ERROR: avg of a4 should be 4");
      if ( sum(a5) / size(a5) != 5 ) die("ERROR: avg of a5 should be 5");
      if ( sum(a6) / size(a6) != 6 ) die("ERROR: avg of a6 should be 6");
      if ( sum(a7) / size(a7) != 7 ) die("ERROR: avg of a7 should be 7");
      if ( sum(a8) / size(a8) != 8 ) die("ERROR: avg of a8 should be 8");
      if ( sum(a9) / size(a9) != 9 ) die("ERROR: avg of a9 should be 9");

      if ( sum(a0_unlim) / size(a0_unlim) != 0 ) die("ERROR: avg of a0_unlim should be 0");
      if ( sum(a1_unlim) / size(a1_unlim) != 1 ) die("ERROR: avg of a1_unlim should be 1");
      if ( sum(a2_unlim) / size(a2_unlim) != 2 ) die("ERROR: avg of a2_unlim should be 2");
      if ( sum(a3_unlim) / size(a3_unlim) != 3 ) die("ERROR: avg of a3_unlim should be 3");
      if ( sum(a4_unlim) / size(a4_unlim) != 4 ) die("ERROR: avg of a4_unlim should be 4");
      if ( sum(a5_unlim) / size(a5_unlim) != 5 ) die("ERROR: avg of a5_unlim should be 5");
      if ( sum(a6_unlim) / size(a6_unlim) != 6 ) die("ERROR: avg of a6_unlim should be 6");
      if ( sum(a8_unlim) / size(a8_unlim) != 8 ) die("ERROR: avg of a8_unlim should be 8");
      if ( sum(a9_unlim) / size(a9_unlim) != 9 ) die("ERROR: avg of a9_unlim should be 9");
      if ( sum(s0_unlim) / size(s0_unlim) != 1 ) die("ERROR: avg of s0_unlim should be 1");
      if ( sum(s1_unlim) / size(s1_unlim) != 2 ) die("ERROR: avg of s1_unlim should be 2");

      if ( text(2,0) != 'm' || text(2,1) != 'o' ) die("ERROR: text is incorrect");
    }

    test_unsigned_long_and_move_ownership();
    test_dimension_order();
    test_attributes();
    test_second_unlimited_dimension_failure();

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif

  return 0;
}
