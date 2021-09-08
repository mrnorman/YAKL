
#include <iostream>
#include "YAKL.h"
#include "YAKL_netcdf.h"

using yakl::Array;
using yakl::styleC;
using yakl::styleFortran;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
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
      Array<signed        char,1,memDevice,styleC      > a0("a0",d1);
      Array<unsigned      char,2,memDevice,styleFortran> a1("a1",d1,d2);
      Array<             short,3,memDevice,styleC      > a2("a2",d3,d2,d1);
      Array<unsigned     short,4,memDevice,styleFortran> a3("a3",d1,d2,d3,d4);
      Array<               int,5,memDevice,styleC      > a4("a4",d5,d4,d3,d2,d1);
      Array<unsigned       int,6,memHost  ,styleFortran> a5("a5",d1,d2,d3,d4,d5,d6);
      Array<         long long,7,memHost  ,styleC      > a6("a6",d7,d6,d5,d4,d3,d2,d1);
      Array<unsigned long long,8,memHost  ,styleFortran> a7("a7",d1,d2,d3,d4,d5,d6,d7,d8);
      Array<             float,3,memHost  ,styleC      > a8("a8",d3,d2,d1);
      Array<            double,3,memHost  ,styleFortran> a9("a9",d1,d2,d3);
      Array<              char,2,memHost  ,styleC      > text("text",4,10);
      Array<               int,1,memHost  ,styleC      > bool8("bool8",d8);
      float s0 = 1;
      int   s1 = 2;

      // Assign values
      yakl::memset(a0,0);
      yakl::memset(a1,1);
      yakl::memset(a2,2);
      yakl::memset(a3,3);
      yakl::memset(a4,4);
      yakl::memset(a5,5);
      yakl::memset(a6,6);
      yakl::memset(a7,7);
      yakl::memset(a8,8);
      yakl::memset(a9,9);
      yakl::memset(bool8,0);
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
      Array<signed        char,1,memDevice,styleC      > a0("a0",d1);
      Array<unsigned      char,2,memHost  ,styleC      > a1("a1",d2,d1);
      Array<             short,3,memDevice,styleC      > a2("a2",d3,d2,d1);
      Array<unsigned     short,4,memHost  ,styleC      > a3("a3",d4,d3,d2,d1);
      Array<               int,5,memDevice,styleC      > a4("a4",d5,d4,d3,d2,d1);
      Array<unsigned       int,6,memHost  ,styleFortran> a5("a5",d1,d2,d3,d4,d5,d6);
      Array<         long long,7,memDevice,styleFortran> a6("a6",d1,d2,d3,d4,d5,d6,d7);
      Array<unsigned long long,8,memHost  ,styleFortran> a7("a7",d1,d2,d3,d4,d5,d6,d7,d8);
      Array<             float,3,memDevice,styleFortran> a8("a8",d1,d2,d3);
      Array<            double,3,memHost  ,styleFortran> a9("a9",d1,d2,d3);
      Array<signed        char,2,memDevice,styleC      > a0_unlim("a0_unlim",num_entries,d1);
      Array<unsigned      char,3,memHost  ,styleC      > a1_unlim("a1_unlim",num_entries,d2,d1);
      Array<             short,4,memDevice,styleC      > a2_unlim("a2_unlim",num_entries,d3,d2,d1);
      Array<unsigned     short,5,memHost  ,styleC      > a3_unlim("a3_unlim",num_entries,d4,d3,d2,d1);
      Array<               int,6,memDevice,styleC      > a4_unlim("a4_unlim",num_entries,d5,d4,d3,d2,d1);
      Array<unsigned       int,7,memHost  ,styleFortran> a5_unlim("a5_unlim",d1,d2,d3,d4,d5,d6,num_entries);
      Array<         long long,8,memDevice,styleFortran> a6_unlim("a6_unlim",d1,d2,d3,d4,d5,d6,d7,num_entries);
      Array<             float,4,memDevice,styleFortran> a8_unlim("a8_unlim",d1,d2,d3,num_entries);
      Array<            double,4,memHost  ,styleFortran> a9_unlim("a9_unlim",d1,d2,d3,num_entries);
      Array<              char,2,memHost  ,styleC      > text("text",4,10);
      Array<              bool,1,memHost  ,styleC      > bool8("bool8",d8);
      Array<float,1,memHost  ,styleC> s0_unlim("s0_unlim",num_entries);
      Array<int  ,1,memDevice,styleC> s1_unlim("s1_unlim",num_entries);
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

  }
  yakl::finalize();
  
  return 0;
}

