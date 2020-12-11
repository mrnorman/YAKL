
#pragma once

#include <vector>
#include "YAKL.h"
#include "mpi.h"
#include <pnetcdf.h>

namespace yakl {

  //Error reporting routine for the PNetCDF I/O
  inline void ncmpiwrap( int ierr , int line ) {
    if (ierr != NC_NOERR) {
      printf("NetCDF Error at line: %d\n", line);
      printf("%s\n",ncmpi_strerror(ierr));
      throw "";
    }
  }


  //////////////////////////////////////////////
  // ncmpi_put_var
  //////////////////////////////////////////////
  void pnetcdf_put_var(int ncid , int varid , signed char const *data) {
    ncmpiwrap( ncmpi_put_var_schar( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_var_uchar( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , short const *data) {
    ncmpiwrap( ncmpi_put_var_short( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_var_ushort( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , int const *data) {
    ncmpiwrap( ncmpi_put_var_int( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_var_uint( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , long long const *data) {
    ncmpiwrap( ncmpi_put_var_longlong( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_var_ulonglong( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , float const *data) {
    ncmpiwrap( ncmpi_put_var_float( ncid , varid , data ) , __LINE__ );
  }
  void pnetcdf_put_var(int ncid , int varid , double const *data) {
    ncmpiwrap( ncmpi_put_var_double( ncid , varid , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_var1
  //////////////////////////////////////////////
  void pnetcdf_put_var1(int ncid , int varid , signed char const *data) {
    ncmpiwrap( ncmpi_put_var1_schar( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_var1_uchar( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , short const *data) {
    ncmpiwrap( ncmpi_put_var1_short( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_var1_ushort( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , int const *data) {
    ncmpiwrap( ncmpi_put_var1_int( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_var1_uint( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , long long const *data) {
    ncmpiwrap( ncmpi_put_var1_longlong( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_var1_ulonglong( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , float const *data) {
    ncmpiwrap( ncmpi_put_var1_float( ncid , varid , 0 , data ) , __LINE__ );
  }
  void pnetcdf_put_var1(int ncid , int varid , double const *data) {
    ncmpiwrap( ncmpi_put_var1_double( ncid , varid , 0 , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_vara
  //////////////////////////////////////////////
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char const *data) {
    ncmpiwrap( ncmpi_put_vara_schar( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_vara_uchar( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short const *data) {
    ncmpiwrap( ncmpi_put_vara_short( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_vara_ushort( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int const *data) {
    ncmpiwrap( ncmpi_put_vara_int( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_vara_uint( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long const *data) {
    ncmpiwrap( ncmpi_put_vara_longlong( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_vara_ulonglong( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float const *data) {
    ncmpiwrap( ncmpi_put_vara_float( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double const *data) {
    ncmpiwrap( ncmpi_put_vara_double( ncid , varid , start , count , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_vara_all
  //////////////////////////////////////////////
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char const *data) {
    ncmpiwrap( ncmpi_put_vara_schar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_vara_uchar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short const *data) {
    ncmpiwrap( ncmpi_put_vara_short_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_vara_ushort_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int const *data) {
    ncmpiwrap( ncmpi_put_vara_int_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_vara_uint_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long const *data) {
    ncmpiwrap( ncmpi_put_vara_longlong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_vara_ulonglong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float const *data) {
    ncmpiwrap( ncmpi_put_vara_float_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double const *data) {
    ncmpiwrap( ncmpi_put_vara_double_all( ncid , varid , start , count , data ) , __LINE__ );
  }


  class SimplePNetCDF {
  protected:

    int ncid;

  public:

    SimplePNetCDF() {
      ncid = -1;
    }


    ~SimplePNetCDF() {
      close();
    }


    SimplePNetCDF(SimplePNetCDF &&in) = delete;
    SimplePNetCDF(SimplePNetCDF const &in) = delete;
    SimplePNetCDF &operator=(SimplePNetCDF &&in) = delete;
    SimplePNetCDF &operator=(SimplePNetCDF const &in) = delete;


    void open(std::string fname) {
      close();
      ncmpiwrap( ncmpi_open( MPI_COMM_WORLD , fname.c_str() , NC_WRITE , MPI_INFO_NULL , &ncid ) , __LINE__ );
    }


    void create(std::string fname) {
      close();
      ncmpiwrap( ncmpi_create( MPI_COMM_WORLD , fname.c_str() , NC_CLOBBER , MPI_INFO_NULL , &ncid ) , __LINE__ );
    }


    void close() {
      if (ncid != -1) {
        ncmpiwrap( ncmpi_close(ncid) , __LINE__ );
      }
      ncid = -1;
    }


    int get_dim_id( std::string dimName ) const {
      int dimid;
      ncmpiwrap( ncmpi_inq_dimid( ncid , dimName.c_str() , &dimid) , __LINE__ );
      return dimid;
    }


    int get_var_id( std::string varName ) const {
      int varid;
      ncmpiwrap( ncmpi_inq_varid( ncid , varName.c_str() , &varid) , __LINE__ );
      return varid;
    }


    bool var_exists( std::string varName ) const {
      int varid;
      int ierr = ncmpi_inq_varid( ncid , varName.c_str() , &varid);
      if (ierr == NC_NOERR) {
        return true;
      } else {
        return false;
      }
    }


    bool dim_exists( std::string dimName ) const {
      int dimid;
      int ierr = ncmpi_inq_dimid( ncid , dimName.c_str() , &dimid);
      if (ierr == NC_NOERR) {
        return true;
      } else {
        return false;
      }
    }


    MPI_Offset get_dim_size( std::string dimName ) const {
      int dimid;
      MPI_Offset dimlen;
      ncmpiwrap( ncmpi_inq_dimid ( ncid , dimName.c_str() , &dimid) , __LINE__ );
      ncmpiwrap( ncmpi_inq_dimlen( ncid , dimid , &dimlen ) , __LINE__ );
      return dimlen;
    }


    template <class T>
    void create_var( std::string varName , std::vector<std::string> dnames ) {
      int ndims = dnames.size();
      std::vector<int> dimids(ndims);
      for (int i=0; i < ndims; i++) {
        dimids[i] = get_dim_id( dnames[i] );
      }
      nc_type xtype = getType<T>();
      int varid;
      ncmpiwrap( ncmpi_def_var( ncid , varName.c_str() , xtype , ndims , dimids.data() , &varid ) , __LINE__ );
    }


    void create_dim( std::string dimName , MPI_Offset len ) {
      int dimid;
      ncmpiwrap( ncmpi_def_dim( ncid , dimName.c_str() , len , &dimid ) , __LINE__ );
    }


    void create_unlim_dim( std::string dimName ) {
      int dimid;
      ncmpiwrap( ncmpi_def_dim( ncid , dimName.c_str() , NC_UNLIMITED , &dimid ) , __LINE__ );
    }


    void enddef() {
      ncmpiwrap( ncmpi_enddef(ncid) , __LINE__ );
    }


    void begin_indep_data() {
      ncmpiwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
    }


    void end_indep_data() {
      ncmpiwrap( ncmpi_end_indep_data(ncid) , __LINE__ );
    }


    /***************************************************************************************************
    Serially write an entire Array at once
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void write(Array<T,rank,myMem,myStyle> const &arr , std::string varName) {
      int varid = get_var_id( varName );
      pnetcdf_put_var( ncid , varid , arr.data() );
    }


    /***************************************************************************************************
    Collectively write an entire Array at once
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void write_all(Array<T,rank,myMem,myStyle> const &arr , std::string varName , std::vector<MPI_Offset> start ) {
      if (rank != start   .size()) { yakl_throw("start.size() != Array's rank"); }
      std::vector<MPI_Offset> count(rank);
      for (int i=0; i < rank; i++) {
        count[i] = arr.dimension[i];
      }

      int varid = get_var_id(varName);

      if (myMem == memDevice) {
        pnetcdf_put_vara_all( ncid , varid , start.data() , count.data() , arr.createHostCopy().data() );
      } else {
        pnetcdf_put_vara_all( ncid , varid , start.data() , count.data() , arr.data() );
      }
    }


    /***************************************************************************************************
    Serially write one entry of a scalar into the unlimited index
    ***************************************************************************************************/
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,int>::type = 0 >
    void write1(T val , std::string varName , int ind , std::string ulDimName="unlim" ) {
      int varid = get_var_id(varName);

      std::vector<MPI_Offset> start(1);
      std::vector<MPI_Offset> count(1);
      start[0] = ind;
      count[0] = 1;

      pnetcdf_put_vara( ncid , varid , start.data() , count.data() , &val );
    }


    /***************************************************************************************************
    Serially write one entry of an Array into the unlimited index
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void write1(Array<T,rank,myMem,myStyle> const &arr , std::string varName ,
                int ind , std::string ulDimName="unlim" ) {
      std::vector<MPI_Offset> start(rank+1);
      std::vector<MPI_Offset> count(rank+1);
      start[0] = ind;
      count[0] = 1;
      for (int i=1; i < rank+1; i++) {
        start[i] = 0;
        count[i] = arr.dimension[i-1];
      }

      int varid = get_var_id(varName);

      if (myMem == memDevice) {
        pnetcdf_put_vara( ncid , varid , start.data() , count.data() , arr.createHostCopy().data() );
      } else {
        pnetcdf_put_vara( ncid , varid , start.data() , count.data() , arr.data() );
      }
    }


    /***************************************************************************************************
    Collectively write one entry of an Array into the unlimited index
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void write1_all(Array<T,rank,myMem,myStyle> const &arr , std::string varName ,
                    int ind , std::vector<MPI_Offset> start_in , std::string ulDimName="unlim" ) {
      if (rank != start_in.size()) { yakl_throw("start_in.size() != Array's rank"); }
      std::vector<MPI_Offset> start(rank+1);
      std::vector<MPI_Offset> count(rank+1);
      start[0] = ind;
      count[0] = 1;
      for (int i=1; i < rank+1; i++) {
        start[i] = start_in[i-1];
        count[i] = arr.dimension[i-1];
      }

      int varid = get_var_id(varName);

      if (myMem == memDevice) {
        pnetcdf_put_vara_all( ncid , varid , start.data() , count.data() , arr.createHostCopy().data() );
      } else {
        pnetcdf_put_vara_all( ncid , varid , start.data() , count.data() , arr.data() );
      }
    }


    // /***************************************************************************************************
    // Serially read an entire Array
    // ***************************************************************************************************/
    // template <class T, int rank, int myMem, int myStyle>
    // void read(Array<T,rank,myMem,myStyle> &arr , std::string varName) {

    //   auto var = file->getVar(varName);

    //   if (myMem == memDevice) {
    //     auto arrHost = arr.createHostCopy();
    //     var.getVar(arrHost.data());
    //     arrHost.deep_copy_to(arr);
    //   } else {
    //     var.getVar(arr.data());
    //   }
    // }


    // /***************************************************************************************************
    // Collectively read an entire Array
    // ***************************************************************************************************/
    // template <class T, int rank, int myMem, int myStyle>
    // void read_all(Array<T,rank,myMem,myStyle> &arr , std::string varName , std::vector<MPI_Offset> start ) {
    //   if (start.size() != rank) { yakl_throw("ERROR: start.size() != arr's rank"); }

    //   auto var = file->getVar(varName);

    //   std::vector<MPI_Offset> count(rank);
    //   for (int i=0; i < rank; i++) {
    //     count[i] = arr.dimension[i];
    //   }

    //   if (myMem == memDevice) {
    //     auto arrHost = arr.createHostCopy();
    //     var.getVar_all( start , count , arrHost.data() );
    //     arrHost.deep_copy_to(arr);
    //   } else {
    //     var.getVar_all( start , count , arr.data() );
    //   }
    // }


    // /***************************************************************************************************
    // Read a single scalar value
    // ***************************************************************************************************/
    // template <class T>
    // void read(T &arr , std::string varName) {
    //   auto var = file->getVar(varName);
    //   var.getVar(&arr);
    // }


    // /***************************************************************************************************
    // Write a single scalar value
    // ***************************************************************************************************/
    // template <class T>
    // void write(T arr , std::string varName) {
    //   auto var = file->getVar(varName);
    //   var.putVar(&arr);
    // }


    /***************************************************************************************************
    Determine the type of a template T
    ***************************************************************************************************/
    template <class T> nc_type getType() const {
           if ( std::is_same<T,          char>::value ) { return NC_CHAR;   }
      else if ( std::is_same<T,unsigned  char>::value ) { return NC_UBYTE;  }
      else if ( std::is_same<T,         short>::value ) { return NC_SHORT;  }
      else if ( std::is_same<T,unsigned short>::value ) { return NC_USHORT; }
      else if ( std::is_same<T,           int>::value ) { return NC_INT;    }
      else if ( std::is_same<T,unsigned   int>::value ) { return NC_UINT;   }
      else if ( std::is_same<T,          long>::value ) { return NC_INT64;  }
      else if ( std::is_same<T,unsigned  long>::value ) { return NC_UINT64; }
      else if ( std::is_same<T,         float>::value ) { return NC_FLOAT;  }
      else if ( std::is_same<T,        double>::value ) { return NC_DOUBLE; }
      else { yakl_throw("Invalid type"); }
      return -1;
    }

  };

}


