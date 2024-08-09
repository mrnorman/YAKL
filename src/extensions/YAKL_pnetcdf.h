
#pragma once

#include <vector>
#include "YAKL.h"
#include "mpi.h"
#include <pnetcdf.h>
#include <stdexcept>

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  //Error reporting routine for the PNetCDF I/O
  /** @private */
  inline void ncmpiwrap( int ierr , int line ) {
    if (ierr != NC_NOERR) {
      std::cerr << "PNetCDF ERROR at line: " << line << std::endl;
      throw std::runtime_error(ncmpi_strerror(ierr));
    }
  }


  //////////////////////////////////////////////
  // ncmpi_put_var
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , signed char const *data) {
    ncmpiwrap( ncmpi_put_var_schar( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_var_uchar( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , short const *data) {
    ncmpiwrap( ncmpi_put_var_short( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_var_ushort( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , int const *data) {
    ncmpiwrap( ncmpi_put_var_int( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_var_uint( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , long long const *data) {
    ncmpiwrap( ncmpi_put_var_longlong( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_var_ulonglong( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , float const *data) {
    ncmpiwrap( ncmpi_put_var_float( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var(int ncid , int varid , double const *data) {
    ncmpiwrap( ncmpi_put_var_double( ncid , varid , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_var1
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , signed char const *data) {
    ncmpiwrap( ncmpi_put_var1_schar( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_var1_uchar( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , short const *data) {
    ncmpiwrap( ncmpi_put_var1_short( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_var1_ushort( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , int const *data) {
    ncmpiwrap( ncmpi_put_var1_int( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_var1_uint( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , long long const *data) {
    ncmpiwrap( ncmpi_put_var1_longlong( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_var1_ulonglong( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , float const *data) {
    ncmpiwrap( ncmpi_put_var1_float( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_var1(int ncid , int varid , double const *data) {
    ncmpiwrap( ncmpi_put_var1_double( ncid , varid , 0 , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_vara
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char const *data) {
    ncmpiwrap( ncmpi_put_vara_schar( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_vara_uchar( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short const *data) {
    ncmpiwrap( ncmpi_put_vara_short( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_vara_ushort( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int const *data) {
    ncmpiwrap( ncmpi_put_vara_int( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_vara_uint( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long const *data) {
    ncmpiwrap( ncmpi_put_vara_longlong( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_vara_ulonglong( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float const *data) {
    ncmpiwrap( ncmpi_put_vara_float( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double const *data) {
    ncmpiwrap( ncmpi_put_vara_double( ncid , varid , start , count , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_vara_all
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char const *data) {
    ncmpiwrap( ncmpi_put_vara_schar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_vara_uchar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short const *data) {
    ncmpiwrap( ncmpi_put_vara_short_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_vara_ushort_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int const *data) {
    ncmpiwrap( ncmpi_put_vara_int_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_vara_uint_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long const *data) {
    ncmpiwrap( ncmpi_put_vara_longlong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_vara_ulonglong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float const *data) {
    ncmpiwrap( ncmpi_put_vara_float_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double const *data) {
    ncmpiwrap( ncmpi_put_vara_double_all( ncid , varid , start , count , data ) , __LINE__ );
  }



  //////////////////////////////////////////////
  // ncmpi_get_var
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , signed char *data) {
    ncmpiwrap( ncmpi_get_var_schar( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , unsigned char *data) {
    ncmpiwrap( ncmpi_get_var_uchar( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , short *data) {
    ncmpiwrap( ncmpi_get_var_short( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , unsigned short *data) {
    ncmpiwrap( ncmpi_get_var_ushort( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , int *data) {
    ncmpiwrap( ncmpi_get_var_int( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , unsigned int *data) {
    ncmpiwrap( ncmpi_get_var_uint( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , long long *data) {
    ncmpiwrap( ncmpi_get_var_longlong( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_var_ulonglong( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , float *data) {
    ncmpiwrap( ncmpi_get_var_float( ncid , varid , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var(int ncid , int varid , double *data) {
    ncmpiwrap( ncmpi_get_var_double( ncid , varid , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_get_var1
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , signed char *data) {
    ncmpiwrap( ncmpi_get_var1_schar( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned char *data) {
    ncmpiwrap( ncmpi_get_var1_uchar( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , short *data) {
    ncmpiwrap( ncmpi_get_var1_short( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned short *data) {
    ncmpiwrap( ncmpi_get_var1_ushort( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , int *data) {
    ncmpiwrap( ncmpi_get_var1_int( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned int *data) {
    ncmpiwrap( ncmpi_get_var1_uint( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , long long *data) {
    ncmpiwrap( ncmpi_get_var1_longlong( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_var1_ulonglong( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , float *data) {
    ncmpiwrap( ncmpi_get_var1_float( ncid , varid , 0 , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_var1(int ncid , int varid , double *data) {
    ncmpiwrap( ncmpi_get_var1_double( ncid , varid , 0 , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_get_vara
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char *data) {
    ncmpiwrap( ncmpi_get_vara_schar( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char *data) {
    ncmpiwrap( ncmpi_get_vara_uchar( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short *data) {
    ncmpiwrap( ncmpi_get_vara_short( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short *data) {
    ncmpiwrap( ncmpi_get_vara_ushort( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int *data) {
    ncmpiwrap( ncmpi_get_vara_int( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int *data) {
    ncmpiwrap( ncmpi_get_vara_uint( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long *data) {
    ncmpiwrap( ncmpi_get_vara_longlong( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_vara_ulonglong( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float *data) {
    ncmpiwrap( ncmpi_get_vara_float( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double *data) {
    ncmpiwrap( ncmpi_get_vara_double( ncid , varid , start , count , data ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_get_vara_all
  //////////////////////////////////////////////
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char *data) {
    ncmpiwrap( ncmpi_get_vara_schar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char *data) {
    ncmpiwrap( ncmpi_get_vara_uchar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short *data) {
    ncmpiwrap( ncmpi_get_vara_short_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short *data) {
    ncmpiwrap( ncmpi_get_vara_ushort_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int *data) {
    ncmpiwrap( ncmpi_get_vara_int_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int *data) {
    ncmpiwrap( ncmpi_get_vara_uint_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long *data) {
    ncmpiwrap( ncmpi_get_vara_longlong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_vara_ulonglong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float *data) {
    ncmpiwrap( ncmpi_get_vara_float_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  /** @private */
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double *data) {
    ncmpiwrap( ncmpi_get_vara_double_all( ncid , varid , start , count , data ) , __LINE__ );
  }




  /** @brief Simple way to write yakl::Array objects to NetCDF files in parallel */
  class SimplePNetCDF {
  protected:

    /** @private */
    int       ncid;
    MPI_Comm  comm;

  public:

    SimplePNetCDF(MPI_Comm comm = MPI_COMM_WORLD) { this->comm = comm;  this->ncid = -1; }
    ~SimplePNetCDF() { close(); }


    /** @brief Open a file */
    void open(std::string fname , int omode = NC_WRITE , MPI_Info info = MPI_INFO_NULL ) {
      close();
      ncmpiwrap( ncmpi_open( comm , fname.c_str() , omode , info , &ncid ) , __LINE__ );
    }


    /** @brief Create a file with an optional flag parameter */
    void create(std::string fname , int flag = NC_CLOBBER , MPI_Info info = MPI_INFO_NULL ) {
      close();
      ncmpiwrap( ncmpi_create( comm , fname.c_str() , flag , info , &ncid ) , __LINE__ );
    }


    /** @brief Close a file */
    void close() {
      if (ncid != -1) ncmpiwrap( ncmpi_close(ncid) , __LINE__ );
      ncid = -1;
    }


    /** @brief Get dimension ID of a dimension name */
    int get_dim_id( std::string dimName ) const {
      int dimid;
      ncmpiwrap( ncmpi_inq_dimid( ncid , dimName.c_str() , &dimid) , __LINE__ );
      return dimid;
    }


    /** @brief Get variable ID of a variable name */
    int get_var_id( std::string varName ) const {
      int varid;
      ncmpiwrap( ncmpi_inq_varid( ncid , varName.c_str() , &varid) , __LINE__ );
      return varid;
    }


    /** @brief Determine if a variable exists */
    bool var_exists( std::string varName ) const {
      int varid;
      int ierr = ncmpi_inq_varid( ncid , varName.c_str() , &varid);
      if (ierr == NC_NOERR) { return true;  } 
      else                  { return false; }
    }


    /** @brief Determine if a dimension exists */
    bool dim_exists( std::string dimName ) const {
      int dimid;
      int ierr = ncmpi_inq_dimid( ncid , dimName.c_str() , &dimid);
      if (ierr == NC_NOERR) { return true;  }
      else                  { return false; }
    }


    /** @brief Get the size of a dimension name */
    MPI_Offset get_dim_size( std::string dimName ) const {
      int dimid;
      MPI_Offset dimlen;
      ncmpiwrap( ncmpi_inq_dimid ( ncid , dimName.c_str() , &dimid) , __LINE__ );
      ncmpiwrap( ncmpi_inq_dimlen( ncid , dimid , &dimlen ) , __LINE__ );
      return dimlen;
    }


    /** @brief Create a variable with the given dimension names */
    template <class T>
    void create_var( std::string varName , std::vector<std::string> dnames ) {
      int ndims = dnames.size();
      std::vector<int> dimids(ndims);
      for (int i=0; i < ndims; i++) { dimids[i] = get_dim_id( dnames[i] ); }
      nc_type xtype = getType<T>();
      int varid;
      ncmpiwrap( ncmpi_def_var( ncid , varName.c_str() , xtype , ndims , dimids.data() , &varid ) , __LINE__ );
    }


    /** @brief Create a dimension with the given size */
    void create_dim( std::string dimName , MPI_Offset len ) {
      int dimid;
      ncmpiwrap( ncmpi_def_dim( ncid , dimName.c_str() , len , &dimid ) , __LINE__ );
    }


    /** @brief Create an unlimited dimension */
    void create_unlim_dim( std::string dimName ) {
      int dimid;
      ncmpiwrap( ncmpi_def_dim( ncid , dimName.c_str() , NC_UNLIMITED , &dimid ) , __LINE__ );
    }


    /** @brief End "define mode" */
    void redef() {
      ncmpiwrap( ncmpi_redef(ncid) , __LINE__ );
    }


    /** @brief End "define mode" */
    void enddef() {
      ncmpiwrap( ncmpi_enddef(ncid) , __LINE__ );
    }


    /** @brief Begin independent data writing mode (non-collective data writing) */
    void begin_indep_data() {
      ncmpiwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
    }


    /** @brief End independent data writing mode (non-collective data writing) */
    void end_indep_data() {
      ncmpiwrap( ncmpi_end_indep_data(ncid) , __LINE__ );
    }


    /** @brief Serially write a scalar into a variable of length one */
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,int>::type = 0 >
    void write(T val, std::string varName ) {
      int varid = get_var_id( varName );
      pnetcdf_put_var( ncid ,  varid , &val );
    }


    /** @brief Serially read a scalar into a variable of length one */
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,int>::type = 0 >
    void read(T &val, std::string varName ) {
      int varid = get_var_id( varName );
      pnetcdf_get_var( ncid ,  varid , &val );
    }


    /** @brief Serially write an entire Array at once */
    template <class T, int rank, int myMem, int myStyle>
    void write(Array<T,rank,myMem,myStyle> const &arr , std::string varName) {
      int varid = get_var_id( varName );
      pnetcdf_put_var( ncid ,  varid , arr.createHostCopy().data() );
    }


    /** @brief Serially read an entire Array at once */
    template <class T, int rank, int myMem, int myStyle>
    void read(Array<T,rank,myMem,myStyle> const &arr_in , std::string varName) {
      int varid = get_var_id( varName );
      Array<T,rank,memHost,myStyle> arr;
      if constexpr (myMem == memDevice) { arr = arr_in.createHostObject(); }
      else                              { arr = arr_in;                    }
      pnetcdf_get_var( ncid ,  varid , arr.data() );
      if (myMem == memDevice) { arr.deep_copy_to(arr_in); }
    }


    /** @brief Collectively write an entire Array at once */
    template <class T, int rank, int myMem, int myStyle>
    void write_all(Array<T,rank,myMem,myStyle> const &arr , std::string varName , std::vector<MPI_Offset> start ) {
      if (rank != start   .size()) { yakl_throw("start.size() != Array's rank"); }
      std::vector<MPI_Offset> count(rank);
      for (int i=0; i < rank; i++) { count[i] = arr.extent(i); }
      int varid = get_var_id(varName);
      pnetcdf_put_vara_all( ncid ,  varid , start.data() , count.data() , arr.createHostCopy().data() );
    }


    /** @brief Collectively read an entire Array at once ... in pieces? */
    template <class T, int rank, int myMem, int myStyle>
    void read_all(Array<T,rank,myMem,myStyle> const &arr_in , std::string varName , std::vector<MPI_Offset> start ) {
      if (rank != start   .size()) { yakl_throw("start.size() != Array's rank"); }
      Array<T,rank,memHost,myStyle> arr;
      if constexpr (myMem == memDevice) { arr = arr_in.createHostObject(); }
      else                              { arr = arr_in;                    }
      std::vector<MPI_Offset> count(rank);
      for (int i=0; i < rank; i++) { count[i] = arr.extent(i); }
      int varid = get_var_id(varName);
      pnetcdf_get_vara_all( ncid ,  varid , start.data() , count.data() , arr.data() );
      if (myMem == memDevice) { arr.deep_copy_to(arr_in); }
    }


    /** @brief Serially write one entry of a scalar into the unlimited index */
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,int>::type = 0 >
    void write1(T val , std::string varName , int ind , std::string ulDimName="unlim" ) {
      int varid = get_var_id(varName);
      std::vector<MPI_Offset> start(1);
      std::vector<MPI_Offset> count(1);
      start[0] = ind;
      count[0] = 1;
      pnetcdf_put_vara( ncid ,  varid , start.data() , count.data() , &val );
    }


    /** @brief Serially write one entry of an Array into the unlimited index */
    template <class T, int rank, int myMem, int myStyle>
    void write1(Array<T,rank,myMem,myStyle> const &arr , std::string varName ,
                int ind , std::string ulDimName="unlim" ) {
      std::vector<MPI_Offset> start(rank+1);
      std::vector<MPI_Offset> count(rank+1);
      start[0] = ind;
      count[0] = 1;
      for (int i=1; i < rank+1; i++) {
        start[i] = 0;
        count[i] = arr.extent(i-1);
      }
      int varid = get_var_id(varName);
      pnetcdf_put_vara( ncid ,  varid , start.data() , count.data() , arr.createHostCopy().data() );
    }


    /** @brief Collectively write one entry of an Array into the unlimited index */
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
        count[i] = arr.extent(i-1);
      }
      int varid = get_var_id(varName);
      pnetcdf_put_vara_all( ncid ,  varid , start.data() , count.data() , arr.createHostCopy().data() );
    }


    /***************************************************************************************************
    Determine the type of a template T
    ***************************************************************************************************/
    /** @private */
    template <class T> nc_type getType() const {
           if ( std::is_same<typename std::remove_cv<T>::type ,          char>::value ) { return NC_CHAR;   }
      else if ( std::is_same<typename std::remove_cv<T>::type ,unsigned  char>::value ) { return NC_UBYTE;  }
      else if ( std::is_same<typename std::remove_cv<T>::type ,         short>::value ) { return NC_SHORT;  }
      else if ( std::is_same<typename std::remove_cv<T>::type ,unsigned short>::value ) { return NC_USHORT; }
      else if ( std::is_same<typename std::remove_cv<T>::type ,           int>::value ) { return NC_INT;    }
      else if ( std::is_same<typename std::remove_cv<T>::type ,unsigned   int>::value ) { return NC_UINT;   }
      else if ( std::is_same<typename std::remove_cv<T>::type ,          long>::value ) { return NC_INT64;  }
      else if ( std::is_same<typename std::remove_cv<T>::type ,unsigned  long>::value ) { return NC_UINT64; }
      else if ( std::is_same<typename std::remove_cv<T>::type ,         float>::value ) { return NC_FLOAT;  }
      else if ( std::is_same<typename std::remove_cv<T>::type ,        double>::value ) { return NC_DOUBLE; }
      else { yakl_throw("Invalid type"); }
      return -1;
    }

  };

}
__YAKL_NAMESPACE_WRAPPER_BEGIN__


