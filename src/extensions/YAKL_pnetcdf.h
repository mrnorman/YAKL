
#pragma once

#include <vector>
#include "YAKL.h"
#include "mpi.h"
#include <pnetcdf.h>
#include <limits>
#include <stdexcept>

namespace yakl {

  inline MPI_Offset pnetcdf_checked_mpi_offset(size_t value) {
    if (!std::in_range<MPI_Offset>(value)) {
      Kokkos::abort("ERROR: PNetCDF Array extent does not fit in MPI_Offset");
    }
    return static_cast<MPI_Offset>(value);
  }

  //Error reporting routine for the PNetCDF I/O
  inline void ncmpiwrap( int ierr , int line ) {
    if (ierr != NC_NOERR) {
      std::cerr << "PNetCDF ERROR at line: " << line << std::endl;
      throw std::runtime_error(ncmpi_strerror(ierr));
    }
  }


  inline size_t pnetcdf_var_element_count(int ncid, int varid) {
    int ndims;
    ncmpiwrap(ncmpi_inq_varndims(ncid,varid,&ndims),__LINE__);
    std::vector<int> dimids(ndims);
    if (ndims > 0) ncmpiwrap(ncmpi_inq_vardimid(ncid,varid,dimids.data()),__LINE__);
    size_t count = 1;
    for (int i=0; i < ndims; i++) {
      MPI_Offset extent;
      ncmpiwrap(ncmpi_inq_dimlen(ncid,dimids[i],&extent),__LINE__);
      if (!std::in_range<size_t>(extent)) {
        Kokkos::abort("ERROR: PNetCDF variable extent does not fit in size_t");
      }
      size_t const size = static_cast<size_t>(extent);
      if (size != 0 && count > std::numeric_limits<size_t>::max()/size) {
        Kokkos::abort("ERROR: PNetCDF variable element count overflows size_t");
      }
      count *= size;
    }
    return count;
  }


  inline size_t pnetcdf_hyperslab_element_count(int ncid, int varid, MPI_Offset const count[]) {
    int ndims;
    ncmpiwrap(ncmpi_inq_varndims(ncid,varid,&ndims),__LINE__);
    size_t num_elements = 1;
    for (int i=0; i < ndims; i++) {
      if (!std::in_range<size_t>(count[i])) {
        Kokkos::abort("ERROR: PNetCDF hyperslab extent does not fit in size_t");
      }
      size_t const size = static_cast<size_t>(count[i]);
      if (size != 0 && num_elements > std::numeric_limits<size_t>::max()/size) {
        Kokkos::abort("ERROR: PNetCDF hyperslab element count overflows size_t");
      }
      num_elements *= size;
    }
    return num_elements;
  }


  template <class To, class From>
  std::vector<To> pnetcdf_convert_buffer(From const *data, size_t count) {
    std::vector<To> converted(count);
    for (size_t i=0; i < count; i++) converted[i] = static_cast<To>(data[i]);
    return converted;
  }


  inline void pnetcdf_copy_unsigned_long(unsigned long *data, std::vector<unsigned long long> const &converted) {
    for (size_t i=0; i < converted.size(); i++) {
      if (converted[i] > std::numeric_limits<unsigned long>::max()) {
        Kokkos::abort("ERROR: PNetCDF value does not fit in unsigned long");
      }
      data[i] = static_cast<unsigned long>(converted[i]);
    }
  }


  inline void pnetcdf_copy_bool(bool *data, std::vector<unsigned char> const &converted) {
    for (size_t i=0; i < converted.size(); i++) data[i] = converted[i] != 0;
  }


  //////////////////////////////////////////////
  // ncmpi_put_var
  //////////////////////////////////////////////
  inline void pnetcdf_put_var(int ncid , int varid , char const *data) {
    ncmpiwrap( ncmpi_put_var_text( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , signed char const *data) {
    ncmpiwrap( ncmpi_put_var_schar( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_var_uchar( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , short const *data) {
    ncmpiwrap( ncmpi_put_var_short( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_var_ushort( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , int const *data) {
    ncmpiwrap( ncmpi_put_var_int( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_var_uint( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , long const *data) {
    ncmpiwrap( ncmpi_put_var_long( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , unsigned long const *data) {
    auto converted = pnetcdf_convert_buffer<unsigned long long>(data,pnetcdf_var_element_count(ncid,varid));
    ncmpiwrap( ncmpi_put_var_ulonglong( ncid , varid , converted.data() ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , long long const *data) {
    ncmpiwrap( ncmpi_put_var_longlong( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_var_ulonglong( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , float const *data) {
    ncmpiwrap( ncmpi_put_var_float( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , double const *data) {
    ncmpiwrap( ncmpi_put_var_double( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var(int ncid , int varid , bool const *data) {
    auto converted = pnetcdf_convert_buffer<unsigned char>(data,pnetcdf_var_element_count(ncid,varid));
    ncmpiwrap( ncmpi_put_var_uchar( ncid , varid , converted.data() ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_var1
  //////////////////////////////////////////////
  inline void pnetcdf_put_var1(int ncid , int varid , char const *data) {
    ncmpiwrap( ncmpi_put_var1_text( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , signed char const *data) {
    ncmpiwrap( ncmpi_put_var1_schar( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_var1_uchar( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , short const *data) {
    ncmpiwrap( ncmpi_put_var1_short( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_var1_ushort( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , int const *data) {
    ncmpiwrap( ncmpi_put_var1_int( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_var1_uint( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , long const *data) {
    ncmpiwrap( ncmpi_put_var1_long( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned long const *data) {
    unsigned long long const converted = static_cast<unsigned long long>(*data);
    ncmpiwrap( ncmpi_put_var1_ulonglong( ncid , varid , 0 , &converted ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , long long const *data) {
    ncmpiwrap( ncmpi_put_var1_longlong( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_var1_ulonglong( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , float const *data) {
    ncmpiwrap( ncmpi_put_var1_float( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , double const *data) {
    ncmpiwrap( ncmpi_put_var1_double( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_put_var1(int ncid , int varid , bool const *data) {
    unsigned char const converted = *data ? 1 : 0;
    ncmpiwrap( ncmpi_put_var1_uchar( ncid , varid , 0 , &converted ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_vara
  //////////////////////////////////////////////
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               char const *data) {
    ncmpiwrap( ncmpi_put_vara_text( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char const *data) {
    ncmpiwrap( ncmpi_put_vara_schar( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_vara_uchar( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short const *data) {
    ncmpiwrap( ncmpi_put_vara_short( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_vara_ushort( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int const *data) {
    ncmpiwrap( ncmpi_put_vara_int( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_vara_uint( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               long const *data) {
    ncmpiwrap( ncmpi_put_vara_long( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               unsigned long const *data) {
    auto converted = pnetcdf_convert_buffer<unsigned long long>(data,pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_put_vara_ulonglong( ncid , varid , start , count , converted.data() ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long const *data) {
    ncmpiwrap( ncmpi_put_vara_longlong( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_vara_ulonglong( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float const *data) {
    ncmpiwrap( ncmpi_put_vara_float( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double const *data) {
    ncmpiwrap( ncmpi_put_vara_double( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               bool const *data) {
    auto converted = pnetcdf_convert_buffer<unsigned char>(data,pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_put_vara_uchar( ncid , varid , start , count , converted.data() ) , __LINE__ );
  }


  //////////////////////////////////////////////
  // ncmpi_put_vara_all
  //////////////////////////////////////////////
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   char const *data) {
    ncmpiwrap( ncmpi_put_vara_text_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char const *data) {
    ncmpiwrap( ncmpi_put_vara_schar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char const *data) {
    ncmpiwrap( ncmpi_put_vara_uchar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short const *data) {
    ncmpiwrap( ncmpi_put_vara_short_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short const *data) {
    ncmpiwrap( ncmpi_put_vara_ushort_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int const *data) {
    ncmpiwrap( ncmpi_put_vara_int_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int const *data) {
    ncmpiwrap( ncmpi_put_vara_uint_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   long const *data) {
    ncmpiwrap( ncmpi_put_vara_long_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   unsigned long const *data) {
    auto converted = pnetcdf_convert_buffer<unsigned long long>(data,pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_put_vara_ulonglong_all( ncid , varid , start , count , converted.data() ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long const *data) {
    ncmpiwrap( ncmpi_put_vara_longlong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long const *data) {
    ncmpiwrap( ncmpi_put_vara_ulonglong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float const *data) {
    ncmpiwrap( ncmpi_put_vara_float_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double const *data) {
    ncmpiwrap( ncmpi_put_vara_double_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_put_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   bool const *data) {
    auto converted = pnetcdf_convert_buffer<unsigned char>(data,pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_put_vara_uchar_all( ncid , varid , start , count , converted.data() ) , __LINE__ );
  }



  //////////////////////////////////////////////
  // ncmpi_get_var
  //////////////////////////////////////////////
  inline void pnetcdf_get_var(int ncid , int varid , char *data) {
    ncmpiwrap( ncmpi_get_var_text( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , signed char *data) {
    ncmpiwrap( ncmpi_get_var_schar( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , unsigned char *data) {
    ncmpiwrap( ncmpi_get_var_uchar( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , short *data) {
    ncmpiwrap( ncmpi_get_var_short( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , unsigned short *data) {
    ncmpiwrap( ncmpi_get_var_ushort( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , int *data) {
    ncmpiwrap( ncmpi_get_var_int( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , unsigned int *data) {
    ncmpiwrap( ncmpi_get_var_uint( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , long *data) {
    ncmpiwrap( ncmpi_get_var_long( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , unsigned long *data) {
    std::vector<unsigned long long> converted(pnetcdf_var_element_count(ncid,varid));
    ncmpiwrap( ncmpi_get_var_ulonglong( ncid , varid , converted.data() ) , __LINE__ );
    pnetcdf_copy_unsigned_long(data,converted);
  }
  inline void pnetcdf_get_var(int ncid , int varid , long long *data) {
    ncmpiwrap( ncmpi_get_var_longlong( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_var_ulonglong( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , float *data) {
    ncmpiwrap( ncmpi_get_var_float( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , double *data) {
    ncmpiwrap( ncmpi_get_var_double( ncid , varid , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var(int ncid , int varid , bool *data) {
    std::vector<unsigned char> converted(pnetcdf_var_element_count(ncid,varid));
    ncmpiwrap( ncmpi_get_var_uchar( ncid , varid , converted.data() ) , __LINE__ );
    pnetcdf_copy_bool(data,converted);
  }


  //////////////////////////////////////////////
  // ncmpi_get_var1
  //////////////////////////////////////////////
  inline void pnetcdf_get_var1(int ncid , int varid , char *data) {
    ncmpiwrap( ncmpi_get_var1_text( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , signed char *data) {
    ncmpiwrap( ncmpi_get_var1_schar( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned char *data) {
    ncmpiwrap( ncmpi_get_var1_uchar( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , short *data) {
    ncmpiwrap( ncmpi_get_var1_short( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned short *data) {
    ncmpiwrap( ncmpi_get_var1_ushort( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , int *data) {
    ncmpiwrap( ncmpi_get_var1_int( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned int *data) {
    ncmpiwrap( ncmpi_get_var1_uint( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , long *data) {
    ncmpiwrap( ncmpi_get_var1_long( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned long *data) {
    unsigned long long converted;
    ncmpiwrap( ncmpi_get_var1_ulonglong( ncid , varid , 0 , &converted ) , __LINE__ );
    if (converted > std::numeric_limits<unsigned long>::max()) {
      Kokkos::abort("ERROR: PNetCDF value does not fit in unsigned long");
    }
    *data = static_cast<unsigned long>(converted);
  }
  inline void pnetcdf_get_var1(int ncid , int varid , long long *data) {
    ncmpiwrap( ncmpi_get_var1_longlong( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_var1_ulonglong( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , float *data) {
    ncmpiwrap( ncmpi_get_var1_float( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , double *data) {
    ncmpiwrap( ncmpi_get_var1_double( ncid , varid , 0 , data ) , __LINE__ );
  }
  inline void pnetcdf_get_var1(int ncid , int varid , bool *data) {
    unsigned char converted;
    ncmpiwrap( ncmpi_get_var1_uchar( ncid , varid , 0 , &converted ) , __LINE__ );
    *data = converted != 0;
  }


  //////////////////////////////////////////////
  // ncmpi_get_vara
  //////////////////////////////////////////////
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               char *data) {
    ncmpiwrap( ncmpi_get_vara_text( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char *data) {
    ncmpiwrap( ncmpi_get_vara_schar( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char *data) {
    ncmpiwrap( ncmpi_get_vara_uchar( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short *data) {
    ncmpiwrap( ncmpi_get_vara_short( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short *data) {
    ncmpiwrap( ncmpi_get_vara_ushort( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int *data) {
    ncmpiwrap( ncmpi_get_vara_int( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int *data) {
    ncmpiwrap( ncmpi_get_vara_uint( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               long *data) {
    ncmpiwrap( ncmpi_get_vara_long( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               unsigned long *data) {
    std::vector<unsigned long long> converted(pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_get_vara_ulonglong( ncid , varid , start , count , converted.data() ) , __LINE__ );
    pnetcdf_copy_unsigned_long(data,converted);
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long *data) {
    ncmpiwrap( ncmpi_get_vara_longlong( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_vara_ulonglong( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float *data) {
    ncmpiwrap( ncmpi_get_vara_float( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double *data) {
    ncmpiwrap( ncmpi_get_vara_double( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                               bool *data) {
    std::vector<unsigned char> converted(pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_get_vara_uchar( ncid , varid , start , count , converted.data() ) , __LINE__ );
    pnetcdf_copy_bool(data,converted);
  }


  //////////////////////////////////////////////
  // ncmpi_get_vara_all
  //////////////////////////////////////////////
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   char *data) {
    ncmpiwrap( ncmpi_get_vara_text_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , signed char *data) {
    ncmpiwrap( ncmpi_get_vara_schar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned char *data) {
    ncmpiwrap( ncmpi_get_vara_uchar_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , short *data) {
    ncmpiwrap( ncmpi_get_vara_short_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned short *data) {
    ncmpiwrap( ncmpi_get_vara_ushort_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , int *data) {
    ncmpiwrap( ncmpi_get_vara_int_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned int *data) {
    ncmpiwrap( ncmpi_get_vara_uint_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   long *data) {
    ncmpiwrap( ncmpi_get_vara_long_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   unsigned long *data) {
    std::vector<unsigned long long> converted(pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_get_vara_ulonglong_all( ncid , varid , start , count , converted.data() ) , __LINE__ );
    pnetcdf_copy_unsigned_long(data,converted);
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , long long *data) {
    ncmpiwrap( ncmpi_get_vara_longlong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , unsigned long long *data) {
    ncmpiwrap( ncmpi_get_vara_ulonglong_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , float *data) {
    ncmpiwrap( ncmpi_get_vara_float_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] , double *data) {
    ncmpiwrap( ncmpi_get_vara_double_all( ncid , varid , start , count , data ) , __LINE__ );
  }
  inline void pnetcdf_get_vara_all(int ncid , int varid , MPI_Offset const start[] , MPI_Offset const count[] ,
                                   bool *data) {
    std::vector<unsigned char> converted(pnetcdf_hyperslab_element_count(ncid,varid,count));
    ncmpiwrap( ncmpi_get_vara_uchar_all( ncid , varid , start , count , converted.data() ) , __LINE__ );
    pnetcdf_copy_bool(data,converted);
  }




  struct SimplePNetCDF {
    int static constexpr MODE_UNOPENED   = -1;
    int static constexpr MODE_DEFINE     = 0;
    int static constexpr MODE_DATA_COLL  = 1;
    int static constexpr MODE_DATA_INDEP = 2;
    int       ncid;
    MPI_Comm  comm;
    int       mode;

    template <class T>
    void check_scalar_variable(int varid) const {
      nc_type xtype;
      int ndims;
      ncmpiwrap(ncmpi_inq_vartype(ncid,varid,&xtype),__LINE__);
      ncmpiwrap(ncmpi_inq_varndims(ncid,varid,&ndims),__LINE__);
      if (xtype != getType<T>() || ndims != 0) {
        Kokkos::abort("ERROR: PNetCDF scalar variable has incompatible type or rank");
      }
    }

    template <class ViewType>
    void check_array_variable(int varid, ViewType const & arr, std::vector<MPI_Offset> const * start = nullptr,
                              bool writing = false) const {
      nc_type xtype;
      int ndims;
      ncmpiwrap(ncmpi_inq_vartype(ncid,varid,&xtype),__LINE__);
      ncmpiwrap(ncmpi_inq_varndims(ncid,varid,&ndims),__LINE__);
      if (xtype != getType<typename ViewType::non_const_value_type>() || ndims != ViewType::rank()) {
        Kokkos::abort("ERROR: PNetCDF variable and Array have incompatible type or rank");
      }
      std::vector<int> dimids(ndims);
      ncmpiwrap(ncmpi_inq_vardimid(ncid,varid,dimids.data()),__LINE__);
      int unlimited = -1;
      ncmpiwrap(ncmpi_inq_unlimdim(ncid,&unlimited),__LINE__);
      for (int i=0; i < ndims; i++) {
        MPI_Offset dim_size;
        ncmpiwrap(ncmpi_inq_dimlen(ncid,dimids[i],&dim_size),__LINE__);
        // NetCDF lists dimensions slowest-to-fastest. Array uses that order, while Array_F stores it in reverse.
        int const array_dim = ViewType::is_cstyle ? i : ndims-1-i;
        MPI_Offset const extent = pnetcdf_checked_mpi_offset(arr.extent(array_dim));
        MPI_Offset const begin = start == nullptr ? 0 : (*start)[i];
        if (extent > std::numeric_limits<MPI_Offset>::max()-begin) {
          Kokkos::abort("ERROR: PNetCDF hyperslab bound overflow");
        }
        if (!(writing && dimids[i] == unlimited) && begin+extent > dim_size) {
          Kokkos::abort("ERROR: PNetCDF Array or hyperslab exceeds its variable dimensions");
        }
        if (start == nullptr && extent != dim_size) {
          Kokkos::abort("ERROR: PNetCDF Array and variable dimension sizes differ");
        }
      }
    }

    SimplePNetCDF(MPI_Comm comm = MPI_COMM_WORLD) : comm(comm) , ncid(-1) , mode(MODE_UNOPENED) {
      if (comm == MPI_COMM_NULL) Kokkos::abort("ERROR: SimplePNetCDF received MPI_COMM_NULL");
    }
    SimplePNetCDF(SimplePNetCDF const &) = delete;
    SimplePNetCDF &operator=(SimplePNetCDF const &) = delete;
    SimplePNetCDF(SimplePNetCDF &&in) noexcept : ncid(in.ncid) , comm(in.comm) , mode(in.mode) {
      in.ncid = -1;
      in.mode = MODE_UNOPENED;
    }
    SimplePNetCDF &operator=(SimplePNetCDF &&in) {
      if (this != &in) {
        close();
        ncid = in.ncid;
        comm = in.comm;
        mode = in.mode;
        in.ncid = -1;
        in.mode = MODE_UNOPENED;
      }
      return *this;
    }
    ~SimplePNetCDF() noexcept {
      if (mode == MODE_DATA_INDEP) {
        int const ierr = ncmpi_end_indep_data(ncid);
        if (ierr != NC_NOERR) std::fprintf(stderr,"PNetCDF destructor error: %s\n",ncmpi_strerror(ierr));
      }
      if (mode != MODE_UNOPENED) {
        int const ierr = ncmpi_close(ncid);
        if (ierr != NC_NOERR) std::fprintf(stderr,"PNetCDF destructor error: %s\n",ncmpi_strerror(ierr));
      }
    }


    // All MPI tasks in the Comm must call this
    void open(std::string fname , int omode = NC_WRITE , MPI_Info info = MPI_INFO_NULL ) {
      if (fname.empty()) Kokkos::abort("ERROR: PNetCDF filename cannot be empty");
      close();
      ncmpiwrap( ncmpi_open( comm , fname.c_str() , omode , info , &ncid ) , __LINE__ );
      mode = MODE_DATA_COLL;
    }


    // All MPI tasks in the Comm must call this
    void create(std::string fname , int flag = NC_CLOBBER , MPI_Info info = MPI_INFO_NULL ) {
      if (fname.empty()) Kokkos::abort("ERROR: PNetCDF filename cannot be empty");
      close();
      ncmpiwrap( ncmpi_create( comm , fname.c_str() , flag , info , &ncid ) , __LINE__ );
      mode = MODE_DEFINE;
    }


    // All MPI tasks in the Comm must call this
    void close() {
      if (mode == MODE_DATA_INDEP) end_indep_data();
      if (mode != MODE_UNOPENED  ) ncmpiwrap( ncmpi_close(ncid) , __LINE__ );
      ncid = -1;
      mode = MODE_UNOPENED;
    }


    // Callable by only one task
    int get_dim_id( std::string dimName ) const {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling get_dim_id on unopened file");
      if (dimName.empty()) Kokkos::abort("ERROR: PNetCDF dimension name cannot be empty");
      int dimid;
      ncmpiwrap( ncmpi_inq_dimid( ncid , dimName.c_str() , &dimid) , __LINE__ );
      return dimid;
    }


    // Callable by only one task
    int get_var_id( std::string varName ) const {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling get_var_id on unopened file");
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      int varid;
      ncmpiwrap( ncmpi_inq_varid( ncid , varName.c_str() , &varid) , __LINE__ );
      return varid;
    }


    // Callable by only one task
    bool var_exists( std::string varName ) const {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling var_exists on unopened file");
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      int varid;
      int ierr = ncmpi_inq_varid( ncid , varName.c_str() , &varid);
      return ierr == NC_NOERR;
    }


    // Callable by only one task
    bool dim_exists( std::string dimName ) const {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling dim_exists on unopened file");
      if (dimName.empty()) Kokkos::abort("ERROR: PNetCDF dimension name cannot be empty");
      int dimid;
      int ierr = ncmpi_inq_dimid( ncid , dimName.c_str() , &dimid);
      return ierr == NC_NOERR;
    }


    // Callable by only one task
    MPI_Offset get_dim_size( std::string dimName ) const {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling get_dim_size on unopened file");
      if (dimName.empty()) Kokkos::abort("ERROR: PNetCDF dimension name cannot be empty");
      int dimid;
      MPI_Offset dimlen;
      ncmpiwrap( ncmpi_inq_dimid ( ncid , dimName.c_str() , &dimid) , __LINE__ );
      ncmpiwrap( ncmpi_inq_dimlen( ncid , dimid , &dimlen ) , __LINE__ );
      return dimlen;
    }


    // All MPI tasks in the Comm must call this
    template <class T>
    void create_var( std::string varName , std::vector<std::string> dnames ) {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling create_var on unopened file");
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      for (auto const & name : dnames) if (name.empty()) Kokkos::abort("ERROR: PNetCDF dimension name cannot be empty");
      redef();
      int ndims = dnames.size();
      std::vector<int> dimids(ndims);
      for (int i=0; i < ndims; i++) { dimids[i] = get_dim_id( dnames[i] ); }
      nc_type xtype = getType<T>();
      int varid;
      ncmpiwrap( ncmpi_def_var( ncid , varName.c_str() , xtype , ndims , dimids.data() , &varid ) , __LINE__ );
    }


    // All MPI tasks in the Comm must call this
    void create_dim( std::string dimName , MPI_Offset len ) {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling create_dim on unopened file");
      if (dimName.empty()) Kokkos::abort("ERROR: PNetCDF dimension name cannot be empty");
      if (len < 0 && len != NC_UNLIMITED) Kokkos::abort("ERROR: PNetCDF dimension length cannot be negative");
      if (len == NC_UNLIMITED) {
        int unlimited = -1;
        ncmpiwrap(ncmpi_inq_unlimdim(ncid,&unlimited),__LINE__);
        if (unlimited >= 0) {
          throw std::runtime_error("ERROR: SimplePNetCDF supports only one unlimited dimension per file");
        }
      }
      redef();
      int dimid;
      ncmpiwrap( ncmpi_def_dim( ncid , dimName.c_str() , len , &dimid ) , __LINE__ );
    }


    // All MPI tasks in the Comm must call this
    void redef() {
      if (mode == MODE_UNOPENED  ) Kokkos::abort("ERROR: calling redef on unopened file");
      if (mode == MODE_DEFINE    ) return;
      if (mode == MODE_DATA_INDEP) end_indep_data();
      if (mode == MODE_DATA_COLL ) ncmpiwrap( ncmpi_redef(ncid) , __LINE__ );
      mode = MODE_DEFINE;
    }


    // All MPI tasks in the Comm must call this
    void enddef() {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: calling enddef on unopened file");
      if (mode == MODE_DEFINE  ) {
        ncmpiwrap( ncmpi_enddef(ncid) , __LINE__ );
        mode = MODE_DATA_COLL;
      }
    }


    // All MPI tasks in the Comm must call this
    void begin_indep_data() {
      if (mode == MODE_UNOPENED  ) Kokkos::abort("ERROR: calling begin_indep_data on unopened file");
      if (mode == MODE_DATA_INDEP) return;
      if (mode == MODE_DEFINE    ) enddef();
      if (mode == MODE_DATA_COLL ) ncmpiwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
      mode = MODE_DATA_INDEP;
    }


    // All MPI tasks in the Comm must call this
    void end_indep_data() {
      if (mode == MODE_UNOPENED  ) Kokkos::abort("ERROR: calling end_indep_data on unopened file");
      if (mode == MODE_DEFINE    ) enddef();
      if (mode == MODE_DATA_COLL ) return;
      if (mode == MODE_DATA_INDEP) {
        ncmpiwrap( ncmpi_end_indep_data(ncid) , __LINE__ );
        mode = MODE_DATA_COLL;
      }
    }


    // Callable by only one task
    template <class T> requires std::is_arithmetic_v<T>
    void write(T val, std::string varName ) {
      if (mode == MODE_UNOPENED ) Kokkos::abort("ERROR: calling write on unopened file");
      if (mode == MODE_DEFINE   ) Kokkos::abort("ERROR: calling write on file in define mode"); // cannot call enddef b/c callable by single task
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      int varid = get_var_id( varName );
      check_scalar_variable<T>(varid);
      pnetcdf_put_var( ncid ,  varid , &val );
    }


    // Callable by only one task
    template <class T> requires std::is_arithmetic_v<T>
    void read(T &val, std::string varName ) {
      if (mode == MODE_UNOPENED ) Kokkos::abort("ERROR: calling read on unopened file");
      if (mode == MODE_DEFINE   ) Kokkos::abort("ERROR: calling read on file in define mode"); // cannot call enddef b/c callable by single task
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      int varid = get_var_id( varName );
      check_scalar_variable<T>(varid);
      pnetcdf_get_var( ncid ,  varid , &val );
    }


    // Callable by only one task
    template <class ViewType> requires is_Array<ViewType>
    void write(ViewType const & arr , std::string varName) {
      if (mode == MODE_UNOPENED ) Kokkos::abort("ERROR: calling write on unopened file");
      if (mode == MODE_DEFINE   ) Kokkos::abort("ERROR: calling write on file in define mode"); // cannot call enddef b/c callable by single task
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      if (!arr.is_allocated()) Kokkos::abort("ERROR: writing an unallocated Array with PNetCDF");
      int varid = get_var_id( varName );
      check_array_variable(varid,arr);
      if constexpr (ViewType::on_device) {
        auto arr_host = arr.createHostCopy();
        pnetcdf_put_var( ncid , varid , arr_host.data() );
      } else {
        pnetcdf_put_var( ncid , varid , arr.data() );
      }
    }


    // Callable by only one task
    template <class ViewType> requires is_Array<ViewType>
    void read(ViewType const & arr_in , std::string varName) {
      if (mode == MODE_UNOPENED ) Kokkos::abort("ERROR: calling read on unopened file");
      if (mode == MODE_DEFINE   ) Kokkos::abort("ERROR: calling read on file in define mode"); // cannot call enddef b/c callable by single task
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      if (!arr_in.is_allocated()) Kokkos::abort("ERROR: reading into an unallocated Array with PNetCDF");
      int varid = get_var_id( varName );
      check_array_variable(varid,arr_in);
      if constexpr (ViewType::on_device) {
        auto arr_host = arr_in.createHostObject();
        pnetcdf_get_var( ncid , varid , arr_host.data() );
        arr_host.deep_copy_to(arr_in);
      } else {
        pnetcdf_get_var( ncid , varid , arr_in.data() );
      }
    }


    // All MPI tasks in the Comm must call this
    template <class ViewType> requires is_Array<ViewType>
    void write_all(ViewType const & arr , std::string varName , std::vector<MPI_Offset> start ) {
      if (mode == MODE_UNOPENED  ) Kokkos::abort("ERROR: calling write_all on unopened file");
      if (mode == MODE_DEFINE    ) enddef();
      if (mode == MODE_DATA_INDEP) end_indep_data();
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      if (!arr.is_allocated()) Kokkos::abort("ERROR: writing an unallocated Array with PNetCDF");
      int constexpr rank = ViewType::rank();
      if (static_cast<size_t>(rank) != start.size()) { Kokkos::abort("start.size() != Array's rank"); }
      for (auto const offset : start) if (offset < 0) Kokkos::abort("ERROR: PNetCDF start offset cannot be negative");
      std::vector<MPI_Offset> count(rank);
      for (int i=0; i < rank; i++) {
        int const array_dim = ViewType::is_cstyle ? i : rank-1-i;
        count[i] = pnetcdf_checked_mpi_offset(arr.extent(array_dim));
      }
      int varid = get_var_id(varName);
      check_array_variable(varid,arr,&start,true);
      if constexpr (ViewType::on_device) {
        auto arr_host = arr.createHostCopy();
        pnetcdf_put_vara_all( ncid , varid , start.data() , count.data() , arr_host.data() );
      } else {
        pnetcdf_put_vara_all( ncid , varid , start.data() , count.data() , arr.data() );
      }
    }


    // All MPI tasks in the Comm must call this
    template <class ViewType> requires is_Array<ViewType>
    void read_all(ViewType const & arr_in , std::string varName , std::vector<MPI_Offset> start ) {
      if (mode == MODE_UNOPENED  ) Kokkos::abort("ERROR: calling read_all on unopened file");
      if (mode == MODE_DEFINE    ) enddef();
      if (mode == MODE_DATA_INDEP) end_indep_data();
      if (varName.empty()) Kokkos::abort("ERROR: PNetCDF variable name cannot be empty");
      if (!arr_in.is_allocated()) Kokkos::abort("ERROR: reading into an unallocated Array with PNetCDF");
      int constexpr rank = ViewType::rank();
      if (static_cast<size_t>(rank) != start   .size()) { Kokkos::abort("start.size() != Array's rank"); }
      for (auto const offset : start) if (offset < 0) Kokkos::abort("ERROR: PNetCDF start offset cannot be negative");
      std::vector<MPI_Offset> count(rank);
      for (int i=0; i < rank; i++) {
        int const array_dim = ViewType::is_cstyle ? i : rank-1-i;
        count[i] = pnetcdf_checked_mpi_offset(arr_in.extent(array_dim));
      }
      int varid = get_var_id(varName);
      check_array_variable(varid,arr_in,&start,false);
      if constexpr (ViewType::on_device) {
        auto arr_host = arr_in.createHostObject();
        pnetcdf_get_vara_all( ncid , varid , start.data() , count.data() , arr_host.data() );
        arr_host.deep_copy_to(arr_in);
      } else {
        pnetcdf_get_vara_all( ncid , varid , start.data() , count.data() , arr_in.data() );
      }
    }


  private:
    template <class T>
    void putAttributeData(int varid, std::string const & attName, T const *data, size_t count) {
      using U = std::remove_cv_t<T>;
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: writing an attribute to an unopened PNetCDF file");
      if (varid < NC_GLOBAL) Kokkos::abort("ERROR: PNetCDF attribute variable does not exist");
      if (attName.empty()) Kokkos::abort("ERROR: PNetCDF attribute name cannot be empty");
      if (!std::in_range<MPI_Offset>(count)) {
        Kokkos::abort("ERROR: PNetCDF attribute length does not fit in MPI_Offset");
      }
      redef();
      MPI_Offset const length = static_cast<MPI_Offset>(count);
      nc_type const xtype = std::is_same_v<U,bool> ? NC_UBYTE : getType<U>();
      if constexpr (std::is_same_v<U,char>) {
        ncmpiwrap(ncmpi_put_att_text(ncid,varid,attName.c_str(),length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,signed char>) {
        ncmpiwrap(ncmpi_put_att_schar(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned char>) {
        ncmpiwrap(ncmpi_put_att_uchar(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,short>) {
        ncmpiwrap(ncmpi_put_att_short(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned short>) {
        ncmpiwrap(ncmpi_put_att_ushort(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,int>) {
        ncmpiwrap(ncmpi_put_att_int(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned int>) {
        ncmpiwrap(ncmpi_put_att_uint(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,long>) {
        ncmpiwrap(ncmpi_put_att_long(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned long>) {
        std::vector<unsigned long long> converted(count);
        for (size_t i=0; i < count; i++) converted[i] = static_cast<unsigned long long>(data[i]);
        ncmpiwrap(ncmpi_put_att_ulonglong(ncid,varid,attName.c_str(),xtype,length,converted.data()),__LINE__);
      } else if constexpr (std::is_same_v<U,long long>) {
        ncmpiwrap(ncmpi_put_att_longlong(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned long long>) {
        ncmpiwrap(ncmpi_put_att_ulonglong(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,float>) {
        ncmpiwrap(ncmpi_put_att_float(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,double>) {
        ncmpiwrap(ncmpi_put_att_double(ncid,varid,attName.c_str(),xtype,length,data),__LINE__);
      } else if constexpr (std::is_same_v<U,bool>) {
        std::vector<unsigned char> converted(count);
        for (size_t i=0; i < count; i++) converted[i] = data[i] ? 1 : 0;
        ncmpiwrap(ncmpi_put_att_uchar(ncid,varid,attName.c_str(),xtype,length,converted.data()),__LINE__);
      }
    }


    template <class T>
    void getAttributeData(int varid, std::string const & attName, T *data) const {
      using U = std::remove_cv_t<T>;
      if constexpr (std::is_same_v<U,char>) {
        ncmpiwrap(ncmpi_get_att_text(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,signed char>) {
        ncmpiwrap(ncmpi_get_att_schar(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned char>) {
        ncmpiwrap(ncmpi_get_att_uchar(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,short>) {
        ncmpiwrap(ncmpi_get_att_short(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned short>) {
        ncmpiwrap(ncmpi_get_att_ushort(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,int>) {
        ncmpiwrap(ncmpi_get_att_int(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned int>) {
        ncmpiwrap(ncmpi_get_att_uint(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,long>) {
        ncmpiwrap(ncmpi_get_att_long(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned long>) {
        MPI_Offset count;
        ncmpiwrap(ncmpi_inq_attlen(ncid,varid,attName.c_str(),&count),__LINE__);
        std::vector<unsigned long long> converted(static_cast<size_t>(count));
        ncmpiwrap(ncmpi_get_att_ulonglong(ncid,varid,attName.c_str(),converted.data()),__LINE__);
        for (size_t i=0; i < converted.size(); i++) {
          if (converted[i] > std::numeric_limits<unsigned long>::max()) {
            Kokkos::abort("ERROR: PNetCDF attribute value does not fit in unsigned long");
          }
          data[i] = static_cast<unsigned long>(converted[i]);
        }
      } else if constexpr (std::is_same_v<U,long long>) {
        ncmpiwrap(ncmpi_get_att_longlong(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,unsigned long long>) {
        ncmpiwrap(ncmpi_get_att_ulonglong(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,float>) {
        ncmpiwrap(ncmpi_get_att_float(ncid,varid,attName.c_str(),data),__LINE__);
      } else if constexpr (std::is_same_v<U,double>) {
        ncmpiwrap(ncmpi_get_att_double(ncid,varid,attName.c_str(),data),__LINE__);
      }
    }


    template <class T>
    size_t checkAttribute(int varid, std::string const & attName) const {
      if (mode == MODE_UNOPENED) Kokkos::abort("ERROR: accessing an attribute in an unopened PNetCDF file");
      if (attName.empty()) Kokkos::abort("ERROR: PNetCDF attribute name cannot be empty");
      nc_type xtype;
      MPI_Offset count;
      ncmpiwrap(ncmpi_inq_att(ncid,varid,attName.c_str(),&xtype,&count),__LINE__);
      nc_type const expected = std::is_same_v<std::remove_cv_t<T>,bool> ? NC_UBYTE : getType<T>();
      if (xtype != expected) Kokkos::abort("ERROR: PNetCDF attribute type differs from the requested type");
      if (!std::in_range<size_t>(count)) {
        Kokkos::abort("ERROR: PNetCDF attribute length does not fit in size_t");
      }
      return static_cast<size_t>(count);
    }


  public:
    // Attribute writes can enter define mode, so every task in the communicator must call them.
    template <class T> requires std::is_arithmetic_v<T>
    void writeGlobalAttribute(T const &value, std::string attName) {
      putAttributeData(NC_GLOBAL,attName,&value,1);
    }


    template <class T> requires std::is_arithmetic_v<T>
    void writeGlobalAttribute(std::vector<T> const &values, std::string attName) {
      if constexpr (std::is_same_v<T,bool>) {
        std::vector<unsigned char> converted(values.size());
        for (size_t i=0; i < values.size(); i++) converted[i] = values[i] ? 1 : 0;
        putAttributeData(NC_GLOBAL,attName,converted.data(),converted.size());
      } else {
        putAttributeData(NC_GLOBAL,attName,values.data(),values.size());
      }
    }


    template <class T> requires std::is_arithmetic_v<T>
    void readGlobalAttribute(T &value, std::string attName) const {
      if (checkAttribute<T>(NC_GLOBAL,attName) != 1) Kokkos::abort("ERROR: PNetCDF attribute is not scalar");
      if constexpr (std::is_same_v<T,bool>) {
        unsigned char converted;
        getAttributeData(NC_GLOBAL,attName,&converted);
        value = converted != 0;
      } else {
        getAttributeData(NC_GLOBAL,attName,&value);
      }
    }


    template <class T> requires std::is_arithmetic_v<T>
    void readGlobalAttribute(std::vector<T> &values, std::string attName) const {
      size_t const count = checkAttribute<T>(NC_GLOBAL,attName);
      if constexpr (std::is_same_v<T,bool>) {
        std::vector<unsigned char> converted(count);
        getAttributeData(NC_GLOBAL,attName,converted.data());
        values.assign(count,false);
        for (size_t i=0; i < count; i++) values[i] = converted[i] != 0;
      } else {
        values.resize(count);
        getAttributeData(NC_GLOBAL,attName,values.data());
      }
    }


    template <class T> requires std::is_arithmetic_v<T>
    void writeVariableAttribute(T const &value, std::string varName, std::string attName) {
      putAttributeData(get_var_id(varName),attName,&value,1);
    }


    template <class T> requires std::is_arithmetic_v<T>
    void writeVariableAttribute(std::vector<T> const &values, std::string varName, std::string attName) {
      int const varid = get_var_id(varName);
      if constexpr (std::is_same_v<T,bool>) {
        std::vector<unsigned char> converted(values.size());
        for (size_t i=0; i < values.size(); i++) converted[i] = values[i] ? 1 : 0;
        putAttributeData(varid,attName,converted.data(),converted.size());
      } else {
        putAttributeData(varid,attName,values.data(),values.size());
      }
    }


    template <class T> requires std::is_arithmetic_v<T>
    void readVariableAttribute(T &value, std::string varName, std::string attName) const {
      int const varid = get_var_id(varName);
      if (checkAttribute<T>(varid,attName) != 1) Kokkos::abort("ERROR: PNetCDF attribute is not scalar");
      if constexpr (std::is_same_v<T,bool>) {
        unsigned char converted;
        getAttributeData(varid,attName,&converted);
        value = converted != 0;
      } else {
        getAttributeData(varid,attName,&value);
      }
    }


    template <class T> requires std::is_arithmetic_v<T>
    void readVariableAttribute(std::vector<T> &values, std::string varName, std::string attName) const {
      int const varid = get_var_id(varName);
      size_t const count = checkAttribute<T>(varid,attName);
      if constexpr (std::is_same_v<T,bool>) {
        std::vector<unsigned char> converted(count);
        getAttributeData(varid,attName,converted.data());
        values.assign(count,false);
        for (size_t i=0; i < count; i++) values[i] = converted[i] != 0;
      } else {
        values.resize(count);
        getAttributeData(varid,attName,values.data());
      }
    }


    template <class T> requires std::is_same_v<std::remove_cvref_t<T>,std::string>
    void writeGlobalAttribute(T const &value, std::string attName) {
      putAttributeData(NC_GLOBAL,attName,value.data(),value.size());
    }


    template <class T> requires std::is_same_v<std::remove_cvref_t<T>,std::string>
    void readGlobalAttribute(T &value, std::string attName) const {
      size_t const count = checkAttribute<char>(NC_GLOBAL,attName);
      value.resize(count);
      getAttributeData(NC_GLOBAL,attName,value.data());
    }


    template <class T> requires std::is_same_v<std::remove_cvref_t<T>,std::string>
    void writeVariableAttribute(T const &value, std::string varName, std::string attName) {
      putAttributeData(get_var_id(varName),attName,value.data(),value.size());
    }


    template <class T> requires std::is_same_v<std::remove_cvref_t<T>,std::string>
    void readVariableAttribute(T &value, std::string varName, std::string attName) const {
      int const varid = get_var_id(varName);
      size_t const count = checkAttribute<char>(varid,attName);
      value.resize(count);
      getAttributeData(varid,attName,value.data());
    }


    template <class T>
    T readGlobalAttribute(std::string attName) const {
      T value;
      readGlobalAttribute(value,attName);
      return value;
    }


    template <class T>
    T readVariableAttribute(std::string varName, std::string attName) const {
      T value;
      readVariableAttribute(value,varName,attName);
      return value;
    }


    /***************************************************************************************************
    Determine the type of a template T
    ***************************************************************************************************/
    template <class T> nc_type getType() const {
           if ( std::is_same_v<typename std::remove_cv_t<T> ,          char> ) { return NC_CHAR;   }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,signed    char> ) { return NC_BYTE;   }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,unsigned  char> ) { return NC_UBYTE;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,         short> ) { return NC_SHORT;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,unsigned short> ) { return NC_USHORT; }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,           int> ) { return NC_INT;    }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,unsigned   int> ) { return NC_UINT;   }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,          long> ) { return NC_INT64;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,unsigned  long> ) { return NC_UINT64; }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,     long long> ) { return NC_INT64;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,unsigned long long> ) { return NC_UINT64; }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,         float> ) { return NC_FLOAT;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,        double> ) { return NC_DOUBLE; }
      else if ( std::is_same_v<typename std::remove_cv_t<T> ,          bool> ) { return NC_UBYTE;  }
      else { Kokkos::abort("Invalid type"); }
      return -1;
    }

  };

}
