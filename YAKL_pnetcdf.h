
#pragma once

#include <vector>
#include "YAKL.h"
#include "mpi.h"
#include <pnetcdf_fixed>

namespace yakl {

  //Error reporting routine for the PNetCDF I/O
  inline void ncwrap( int ierr , int line ) {
    if (ierr != NC_NOERR) {
      printf("NetCDF Error at line: %d\n", line);
      printf("%s\n",ncmpi_strerror(ierr));
      throw "";
    }
  }

  using namespace PnetCDF;

  NcmpiFile::FileMode constexpr PNETCDF_MODE_READ    = NcmpiFile::read;
  NcmpiFile::FileMode constexpr PNETCDF_MODE_WRITE   = NcmpiFile::write;
  NcmpiFile::FileMode constexpr PNETCDF_MODE_REPLACE = NcmpiFile::replace;
  NcmpiFile::FileMode constexpr PNETCDF_MODE_NEW     = NcmpiFile::newFile;

  class SimplePNetCDF {
  protected:

    NcmpiFile *file;

  public:

    SimplePNetCDF() {
      file = nullptr;
    }

    ~SimplePNetCDF() {
      close();
    }


    void open  (std::string fname , NcmpiFile::FileMode mode=NcmpiFile::read   ) {
      close();
      file = new NcmpiFile(MPI_COMM_WORLD,fname,mode);
    }


    void create(std::string fname , NcmpiFile::FileMode mode=NcmpiFile::replace) {
      close();
      file = new NcmpiFile(MPI_COMM_WORLD,fname,mode);
    }


    void close() {
      if (file != nullptr) {
        delete file;
      }
      file = nullptr;
    }


    bool varExists( std::string varName ) const { return ! file->getVar(varName).isNull(); }


    bool dimExists( std::string dimName ) const { return ! file->getDim(dimName).isNull(); }


    int getDimSize( std::string dimName ) const { return file->getDim(dimName).getSize(); }


    template <class T>
    void createVar( std::string varName , std::vector<std::string> dnames ) {
      std::vector<NcmpiDim> dims(dnames.size());
      for (int i=0; i < dnames.size(); i++) {
        dims[i] = file->getDim(dnames[i]);
      }
      auto var = file->addVar( varName , getType<T>() , dims );
    }


    void createDim( std::string dimName , int len ) {
      file->addDim( dimName , len );
    }


    void createDim( std::string dimName ) {
      file->addDim( dimName );
    }


    void enddef() {
      int ncid = file->getId();
      ncwrap( ncmpi_enddef(ncid) , __LINE__ );
    }


    void begin_indep_data() {
      int ncid = file->getId();
      ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
    }


    void end_indep_data() {
      int ncid = file->getId();
      ncwrap( ncmpi_end_indep_data(ncid) , __LINE__ );
    }


    /***************************************************************************************************
    Serially write an entire Array at once
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void write(Array<T,rank,myMem,myStyle> const &arr , std::string varName) {
      auto var = file->getVar(varName);

      if (myMem == memDevice) {
        var.putVar(arr.createHostCopy().data());
      } else {
        var.putVar(arr.data());
      }
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

      auto var = file->getVar(varName);

      if (myMem == memDevice) {
        var.putVar_all(start , count , arr.createHostCopy().data() );
      } else {
        var.putVar_all(start , count , arr.data() );
      }
    }


    /***************************************************************************************************
    Serially write one entry of a scalar into the unlimited index
    ***************************************************************************************************/
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,int>::type = 0 >
    void write1(T val , std::string varName , int ind , std::string ulDimName="unlim" ) {
      auto var = file->getVar(varName);

      std::vector<MPI_Offset> start(1);
      std::vector<MPI_Offset> count(1);
      start[0] = ind;
      count[0] = 1;

      var.putVar(start,count,&val);
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

      auto var = file->getVar(varName);

      if (myMem == memDevice) {
        var.putVar( start , count , arr.createHostCopy().data() );
      } else {
        var.putVar( start , count , arr.data() );
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

      auto var = file->getVar(varName);

      if (myMem == memDevice) {
        var.putVar_all(start,count,arr.createHostCopy().data());
      } else {
        var.putVar_all(start,count,arr.data());
      }
    }


    /***************************************************************************************************
    Serially read an entire Array
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void read(Array<T,rank,myMem,myStyle> &arr , std::string varName) {

      auto var = file->getVar(varName);

      if (myMem == memDevice) {
        auto arrHost = arr.createHostCopy();
        var.getVar(arrHost.data());
        arrHost.deep_copy_to(arr);
      } else {
        var.getVar(arr.data());
      }
    }


    /***************************************************************************************************
    Collectively read an entire Array
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void read_all(Array<T,rank,myMem,myStyle> &arr , std::string varName , std::vector<MPI_Offset> start ) {
      if (start.size() != rank) { yakl_throw("ERROR: start.size() != arr's rank"); }

      auto var = file->getVar(varName);

      std::vector<MPI_Offset> count(rank);
      for (int i=0; i < rank; i++) {
        count[i] = arr.dimension[i];
      }

      if (myMem == memDevice) {
        auto arrHost = arr.createHostCopy();
        var.getVar_all( start , count , arrHost.data() );
        arrHost.deep_copy_to(arr);
      } else {
        var.getVar_all( start , count , arr.data() );
      }
    }


    /***************************************************************************************************
    Read a single scalar value
    ***************************************************************************************************/
    template <class T>
    void read(T &arr , std::string varName) {
      auto var = file->getVar(varName);
      var.getVar(&arr);
    }


    /***************************************************************************************************
    Write a single scalar value
    ***************************************************************************************************/
    template <class T>
    void write(T arr , std::string varName) {
      auto var = file->getVar(varName);
      var.putVar(&arr);
    }


    /***************************************************************************************************
    Determine the type of a template T
    ***************************************************************************************************/
    template <class T> NcmpiType getType() const {
           if ( std::is_same<T,          char>::value ) { return ncmpiChar;   }
      else if ( std::is_same<T,unsigned  char>::value ) { return ncmpiUbyte;  }
      else if ( std::is_same<T,         short>::value ) { return ncmpiShort;  }
      else if ( std::is_same<T,unsigned short>::value ) { return ncmpiUshort; }
      else if ( std::is_same<T,           int>::value ) { return ncmpiInt;    }
      else if ( std::is_same<T,unsigned   int>::value ) { return ncmpiUint;   }
      else if ( std::is_same<T,          long>::value ) { return ncmpiInt64;  }
      else if ( std::is_same<T,unsigned  long>::value ) { return ncmpiUint64; }
      else if ( std::is_same<T,         float>::value ) { return ncmpiFloat;  }
      else if ( std::is_same<T,        double>::value ) { return ncmpiDouble; }
      else { yakl_throw("Invalid type"); }
      return -1;
    }

  };

}


