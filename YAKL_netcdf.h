
#pragma once

#include <netcdf>
#include <vector>
#include "YAKL.h"
using namespace netCDF;

namespace yakl {

  NcFile::FileMode constexpr MODE_READ    = NcFile::read;
  NcFile::FileMode constexpr MODE_WRITE   = NcFile::write;
  NcFile::FileMode constexpr MODE_REPLACE = NcFile::replace;
  NcFile::FileMode constexpr MODE_NEW     = NcFile::newFile;

  class SimpleNetCDF {
  protected:

    NcFile file;

  public:

    SimpleNetCDF() {};
    ~SimpleNetCDF() { close(); }


    void open  (std::string fname , NcFile::FileMode mode=NcFile::read   ) { file.open(fname,mode); }


    void create(std::string fname , NcFile::FileMode mode=NcFile::replace) { file.open(fname,mode); }


    void close() { file.close(); }


    bool varExists( std::string varName ) { return ! file.getVar(varName).isNull(); }


    bool dimExists( std::string dimName ) { return ! file.getDim(dimName).isNull(); }


    int getDimSize( std::string dimName ) { return file.getDim(dimName).getSize(); }


    template <class T, int rank, int myMem, int myStyle>
    void write(Array<T,rank,myMem,myStyle> const &arr , std::string varName , std::vector<std::string> dimNames) {
      if (rank != dimNames.size()) { throw "dimNames.size() != Array's rank"; }
      std::vector<NcDim> dims(rank);
      // Make sure the dimensions are in there and are the right sizes
      for (int i=0; i<rank; i++) {
        auto dimLoc = file.getDim( dimNames[i] );
        // If dimension doesn't exist, create it; otherwise, make sure it's the right size
        if ( dimLoc.isNull() ) {
          dims[i] = file.addDim( dimNames[i] , arr.dimension[i] );
        } else {
          if (dimLoc.getSize() != arr.dimension[i]) { throw("dimension size differs from the file"); }
          dims[i] = dimLoc;
        }
      }
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        var = file.addVar( varName , getType<T>() , dims );
      } else {
        if ( var.getType() != getType<T>() ) { throw "Existing variable's type != array's type"; }
        auto varDims = var.getDims();
        if (varDims.size() != rank) { throw "Existing variable's rank != array's rank"; }
        for (int i=0; i < varDims.size(); i++) {
          if (varDims[i].getSize() != arr.dimension[i]) {
            throw "Existing variable's dimension sizes are not the same as the array's";
          }
        }
      }

      if (myMem == memDevice) {
        var.putVar(arr.createHostCopy().data());
      } else {
        var.putVar(arr.data());
      }
    }


    template <class T, int rank, int myMem, int myStyle>
    void read(Array<T,rank,myMem,myStyle> &arr , std::string varName) {
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      if ( ! var.isNull() ) {
        // if ( var.getType() != getType<T>() ) { throw "Existing variable's type != array's type"; }
        auto varDims = var.getDims();
        if (varDims.size() != rank) { throw "Existing variable's rank != array's rank"; }
        std::vector<int> dimSizes(rank);
        for (int i=0; i < varDims.size(); i++) { dimSizes[i] = varDims[i].getSize(); }
        bool createArr = ! arr.initialized();
        if (arr.initialized()) {
          for (int i=0; i < dimSizes.size(); i++) {
            if (dimSizes[i] != arr.dimension[i]) {
              #ifdef YAKL_DEBUG
                std::cout << "WARNING: Array dims wrong size; deallocating previous array and allocating a new one\n";
              #endif
              createArr = true;
            }
          }
        }
        if (createArr) { arr = Array<T,rank,myMem,myStyle>(varName.c_str(),dimSizes); }
      } else { throw "Variable does not exist"; }

      if (myMem == memDevice) {
        auto arrHost = arr.createHostCopy();
        var.getVar(arrHost.data());
        arrHost.deep_copy_to(arr);
      } else {
        var.getVar(arr.data());
      }
    }


    template <class T> NcType getType() {
           if ( std::is_same<T,          char>::value ) { return ncChar;   }
      else if ( std::is_same<T,unsigned  char>::value ) { return ncUbyte;  }
      else if ( std::is_same<T,         short>::value ) { return ncShort;  }
      else if ( std::is_same<T,unsigned short>::value ) { return ncUshort; }
      else if ( std::is_same<T,           int>::value ) { return ncInt;    }
      else if ( std::is_same<T,unsigned   int>::value ) { return ncUint;   }
      else if ( std::is_same<T,          long>::value ) { return ncInt64;  }
      else if ( std::is_same<T,unsigned  long>::value ) { return ncUint64; }
      else if ( std::is_same<T,         float>::value ) { return ncFloat;  }
      else if ( std::is_same<T,        double>::value ) { return ncDouble; }
      else if ( std::is_same<T,std::string   >::value ) { return ncString; }
      else { throw "Invalid type"; }
    }

  };



}


