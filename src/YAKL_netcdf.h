
#pragma once

#include <netcdf.h>
#include <vector>
#include "YAKL.h"



namespace yakl {
  //Error reporting routine for the PNetCDF I/O
  inline void ncwrap( int ierr , int line ) {
    if (ierr != NC_NOERR) {
      printf("NetCDF Error at line: %d\n", line);
      printf("%s\n",nc_strerror(ierr));
      yakl_throw("");
    }
  }

  int constexpr NETCDF_MODE_READ    = NC_NOWRITE;
  int constexpr NETCDF_MODE_WRITE   = NC_WRITE;
  int constexpr NETCDF_MODE_REPLACE = NC_CLOBBER;
  int constexpr NETCDF_MODE_NEW     = NC_NOCLOBBER;

  class SimpleNetCDF {
  public:


    class NcDim {
    public:
      std::string name;
      size_t      len;
      int         id;
      bool        is_unlim;
      NcDim() {
        name = "";
        id = -999;
        len = 0;
        is_unlim = false;
      }
      ~NcDim() {}
      NcDim(std::string name, size_t len, int id, bool is_unlim) {
        this->name     = name;
        this->len      = len;
        this->id       = id;
        this->is_unlim = is_unlim;
      }
      NcDim(NcDim &&in) {
        this->name     = in.name;
        this->len      = in.len;
        this->id       = in.id;
        this->is_unlim = in.is_unlim;
      }
      NcDim(NcDim const &in) {
        this->name     = in.name;
        this->len      = in.len;
        this->id       = in.id;
        this->is_unlim = in.is_unlim;
      }
      NcDim &operator=(NcDim &&in) {
        this->name     = in.name;
        this->len      = in.len;
        this->id       = in.id;
        this->is_unlim = in.is_unlim;
        return *this;
      }
      NcDim &operator=(NcDim const &in) {
        this->name     = in.name;
        this->len      = in.len;
        this->id       = in.id;
        this->is_unlim = in.is_unlim;
        return *this;
      }
      std::string getName()                    const { return name; }
      size_t      getSize()                    const { return len; }
      int         getId()                      const { return id; }
      bool        isNull()                     const { return id == -999; }
      bool        operator==(NcDim const &rhs) const { return this->name == rhs.name && !isNull(); }
      bool        operator!=(NcDim const &rhs) const { return this->name != rhs.name || isNull(); }
      bool        isUnlimited()                const { return is_unlim; }
    };


    class NcVar {
    public:
      int                ncid;
      std::string        name;
      std::vector<NcDim> dims;
      int                id;
      int                type;
      NcVar() {
        ncid = -999;
        name = "";
        dims = std::vector<NcDim>(0);
        id   = -999;
        type = -999;
      }
      ~NcVar() {}
      NcVar(int ncid , std::string name, std::vector<NcDim> dims, int id, int type) {
        this->ncid = ncid;
        this->name = name;
        this->dims = dims;
        this->id   = id;
        this->type = type;
      }
      NcVar(NcVar &&in) {
        this->ncid = in.ncid;
        this->name = in.name;
        this->dims = in.dims;
        this->id   = in.id;
        this->type = in.type;
      }
      NcVar(NcVar const &in) {
        this->ncid = in.ncid;
        this->name = in.name;
        this->dims = in.dims;
        this->id   = in.id;
        this->type = in.type;
      }
      NcVar &operator=(NcVar &&in) {
        this->ncid = in.ncid;
        this->name = in.name;
        this->dims = in.dims;
        this->id   = in.id;
        this->type = in.type;
        return *this;
      }
      NcVar &operator=(NcVar const &in) {
        this->ncid = in.ncid;
        this->name = in.name;
        this->dims = in.dims;
        this->id   = in.id;
        this->type = in.type;
        return *this;
      }
      std::string        getName()                    const { return name; }
      std::vector<NcDim> getDims()                    const { return dims; }
      int                getDimCount()                const { return dims.size(); }
      int                getId()                      const { return id; }
      int                getType()                    const { return type; }
      bool               isNull ()                    const { return id == -999; }
      bool               operator==(NcDim const &rhs) const { return this->name == rhs.name && !isNull(); }
      bool               operator!=(NcDim const &rhs) const { return this->name != rhs.name || isNull(); }
      NcDim getDim(int i) const {
        if (isNull() || dims.size() <= i) {
          return NcDim();
        } else {
          return dims[i];
        }
      }

      void putVar(double             const *data) { ncwrap( nc_put_var_double   ( ncid , id , data ) , __LINE__ ); }
      void putVar(float              const *data) { ncwrap( nc_put_var_float    ( ncid , id , data ) , __LINE__ ); }
      void putVar(int                const *data) { ncwrap( nc_put_var_int      ( ncid , id , data ) , __LINE__ ); }
      void putVar(long               const *data) { ncwrap( nc_put_var_long     ( ncid , id , data ) , __LINE__ ); }
      void putVar(long long          const *data) { ncwrap( nc_put_var_longlong ( ncid , id , data ) , __LINE__ ); }
      void putVar(signed char        const *data) { ncwrap( nc_put_var_schar    ( ncid , id , data ) , __LINE__ ); }
      void putVar(short              const *data) { ncwrap( nc_put_var_short    ( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned char      const *data) { ncwrap( nc_put_var_uchar    ( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned int       const *data) { ncwrap( nc_put_var_uint     ( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned long      const *data) { ncwrap( nc_put_var_uint     ( ncid , id , (unsigned int const *) data ) , __LINE__ ); }
      void putVar(unsigned long long const *data) { ncwrap( nc_put_var_ulonglong( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned short     const *data) { ncwrap( nc_put_var_ushort   ( ncid , id , data ) , __LINE__ ); }
      void putVar(char               const *data) { ncwrap( nc_put_var_text     ( ncid , id , data ) , __LINE__ ); }
      void putVar(bool               const *data) { yakl_throw("ERROR: Cannot write bools to netCDF file"); }

      void putVar(std::vector<size_t> start , std::vector<size_t> count, double             const *data) { ncwrap( nc_put_vara_double   ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, float              const *data) { ncwrap( nc_put_vara_float    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, int                const *data) { ncwrap( nc_put_vara_int      ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, long               const *data) { ncwrap( nc_put_vara_long     ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, long long          const *data) { ncwrap( nc_put_vara_longlong ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, signed char        const *data) { ncwrap( nc_put_vara_schar    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, short              const *data) { ncwrap( nc_put_vara_short    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned char      const *data) { ncwrap( nc_put_vara_uchar    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned int       const *data) { ncwrap( nc_put_vara_uint     ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned long      const *data) { ncwrap( nc_put_vara_uint     ( ncid , id , start.data() , count.data(), (unsigned int const *) data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned long long const *data) { ncwrap( nc_put_vara_ulonglong( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned short     const *data) { ncwrap( nc_put_vara_ushort   ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, char               const *data) { ncwrap( nc_put_vara_text     ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, bool               const *data) { yakl_throw("ERROR: Cannot write bools to netCDF file"); }

      void getVar(double             *data) const { ncwrap( nc_get_var_double   ( ncid , id , data ) , __LINE__ ); }
      void getVar(float              *data) const { ncwrap( nc_get_var_float    ( ncid , id , data ) , __LINE__ ); }
      void getVar(int                *data) const { ncwrap( nc_get_var_int      ( ncid , id , data ) , __LINE__ ); }
      void getVar(long               *data) const { ncwrap( nc_get_var_long     ( ncid , id , data ) , __LINE__ ); }
      void getVar(long long          *data) const { ncwrap( nc_get_var_longlong ( ncid , id , data ) , __LINE__ ); }
      void getVar(signed char        *data) const { ncwrap( nc_get_var_schar    ( ncid , id , data ) , __LINE__ ); }
      void getVar(short              *data) const { ncwrap( nc_get_var_short    ( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned char      *data) const { ncwrap( nc_get_var_uchar    ( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned int       *data) const { ncwrap( nc_get_var_uint     ( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned long      *data) const { ncwrap( nc_get_var_uint     ( ncid , id , (unsigned int *) data ) , __LINE__ ); }
      void getVar(unsigned long long *data) const { ncwrap( nc_get_var_ulonglong( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned short     *data) const { ncwrap( nc_get_var_ushort   ( ncid , id , data ) , __LINE__ ); }
      void getVar(char               *data) const { ncwrap( nc_get_var_text     ( ncid , id , data ) , __LINE__ ); }
      void getVar(bool               *data) const { yakl_throw("ERROR: Cannot read bools directly from netCDF file. This should've been intercepted and changed to int."); }

      void print() {
        std::cout << "Variable Name: " << name << "\n";
        std::cout << "Dims: \n";
        for (int i=0; i < dims.size(); i++) {
          std::cout << "  " << dims[i].getName() << ";  Size: " << dims[i].getSize() << "\n\n";
        }
      }
    };


    class NcFile {
    public:
      int ncid;
      NcFile() { ncid = -999; }
      ~NcFile() {}
      NcFile(int ncid) { this->ncid = ncid; }
      NcFile(NcFile &&in) {
        this->ncid = in.ncid;
      }
      NcFile(NcFile const &in) {
        this->ncid = in.ncid;
      }
      NcFile &operator=(NcFile &&in) {
        this->ncid = in.ncid;
        return *this;
      }
      NcFile &operator=(NcFile const &in) {
        this->ncid = in.ncid;
        return *this;
      }

      bool isNull() { return ncid == -999; }

      void open( std::string fname , int mode ) {
        close();
        if (! (mode == NETCDF_MODE_READ || mode == NETCDF_MODE_WRITE) ) {
          yakl_throw("ERROR: open mode can be NETCDF_MODE_READ or NETCDF_MODE_WRITE");
        }
        ncwrap( nc_open( fname.c_str() , mode , &ncid ) , __LINE__ );
      }

      void create( std::string fname , int mode ) {
        close();
        if (! (mode == NETCDF_MODE_NEW || mode == NETCDF_MODE_REPLACE) ) {
          yakl_throw("ERROR: open mode can be NETCDF_MODE_NEW or NETCDF_MODE_REPLACE");
        }
        ncwrap( nc_create( fname.c_str() , mode | NC_NETCDF4 , &ncid ) , __LINE__ );
      }

      void close() {
        if (ncid != -999) ncwrap( nc_close( ncid ) , __LINE__ );
        ncid = -999;
      }

      NcVar getVar( std::string varName ) const {
        int varid;
        int ierr = nc_inq_varid( ncid , varName.c_str() , &varid);
        if (ierr != NC_NOERR) return NcVar();
        char vname[NC_MAX_NAME+1];
        int  type;
        int  ndims;
        int  dimids[NC_MAX_VAR_DIMS];
        int  natts;
        // Get variable information
        ncwrap( nc_inq_var(ncid , varid , vname , &type , &ndims , dimids , &natts ) , __LINE__ );
        // Accumulate the dimensions
        std::vector<NcDim> dims(ndims);
        for (int i=0; i < ndims; i++) {
          dims[i] = getDim( dimids[i] );
        }
        return NcVar( ncid , varName , dims , varid , type );
      }

      NcDim getDim( std::string dimName ) const {
        int dimid;
        int ierr = nc_inq_dimid( ncid , dimName.c_str() , &dimid);
        if (ierr != NC_NOERR) return NcDim();
        return getDim( dimid );
      }

      NcDim getDim( int dimid ) const {
        char   dname[NC_MAX_NAME+1];
        size_t len;
        int    unlim_dimid;
        ncwrap( nc_inq_dim( ncid , dimid , dname , &len ) , __LINE__ );
        ncwrap( nc_inq_unlimdim( ncid , &unlim_dimid ) , __LINE__ );
        return NcDim( std::string(dname) , len , dimid , dimid == unlim_dimid );
      }

      NcVar addVar( std::string varName , int type , std::vector<NcDim> &dims ) {
        std::vector<int> dimids(dims.size());
        for (int i=0; i < dims.size(); i++) { dimids[i] = dims[i].getId(); }
        int varid;
        ncwrap( nc_def_var(ncid , varName.c_str() , type , dims.size() , dimids.data() , &varid) , __LINE__ );
        return NcVar( ncid , varName , dims , varid , type );
      }

      NcVar addVar( std::string varName , int type ) {
        int varid;
        int *dummy = nullptr;
        ncwrap( nc_def_var(ncid , varName.c_str() , type , 0 , dummy , &varid) , __LINE__ );
        return NcVar( ncid , varName , std::vector<NcDim>(0) , varid , type );
      }

      NcDim addDim( std::string dimName , size_t len ) {
        int dimid;
        ncwrap( nc_def_dim(ncid , dimName.c_str() , len , &dimid ) , __LINE__ );
        return NcDim( dimName , len , dimid , false );
      }

      NcDim addDim( std::string dimName ) {
        int dimid;
        ncwrap( nc_def_dim(ncid , dimName.c_str() , NC_UNLIMITED , &dimid ) , __LINE__ );
        return NcDim( dimName , 0 , dimid , true );
      }

    };


    NcFile file;


    SimpleNetCDF() { }


    ~SimpleNetCDF() { close(); }


    void open(std::string fname , int mode = NETCDF_MODE_READ) { file.open(fname,mode); }


    void create(std::string fname , int mode = NC_CLOBBER) { file.create(fname,mode); }


    void close() { file.close(); }


    bool varExists( std::string varName ) const { return ! file.getVar(varName).isNull(); }


    bool dimExists( std::string dimName ) const { return ! file.getDim(dimName).isNull(); }


    size_t getDimSize( std::string dimName ) const { return file.getDim(dimName).getSize(); }


    void createDim( std::string dimName , size_t len ) { file.addDim( dimName , len ); }


    /***************************************************************************************************
    Write an entire Array at once
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void write(Array<T,rank,myMem,myStyle> const &arr , std::string varName , std::vector<std::string> dimNames) {
      if (rank != dimNames.size()) { yakl_throw("dimNames.size() != Array's rank"); }
      std::vector<NcDim> dims(rank); // List of dimensions for this variable
      // Make sure the dimensions are in there and are the right sizes
      for (int i=0; i<rank; i++) {
        auto dimLoc = file.getDim( dimNames[i] );
        // If dimension doesn't exist, create it; otherwise, make sure it's the right size
        NcDim tmp;
        if ( dimLoc.isNull() ) {
          tmp = file.addDim( dimNames[i] , arr.dimension[i] );
        } else {
          if (dimLoc.getSize() != arr.dimension[i]) {
            yakl_throw("dimension size differs from the file");
          }
          tmp = dimLoc;
        }
        if (myStyle == styleC) {
          dims[i] = tmp;
        } else {
          dims[rank-1-i] = tmp;
        }
      }
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        var = file.addVar( varName , getType<T>() , dims );
      } else {
        if ( var.getType() != getType<T>() ) { yakl_throw("Existing variable's type != array's type"); }
        auto varDims = var.getDims();
        if (varDims.size() != rank) { yakl_throw("Existing variable's rank != array's rank"); }
        for (int i=0; i < varDims.size(); i++) {
          if (myStyle == styleC) {
            if (varDims[i].getSize() != arr.dimension[i]) {
              yakl_throw("Existing variable's dimension sizes are not the same as the array's");
            }
          } else {
            if (varDims[rank-1-i].getSize() != arr.dimension[i]) {
              yakl_throw("Existing variable's dimension sizes are not the same as the array's");
            }
          }
        }
      }

      if (myMem == memDevice) {
        var.putVar(arr.createHostCopy().data());
      } else {
        var.putVar(arr.data());
      }
    }


    /***************************************************************************************************
    Write one entry of a scalar into the unlimited index
    ***************************************************************************************************/
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,int>::type = 0 >
    void write1(T val , std::string varName , int ind , std::string ulDimName="unlim" ) {
      // Get the unlimited dimension or create it if it doesn't exist
      auto ulDim = file.getDim( ulDimName );
      if ( ulDim.isNull() ) {
        ulDim = file.addDim( ulDimName );
      }
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        std::vector<NcDim> dims(1);
        dims[0] = ulDim;
        var = file.addVar( varName , getType<T>() , dims );
      }
      std::vector<size_t> start(1);
      std::vector<size_t> count(1);
      start[0] = ind;
      count[0] = 1;
      var.putVar(start,count,&val);
    }


    /***************************************************************************************************
    Write one entry of an Array into the unlimited index
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void write1(Array<T,rank,myMem,myStyle> const &arr , std::string varName , std::vector<std::string> dimNames ,
                int ind , std::string ulDimName="unlim" ) {
      if (rank != dimNames.size()) { yakl_throw("dimNames.size() != Array's rank"); }
      std::vector<NcDim> dims(rank+1); // List of dimensions for this variable
      // Get the unlimited dimension or create it if it doesn't exist
      dims[0] = file.getDim( ulDimName );
      if ( dims[0].isNull() ) {
        dims[0] = file.addDim( ulDimName );
      }
      // Make sure the dimensions are in there and are the right sizes
      for (int i=0; i<rank; i++) {
        auto dimLoc = file.getDim( dimNames[i] );
        // If dimension doesn't exist, create it; otherwise, make sure it's the right size
        NcDim tmp;
        if ( dimLoc.isNull() ) {
          tmp = file.addDim( dimNames[i] , arr.dimension[i] );
        } else {
          if (dimLoc.getSize() != arr.dimension[i]) {
            yakl_throw("dimension size differs from the file");
          }
          tmp = dimLoc;
        }
        if (myStyle == styleC) {
          dims[1+i] = tmp;
        } else {
          dims[1+rank-1-i] = tmp;
        }
      }
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        var = file.addVar( varName , getType<T>() , dims );
      } else {
        if ( var.getType() != getType<T>() ) { yakl_throw("Existing variable's type != array's type"); }
        auto varDims = var.getDims();
        if (varDims.size() != rank+1) {
          yakl_throw("Existing variable's rank != array's rank");
        }
        for (int i=1; i < varDims.size(); i++) {
          if (myStyle == styleC) {
            if (varDims[i].getSize() != arr.dimension[i-1]) {
              yakl_throw("Existing variable's dimension sizes are not the same as the array's");
            }
          } else {
            if (varDims[1+rank-i].getSize() != arr.dimension[i-1]) {
              yakl_throw("Existing variable's dimension sizes are not the same as the array's");
            }
          }
        }
      }

      std::vector<size_t> start(rank+1);
      std::vector<size_t> count(rank+1);
      start[0] = ind;
      count[0] = 1;
      for (int i=1; i < rank+1; i++) {
        start[i] = 0;
        count[i] = dims[i].getSize();
      }
      if (myMem == memDevice) {
        var.putVar(start,count,arr.createHostCopy().data());
      } else {
        var.putVar(start,count,arr.data());
      }
    }


    /***************************************************************************************************
    Read an entire Array
    ***************************************************************************************************/
    template <class T, int rank, int myMem, int myStyle>
    void read(Array<T,rank,myMem,myStyle> &arr , std::string varName) {
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      std::vector<int> dimSizes(rank);
      if ( ! var.isNull() ) {
        auto varDims = var.getDims();
        if (varDims.size() != rank) { yakl_throw("Existing variable's rank != array's rank"); }
        if (myStyle == styleC) {
          for (int i=0; i < varDims.size(); i++) { dimSizes[i] = varDims[i].getSize(); }
        } else if (myStyle == styleFortran) {
          for (int i=0; i < varDims.size(); i++) { dimSizes[i] = varDims[varDims.size()-1-i].getSize(); }
        }
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
      } else { yakl_throw("Variable does not exist"); }

      if (myMem == memDevice) {
        auto arrHost = arr.createHostCopy();
        if (std::is_same<T,bool>::value) {
          Array<int,rank,memHost,myStyle> tmp("tmp",dimSizes);
          var.getVar(tmp.data());
          for (int i=0; i < arr.totElems(); i++) { arrHost.myData[i] = tmp.myData[i] == 1; }
        } else {
          var.getVar(arrHost.data());
        }
        arrHost.deep_copy_to(arr);
        fence();
      } else {
        if (std::is_same<T,bool>::value) {
          Array<int,rank,memHost,myStyle> tmp("tmp",dimSizes);
          var.getVar(tmp.data());
          for (int i=0; i < arr.totElems(); i++) { arr.myData[i] = tmp.myData[i] == 1; }
        } else {
          var.getVar(arr.data());
        }
      }
    }


    /***************************************************************************************************
    Read a single scalar value
    ***************************************************************************************************/
    template <class T>
    void read(T &arr , std::string varName) {
      auto var = file.getVar(varName);
      if ( var.isNull() ) { yakl_throw("Variable does not exist"); }
      var.getVar(&arr);
    }


    /***************************************************************************************************
    Write a single scalar value
    ***************************************************************************************************/
    template <class T>
    void write(T arr , std::string varName) {
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        var = file.addVar( varName , getType<T>() );
      }
      var.putVar(&arr);
    }


    /***************************************************************************************************
    Determine the type of a template T
    ***************************************************************************************************/
    template <class T> int getType() const {
           if ( std::is_same<T,signed        char>::value ) { return NC_BYTE;   }
      else if ( std::is_same<T,unsigned      char>::value ) { return NC_UBYTE;  }
      else if ( std::is_same<T,             short>::value ) { return NC_SHORT;  }
      else if ( std::is_same<T,unsigned     short>::value ) { return NC_USHORT; }
      else if ( std::is_same<T,               int>::value ) { return NC_INT;    }
      else if ( std::is_same<T,unsigned       int>::value ) { return NC_UINT;   }
      else if ( std::is_same<T,              long>::value ) { return NC_INT;    }
      else if ( std::is_same<T,unsigned      long>::value ) { return NC_UINT;   }
      else if ( std::is_same<T,         long long>::value ) { return NC_INT64;  }
      else if ( std::is_same<T,unsigned long long>::value ) { return NC_UINT64; }
      else if ( std::is_same<T,             float>::value ) { return NC_FLOAT;  }
      else if ( std::is_same<T,            double>::value ) { return NC_DOUBLE; }
           if ( std::is_same<T,              char>::value ) { return NC_CHAR;   }
      else { yakl_throw("Invalid type"); }
      return -1;
    }

  };



}


