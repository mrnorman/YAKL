
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
      Kokkos::abort(nc_strerror(ierr));
    }
  }

  int constexpr NETCDF_MODE_READ    = NC_NOWRITE;
  int constexpr NETCDF_MODE_WRITE   = NC_WRITE;
  int constexpr NETCDF_MODE_REPLACE = NC_CLOBBER;
  int constexpr NETCDF_MODE_NEW     = NC_NOCLOBBER;

  // Evidently there were ton of issues when using the C++ interface for NetCDF
  // People can't seem to install it correctly.
  // Therefore, I'm replicating the basic functionality so that I can use the code
  // I previously wrote for handling netCDF files for YAKL Array objects
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
        if (isNull() || i < 0 || static_cast<size_t>(i) >= dims.size()) {
          return NcDim();
        } else {
          return dims[i];
        }
      }

      template <class T>
      void checkData(T const * data) const {
        if (isNull()) Kokkos::abort("ERROR: I/O attempted with a null netCDF variable");
        bool empty = false;
        for (auto const & dim : dims) empty = empty || dim.getSize() == 0;
        if (data == nullptr && !empty) Kokkos::abort("ERROR: netCDF I/O received a null data pointer");
      }

      void checkHyperslab(std::vector<size_t> const & start, std::vector<size_t> const & count) const {
        if (start.size() != dims.size() || count.size() != dims.size()) {
          Kokkos::abort("ERROR: netCDF hyperslab rank differs from variable rank");
        }
        for (size_t i=0; i < dims.size(); i++) {
          if (count[i] > std::numeric_limits<size_t>::max()-start[i]) {
            Kokkos::abort("ERROR: netCDF hyperslab bound overflow");
          }
          if (!dims[i].isUnlimited() && start[i]+count[i] > dims[i].getSize()) {
            Kokkos::abort("ERROR: netCDF hyperslab exceeds a fixed dimension");
          }
        }
      }

      size_t elementCount() const {
        size_t count = 1;
        for (auto const & dim : dims) {
          if (dim.getSize() != 0 && count > std::numeric_limits<size_t>::max()/dim.getSize()) {
            Kokkos::abort("ERROR: netCDF variable element count overflows size_t");
          }
          count *= dim.getSize();
        }
        return count;
      }

      size_t elementCount(std::vector<size_t> const & extents) const {
        size_t count = 1;
        for (auto const extent : extents) {
          if (extent != 0 && count > std::numeric_limits<size_t>::max()/extent) {
            Kokkos::abort("ERROR: netCDF hyperslab element count overflows size_t");
          }
          count *= extent;
        }
        return count;
      }

      void putVar(double             const *data) { checkData(data); ncwrap( nc_put_var_double   ( ncid , id , data ) , __LINE__ ); }
      void putVar(float              const *data) { checkData(data); ncwrap( nc_put_var_float    ( ncid , id , data ) , __LINE__ ); }
      void putVar(int                const *data) { checkData(data); ncwrap( nc_put_var_int      ( ncid , id , data ) , __LINE__ ); }
      void putVar(long               const *data) { checkData(data); ncwrap( nc_put_var_long     ( ncid , id , data ) , __LINE__ ); }
      void putVar(long long          const *data) { checkData(data); ncwrap( nc_put_var_longlong ( ncid , id , data ) , __LINE__ ); }
      void putVar(signed char        const *data) { checkData(data); ncwrap( nc_put_var_schar    ( ncid , id , data ) , __LINE__ ); }
      void putVar(short              const *data) { checkData(data); ncwrap( nc_put_var_short    ( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned char      const *data) { checkData(data); ncwrap( nc_put_var_uchar    ( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned int       const *data) { checkData(data); ncwrap( nc_put_var_uint     ( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned long const *data) {
        checkData(data);
        size_t const count = elementCount();
        std::vector<unsigned long long> converted(count);
        for (size_t i=0; i < count; i++) converted[i] = static_cast<unsigned long long>(data[i]);
        ncwrap( nc_put_var_ulonglong( ncid , id , converted.data() ) , __LINE__ );
      }
      void putVar(unsigned long long const *data) { checkData(data); ncwrap( nc_put_var_ulonglong( ncid , id , data ) , __LINE__ ); }
      void putVar(unsigned short     const *data) { checkData(data); ncwrap( nc_put_var_ushort   ( ncid , id , data ) , __LINE__ ); }
      void putVar(char               const *data) { checkData(data); ncwrap( nc_put_var_text     ( ncid , id , data ) , __LINE__ ); }
      void putVar(bool               const *data) { Kokkos::abort("ERROR: Cannot write bools to netCDF file"); }

      void putVar(std::vector<size_t> start , std::vector<size_t> count, double             const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_double   ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, float              const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_float    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, int                const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_int      ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, long               const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_long     ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, long long          const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_longlong ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, signed char        const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_schar    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, short              const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_short    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned char      const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_uchar    ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned int       const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_uint     ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start, std::vector<size_t> count, unsigned long const *data) {
        checkData(data);
        checkHyperslab(start,count);
        size_t const numElements = elementCount(count);
        std::vector<unsigned long long> converted(numElements);
        for (size_t i=0; i < numElements; i++) converted[i] = static_cast<unsigned long long>(data[i]);
        ncwrap( nc_put_vara_ulonglong( ncid , id , start.data() , count.data() , converted.data() ) , __LINE__ );
      }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned long long const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_ulonglong( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, unsigned short     const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_ushort   ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, char               const *data) { checkData(data); checkHyperslab(start,count); ncwrap( nc_put_vara_text     ( ncid , id , start.data() , count.data(), data ) , __LINE__ ); }
      void putVar(std::vector<size_t> start , std::vector<size_t> count, bool               const *data) { Kokkos::abort("ERROR: Cannot write bools to netCDF file"); }

      void getVar(double             *data) const { checkData(data); ncwrap( nc_get_var_double   ( ncid , id , data ) , __LINE__ ); }
      void getVar(float              *data) const { checkData(data); ncwrap( nc_get_var_float    ( ncid , id , data ) , __LINE__ ); }
      void getVar(int                *data) const { checkData(data); ncwrap( nc_get_var_int      ( ncid , id , data ) , __LINE__ ); }
      void getVar(long               *data) const { checkData(data); ncwrap( nc_get_var_long     ( ncid , id , data ) , __LINE__ ); }
      void getVar(long long          *data) const { checkData(data); ncwrap( nc_get_var_longlong ( ncid , id , data ) , __LINE__ ); }
      void getVar(signed char        *data) const { checkData(data); ncwrap( nc_get_var_schar    ( ncid , id , data ) , __LINE__ ); }
      void getVar(short              *data) const { checkData(data); ncwrap( nc_get_var_short    ( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned char      *data) const { checkData(data); ncwrap( nc_get_var_uchar    ( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned int       *data) const { checkData(data); ncwrap( nc_get_var_uint     ( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned long *data) const {
        checkData(data);
        size_t const count = elementCount();
        std::vector<unsigned long long> converted(count);
        ncwrap( nc_get_var_ulonglong( ncid , id , converted.data() ) , __LINE__ );
        for (size_t i=0; i < count; i++) {
          if (converted[i] > std::numeric_limits<unsigned long>::max()) {
            Kokkos::abort("ERROR: netCDF value does not fit in unsigned long");
          }
          data[i] = static_cast<unsigned long>(converted[i]);
        }
      }
      void getVar(unsigned long long *data) const { checkData(data); ncwrap( nc_get_var_ulonglong( ncid , id , data ) , __LINE__ ); }
      void getVar(unsigned short     *data) const { checkData(data); ncwrap( nc_get_var_ushort   ( ncid , id , data ) , __LINE__ ); }
      void getVar(char               *data) const { checkData(data); ncwrap( nc_get_var_text     ( ncid , id , data ) , __LINE__ ); }
      void getVar(bool               *data) const { Kokkos::abort("ERROR: Cannot read bools directly from netCDF file. This should've been intercepted and changed to int."); }

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
      NcFile(NcFile const &) = delete;
      NcFile &operator=(NcFile const &) = delete;
      NcFile(NcFile &&in) noexcept {
        this->ncid = in.ncid;
        in.ncid = -999;
      }
      NcFile &operator=(NcFile &&in) noexcept {
        if (this != &in) {
          close();
          this->ncid = in.ncid;
          in.ncid = -999;
        }
        return *this;
      }

      bool isNull() { return ncid == -999; }

      void open( std::string fname , int mode ) {
        if (fname.empty()) Kokkos::abort("ERROR: cannot open a netCDF file with an empty name");
        close();
        if (! (mode == NETCDF_MODE_READ || mode == NETCDF_MODE_WRITE) ) {
          Kokkos::abort("ERROR: open mode can be NETCDF_MODE_READ or NETCDF_MODE_WRITE");
        }
        ncwrap( nc_open( fname.c_str() , mode , &ncid ) , __LINE__ );
      }

      void create( std::string fname , int mode ) {
        if (fname.empty()) Kokkos::abort("ERROR: cannot create a netCDF file with an empty name");
        close();
        if (! (mode == NETCDF_MODE_NEW || mode == NETCDF_MODE_REPLACE) ) {
          Kokkos::abort("ERROR: open mode can be NETCDF_MODE_NEW or NETCDF_MODE_REPLACE");
        }
        ncwrap( nc_create( fname.c_str() , mode | NC_NETCDF4 , &ncid ) , __LINE__ );
      }

      void close() {
        if (ncid != -999) ncwrap( nc_close( ncid ) , __LINE__ );
        ncid = -999;
      }

      NcVar getVar( std::string varName ) const {
        if (ncid == -999) Kokkos::abort("ERROR: querying a variable in an unopened netCDF file");
        if (varName.empty()) Kokkos::abort("ERROR: netCDF variable name cannot be empty");
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
        if (ncid == -999) Kokkos::abort("ERROR: querying a dimension in an unopened netCDF file");
        if (dimName.empty()) Kokkos::abort("ERROR: netCDF dimension name cannot be empty");
        int dimid;
        int ierr = nc_inq_dimid( ncid , dimName.c_str() , &dimid);
        if (ierr != NC_NOERR) return NcDim();
        return getDim( dimid );
      }

      NcDim getDim( int dimid ) const {
        if (ncid == -999) Kokkos::abort("ERROR: querying a dimension in an unopened netCDF file");
        if (dimid < 0) Kokkos::abort("ERROR: netCDF dimension ID cannot be negative");
        char   dname[NC_MAX_NAME+1];
        size_t len;
        int    unlim_dimid;
        ncwrap( nc_inq_dim( ncid , dimid , dname , &len ) , __LINE__ );
        ncwrap( nc_inq_unlimdim( ncid , &unlim_dimid ) , __LINE__ );
        return NcDim( std::string(dname) , len , dimid , dimid == unlim_dimid );
      }

      NcVar addVar( std::string varName , int type , std::vector<NcDim> &dims ) {
        if (ncid == -999) Kokkos::abort("ERROR: defining a variable in an unopened netCDF file");
        if (varName.empty()) Kokkos::abort("ERROR: netCDF variable name cannot be empty");
        for (auto const & dim : dims) if (dim.isNull()) Kokkos::abort("ERROR: netCDF variable has a null dimension");
        std::vector<int> dimids(dims.size());
        for (int i=0; i < dims.size(); i++) { dimids[i] = dims[i].getId(); }
        int varid;
        ncwrap( nc_def_var(ncid , varName.c_str() , type , dims.size() , dimids.data() , &varid) , __LINE__ );
        return NcVar( ncid , varName , dims , varid , type );
      }

      NcVar addVar( std::string varName , int type ) {
        if (ncid == -999) Kokkos::abort("ERROR: defining a variable in an unopened netCDF file");
        if (varName.empty()) Kokkos::abort("ERROR: netCDF variable name cannot be empty");
        int varid;
        int *dummy = nullptr;
        ncwrap( nc_def_var(ncid , varName.c_str() , type , 0 , dummy , &varid) , __LINE__ );
        return NcVar( ncid , varName , std::vector<NcDim>(0) , varid , type );
      }

      NcDim addDim( std::string dimName , size_t len ) {
        if (ncid == -999) Kokkos::abort("ERROR: defining a dimension in an unopened netCDF file");
        if (dimName.empty()) Kokkos::abort("ERROR: netCDF dimension name cannot be empty");
        int dimid;
        ncwrap( nc_def_dim(ncid , dimName.c_str() , len , &dimid ) , __LINE__ );
        return NcDim( dimName , len , dimid , false );
      }

      NcDim addDim( std::string dimName ) {
        if (ncid == -999) Kokkos::abort("ERROR: defining a dimension in an unopened netCDF file");
        if (dimName.empty()) Kokkos::abort("ERROR: netCDF dimension name cannot be empty");
        int dimid;
        ncwrap( nc_def_dim(ncid , dimName.c_str() , NC_UNLIMITED , &dimid ) , __LINE__ );
        return NcDim( dimName , 0 , dimid , true );
      }

    };


    NcFile file;


    SimpleNetCDF() { }

    SimpleNetCDF(SimpleNetCDF const &) = delete;
    SimpleNetCDF &operator=(SimpleNetCDF const &) = delete;
    SimpleNetCDF(SimpleNetCDF &&) noexcept = default;
    SimpleNetCDF &operator=(SimpleNetCDF &&) noexcept = default;


    ~SimpleNetCDF() { close(); }


    void open(std::string fname , int mode = NETCDF_MODE_READ) { file.open(fname,mode); }


    void create(std::string fname , int mode = NC_CLOBBER) { file.create(fname,mode); }


    void close() { file.close(); }


    bool varExists( std::string varName ) const { return ! file.getVar(varName).isNull(); }


    bool dimExists( std::string dimName ) const { return ! file.getDim(dimName).isNull(); }


    size_t getDimSize( std::string dimName ) const {
      auto const dim = file.getDim(dimName);
      if (dim.isNull()) Kokkos::abort("ERROR: requested netCDF dimension does not exist");
      return dim.getSize();
    }


    void createDim( std::string dimName , size_t len ) { file.addDim( dimName , len ); }

    void createDim( std::string dimName ) { file.addDim( dimName ); }


    template <class ViewType> requires is_Array<ViewType>
    void write(ViewType const & arr , std::string varName , std::vector<std::string> dimNames) {
      int constexpr rank = ViewType::rank();
      using T = typename ViewType::non_const_value_type;
      if (!arr.is_allocated()) Kokkos::abort("ERROR: writing an unallocated Array to netCDF");
      if (varName.empty()) Kokkos::abort("ERROR: netCDF variable name cannot be empty");
      if (rank != dimNames.size()) { Kokkos::abort("dimNames.size() != Array's rank"); }
      for (auto const & name : dimNames) if (name.empty()) Kokkos::abort("ERROR: netCDF dimension name cannot be empty");
      std::vector<NcDim> dims(rank); // List of dimensions for this variable
      // Make sure the dimensions are in there and are the right sizes
      for (int i=0; i<rank; i++) {
        auto dimLoc = file.getDim( dimNames[i] );
        // If dimension doesn't exist, create it; otherwise, make sure it's the right size
        NcDim tmp;
        if ( dimLoc.isNull() ) {
          tmp = file.addDim( dimNames[i] , arr.extent(i) );
        } else {
          if (dimLoc.getSize() != arr.extent(i)) {
            Kokkos::abort("dimension size differs from the file");
          }
          tmp = dimLoc;
        }
        if (ViewType::is_cstyle) { dims[i] = tmp; }
        else                     { dims[rank-1-i] = tmp; }
      }
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        var = file.addVar( varName , getType<T>() , dims );
      } else {
        if ( var.getType() != getType<T>() ) { Kokkos::abort("Existing variable's type != array's type"); }
        auto varDims = var.getDims();
        if (varDims.size() != rank) { Kokkos::abort("Existing variable's rank != array's rank"); }
        for (int i=0; i < varDims.size(); i++) {
          if (ViewType::is_cstyle) {
            if (varDims[i].getSize() != arr.extent(i)) {
              Kokkos::abort("Existing variable's dimension sizes are not the same as the array's");
            }
          } else {
            if (varDims[rank-1-i].getSize() != arr.extent(i)) {
              Kokkos::abort("Existing variable's dimension sizes are not the same as the array's");
            }
          }
        }
      }

      if (ViewType::on_device) { var.putVar(arr.createHostCopy().data()); }
      else                     { var.putVar(arr.data()); }
    }


    template <class T> requires std::is_arithmetic_v<T>
    void write1(T val , std::string varName , int ind , std::string ulDimName="unlim" ) {
      if (varName.empty() || ulDimName.empty()) Kokkos::abort("ERROR: netCDF names cannot be empty");
      if (ind < 0) Kokkos::abort("ERROR: netCDF record index cannot be negative");
      // Get the unlimited dimension or create it if it doesn't exist
      auto ulDim = file.getDim( ulDimName );
      if ( ulDim.isNull() )  ulDim = file.addDim( ulDimName );
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        std::vector<NcDim> dims(1);
        dims[0] = ulDim;
        var = file.addVar( varName , getType<T>() , dims );
      } else {
        if (var.getType() != getType<T>() || var.getDimCount() != 1 || !var.getDim(0).isUnlimited()) {
          Kokkos::abort("ERROR: existing netCDF record variable has incompatible type or dimensions");
        }
      }
      std::vector<size_t> start(1);
      std::vector<size_t> count(1);
      start[0] = ind;
      count[0] = 1;
      var.checkHyperslab(start,count);
      var.putVar(start,count,&val);
    }


    template <class ViewType> requires is_Array<ViewType>
    void write1(ViewType const & arr , std::string varName , std::vector<std::string> dimNames ,
                int ind , std::string ulDimName="unlim" ) {
      int constexpr rank = ViewType::rank();
      using T = typename ViewType::non_const_value_type;
      if (!arr.is_allocated()) Kokkos::abort("ERROR: writing an unallocated Array to netCDF");
      if (varName.empty() || ulDimName.empty()) Kokkos::abort("ERROR: netCDF names cannot be empty");
      if (ind < 0) Kokkos::abort("ERROR: netCDF record index cannot be negative");
      if (rank != dimNames.size()) { Kokkos::abort("dimNames.size() != Array's rank"); }
      for (auto const & name : dimNames) if (name.empty()) Kokkos::abort("ERROR: netCDF dimension name cannot be empty");
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
          tmp = file.addDim( dimNames[i] , arr.extent(i) );
        } else {
          if (dimLoc.getSize() != arr.extent(i)) {
            Kokkos::abort("dimension size differs from the file");
          }
          tmp = dimLoc;
        }
        if (ViewType::is_cstyle) {
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
        if ( var.getType() != getType<T>() ) { Kokkos::abort("Existing variable's type != array's type"); }
        auto varDims = var.getDims();
        if (varDims.size() != rank+1) {
          Kokkos::abort("Existing variable's rank != array's rank");
        }
        if (!varDims[0].isUnlimited()) Kokkos::abort("ERROR: first netCDF record dimension is not unlimited");
        for (int i=1; i < varDims.size(); i++) {
          if (ViewType::is_cstyle) {
            if (varDims[i].getSize() != arr.extent(i-1)) {
              Kokkos::abort("Existing variable's dimension sizes are not the same as the array's");
            }
          } else {
            if (varDims[1+rank-i].getSize() != arr.extent(i-1)) {
              Kokkos::abort("Existing variable's dimension sizes are not the same as the array's");
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
      var.checkHyperslab(start,count);
      if (ViewType::on_device) { var.putVar(start,count,arr.createHostCopy().data()); }
      else                     { var.putVar(start,count,arr.data()); }
    }


    template <class ViewType> requires is_Array<ViewType>
    void read(ViewType const & arr , std::string varName) {
      int constexpr rank = ViewType::rank();
      using T = typename ViewType::non_const_value_type;
      if (!arr.is_allocated()) Kokkos::abort("ERROR: reading netCDF into an unallocated Array");
      if (varName.empty()) Kokkos::abort("ERROR: netCDF variable name cannot be empty");
      // Make sure the variable is there and is the right dimension
      auto var = file.getVar(varName);
      std::vector<int> dimSizes(rank);
      if ( ! var.isNull() ) {
        int expectedType;
        if constexpr (std::is_same_v<T,bool>) expectedType = NC_INT;
        else                                  expectedType = getType<T>();
        if (var.getType() != expectedType) Kokkos::abort("ERROR: netCDF variable and Array types differ");
        auto varDims = var.getDims();
        if (varDims.size() != rank) { Kokkos::abort("Existing variable's rank != array's rank"); }
        if (ViewType::is_cstyle) { for (int i=0; i < varDims.size(); i++) { dimSizes[i] = varDims[i].getSize(); } }
        else                     { for (int i=0; i < varDims.size(); i++) { dimSizes[i] = varDims[varDims.size()-1-i].getSize(); } }
        for (int i=0; i < dimSizes.size(); i++) {
          if (dimSizes[i] != arr.extent(i)) Kokkos::abort("ERROR: Array & var dims mismatch");
        }
      } else { Kokkos::abort("Variable does not exist"); }

      if (ViewType::on_device) {
        auto arrHost = arr.createHostObject();
        if constexpr (std::is_same_v<T,bool>) {
          auto tmp = arr.template clone_object<Kokkos::HostSpace,int>();
          var.getVar(tmp.data());
          for (int i=0; i < arr.size(); i++) { arrHost.data()[i] = tmp.data()[i] == 1; }
        } else {
          var.getVar(arrHost.data());
        }
        arrHost.deep_copy_to(arr);
        Kokkos::fence();
      } else {
        if constexpr (std::is_same_v<T,bool>) {
          auto tmp = arr.template clone_object<Kokkos::HostSpace,int>();
          var.getVar(tmp.data());
          for (int i=0; i < arr.size(); i++) { arr.data()[i] = tmp.data()[i] == 1; }
        } else {
          var.getVar(arr.data());
        }
      }
    }


    template <class T> requires std::is_arithmetic_v<T>
    void read(T &arr , std::string varName) {
      if (varName.empty()) Kokkos::abort("ERROR: netCDF variable name cannot be empty");
      auto var = file.getVar(varName);
      if ( var.isNull() ) { Kokkos::abort("Variable does not exist"); }
      if (var.getType() != getType<T>() || var.getDimCount() != 0) {
        Kokkos::abort("ERROR: netCDF scalar variable has incompatible type or rank");
      }
      var.getVar(&arr);
    }


    template <class T> requires std::is_arithmetic_v<T>
    void write(T arr , std::string varName) {
      if (varName.empty()) Kokkos::abort("ERROR: netCDF variable name cannot be empty");
      auto var = file.getVar(varName);
      if ( var.isNull() ) {
        var = file.addVar( varName , getType<T>() );
      } else if (var.getType() != getType<T>() || var.getDimCount() != 0) {
        Kokkos::abort("ERROR: existing netCDF scalar variable has incompatible type or rank");
      }
      var.putVar(&arr);
    }


    template <class T> int getType() const {
           if ( std::is_same_v<typename std::remove_cv_t<T>,signed        char> ) { return NC_BYTE;   }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,unsigned      char> ) { return NC_UBYTE;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,             short> ) { return NC_SHORT;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,unsigned     short> ) { return NC_USHORT; }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,               int> ) { return NC_INT;    }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,unsigned       int> ) { return NC_UINT;   }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,              long> ) { return NC_INT;    }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,unsigned      long> ) {
        return sizeof(unsigned long) == sizeof(unsigned int) ? NC_UINT : NC_UINT64;
      }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,         long long> ) { return NC_INT64;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,unsigned long long> ) { return NC_UINT64; }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,             float> ) { return NC_FLOAT;  }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,            double> ) { return NC_DOUBLE; }
      else if ( std::is_same_v<typename std::remove_cv_t<T>,              char> ) { return NC_CHAR;   }
      else { Kokkos::abort("Invalid type"); }
      return -1;
    }

  };



}
