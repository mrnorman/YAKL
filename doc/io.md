# NetCDF and PNetCDF

[API home](README.md) · [Arrays](arrays.md) · [Memory](memory.md)

The file wrappers are optional extensions and are not included by `YAKL.h`. Their checks are unconditional because file I/O
already dominates their cost. Errors are reported through `Kokkos::abort`, not recoverable exceptions.

## `SimpleNetCDF`

Include the serial wrapper only in a target linked to the NetCDF C library:

```cpp
#include "extensions/YAKL_netcdf.h"

yakl::SimpleNetCDF nc;
nc.create("state.nc",yakl::NETCDF_MODE_REPLACE);
nc.write(state,"state",{"z","y","x"});
nc.write(time,"time");
nc.close();
```

Mode constants are `NETCDF_MODE_READ`, `NETCDF_MODE_WRITE`, `NETCDF_MODE_REPLACE`, and `NETCDF_MODE_NEW`. `open` accepts
read/write modes; `create` accepts replace/new modes and creates a NetCDF-4 file.

### `SimpleNetCDF` API

| Member | Semantics |
| --- | --- |
| `open(filename, mode=NETCDF_MODE_READ)` | Close any current file and open an existing file. |
| `create(filename, mode=NETCDF_MODE_REPLACE)` | Close any current file and create a NetCDF-4 file. |
| `close()` | Close an open file; safe on an unopened wrapper. |
| `varExists(name)`, `dimExists(name)` | Query existence. |
| `getDimSize(name)` | Return an existing dimension length. |
| `createDim(name,len)` | Create a fixed dimension. |
| `createDim(name)` | Create an unlimited dimension. |
| `write(array,name,dimNames)` | Create/validate dimensions and the variable, then write the complete array. |
| `read(array,name)` | Read a complete existing variable into an already allocated, exactly matching array. |
| `write(scalar,name)`, `read(scalar,name)` | Write/create or read a rank-zero variable of matching type. |
| `write1(value,name,index,unlimitedName)` | Write one scalar record in an unlimited dimension. |
| `write1(array,name,dimNames,index,unlimitedName)` | Write one array record, with the unlimited dimension first in the file. |

Array writes create missing fixed dimensions and variables, but reject incompatible existing types, ranks, or extents.
Device arrays are staged through host memory. Reads into device arrays stage on the host, deep-copy to the device, and fence
before returning. C-style file dimension order matches array dimension order. Fortran-style arrays reverse dimension order
in the file so their contiguous memory is represented without transposing values; user `dimNames` are specified in the
array's logical order.

Mappings are `signed char`/`unsigned char` to `NC_BYTE`/`NC_UBYTE`, short and int families to their corresponding NetCDF
types, `long` to `NC_INT`, long long to `NC_INT64`, and float/double/char to `NC_FLOAT`/`NC_DOUBLE`/`NC_CHAR`. On LP64
systems, `unsigned long` is represented with `NC_UINT64` and converted element-by-element without reinterpreting it as
`unsigned int`; when it has unsigned-int width it maps to `NC_UINT`. Direct Boolean writes are unsupported by the serial
wrapper; Boolean array reads accept an `NC_INT` variable and convert values equal to one to `true`.

`SimpleNetCDF` is movable but not copyable. It owns its raw file handle, and destruction closes an open file. Explicit
`close()` is preferable when the application needs the close error reported at a controlled point.

### Low-level serial objects

`SimpleNetCDF::NcDim`, `NcVar`, and `NcFile` are public building blocks held by `SimpleNetCDF::file`.

- `NcDim` carries name, length, ID, and unlimited/null state. Accessors are `getName`, `getSize`, `getId`, `isNull`, and
  `isUnlimited`.
- `NcVar` carries a file ID, name, dimensions, variable ID, and NetCDF type. It exposes metadata accessors, `getDim`,
  full-variable `putVar/getVar`, hyperslab `putVar`, validation helpers, element counts, and diagnostic `print`. Pointer
  overloads cover the supported scalar types; pointers must refer to host-accessible contiguous storage.
- `NcFile` owns no automatic close in its destructor but provides move-only handle state, `open`, `create`, `close`,
  `getVar`, `getDim`, `addVar`, and `addDim`. Prefer the outer wrapper for automatic handle closure.

Null low-level objects use an invalid sentinel ID and are returned for missing names. Hyperslab `start` and `count` vectors
must match variable rank, remain within fixed dimensions, and have nonoverflowing products.

## `SimplePNetCDF`

PNetCDF requires MPI and the PNetCDF library:

```cpp
#include "extensions/YAKL_pnetcdf.h"

yakl::SimplePNetCDF nc(MPI_COMM_WORLD);
nc.create("distributed.nc");                    // collective
nc.create_dim("global_x",global_n);             // collective
nc.create_var<double>("field",{"global_x"});   // collective
nc.enddef();                                     // collective
nc.write_all(local,"field",{global_offset});    // collective
nc.close();                                      // collective
```

The wrapper is a state machine with unopened, define, collective-data, and independent-data modes. State-changing methods
perform the necessary transitions where documented.

### Collective versus independent calls

All ranks in the wrapper's communicator must call:

- `open`, `create`, and `close`;
- `create_dim`, `create_var<T>`, `redef`, and `enddef`;
- `begin_indep_data` and `end_indep_data`; and
- `write_all` and `read_all`.

The metadata query methods and scalar/whole-variable `write` and `read` are documented for a single task and must be used
only in a valid data mode. Calling a collective method on only some communicator ranks can deadlock.

### `SimplePNetCDF` API

| Member | Semantics |
| --- | --- |
| constructor `(comm=MPI_COMM_WORLD)` | Start unopened; `MPI_COMM_NULL` is invalid. |
| `open(filename,mode=NC_WRITE,info=MPI_INFO_NULL)` | Collectively open, closing any prior file. |
| `create(filename,flag=NC_CLOBBER,info=MPI_INFO_NULL)` | Collectively create in define mode. |
| `close()` | End independent mode if needed and collectively close. |
| `get_dim_id`, `get_var_id`, `dim_exists`, `var_exists`, `get_dim_size` | Metadata lookup in an open file. |
| `create_dim(name,len)` | Collectively define a fixed or `NC_UNLIMITED` dimension. |
| `create_var<T>(name,dimensionNames)` | Collectively define a variable. An empty list creates a scalar. |
| `redef()`, `enddef()` | Collectively enter or leave define mode; idempotent in the target state. |
| `begin_indep_data()`, `end_indep_data()` | Collectively enter or leave independent data mode. |
| `write(value,name)`, `read(value,name)` | Single-task scalar I/O in data mode. |
| `write(array,name)`, `read(array,name)` | Single-task whole-variable I/O with exact rank/type/extents. |
| `write_all(array,name,start)`, `read_all(array,name,start)` | Collective hyperslab I/O; `start` has one nonnegative offset per array dimension. |

Array data is staged through host memory as needed. Collective hyperslab counts come from array extents. Fixed-dimension
hyperslabs must fit; writes may extend an unlimited dimension. The wrapper requires file dimension order to match the
array's `extent(i)` order; unlike `SimpleNetCDF`, it does not reverse `_F` dimensions automatically.

Supported PNetCDF types are `char`, signed/unsigned character, signed/unsigned short, signed/unsigned int, `long`,
`unsigned long`, signed/unsigned long long, `float`, `double`, and `bool`. `long`/`unsigned long` use 64-bit NetCDF types in
this interface. Boolean values are stored as `NC_UBYTE`. Conversion buffers are used where the native PNetCDF overload does
not match the C++ representation, including `unsigned long` and `bool`, and narrowing reads are range checked.

`SimplePNetCDF` is movable but not copyable. Its destructor is `noexcept`: it attempts to leave independent mode and close,
printing errors to standard error rather than throwing or aborting. Because close is collective, applications should always
call `close()` collectively before destruction; destructor cleanup is only a last resort.

## Build and test opt-in

The repository unit tests compile NetCDF and PNetCDF coverage only when `YAKL_TEST_NETCDF` and `YAKL_TEST_PNETCDF` are
enabled, respectively, with dependencies supplied by the machine configuration. These test switches do not automatically
link an application's targets; consumers remain responsible for finding and linking the libraries used by each extension.

[Previous: Memory](memory.md) · [API home](README.md) · [Compile-time configuration](configuration.md)
