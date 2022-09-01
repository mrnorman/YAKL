
#pragma once
// Included by YAKL.h

namespace yakl {

  /** @brief Class to handle scalars that exist before kernels, are written to by kernels, and read after the kernel terminates.
    * 
    * Handles the case where a scalar value is written to in a kernel and must be read on the host after the kernel
    * completes.
    * Passing a value to the constructor will initialize the device pointer with that data from the host.
    * Using the = operator will assign to the value inside a kernel. This is the most common operation.
    * To access the value as a reference that is writable, use the operator() (e.g., in an atomic operation perhaps).
    * To read on the host afterward, using hostRead()
    * To write after construction, use hostWrite()
    * 
    * ```
    * ScalarLiveOut<float> sum(0.);
    * // You'd use a reduction for this in reality. This is just for demonstration.
    * parallel_for( Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
    *   // sum() obtains a modifiable reference to the data contained by the ScalarLiveOut object
    *   atomicAdd( sum() , 1. );
    * });
    * if (sum.hostRead() != ny*nx) yakl::yakl_throw("ERROR: Wrong sum");
    * ```
    * 
    * ```
    * ScalarLiveOut<bool> data_is_bad(false);
    * parallel_for( Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
    *   // operator= always assigns on the device
    *   if (density(j,i) < 0) data_is_bad = true;
    * });
    * if (data_is_bad.hostRead()) yakl::yakl_throw("ERROR: Data is bad");
    * ```
    *
    * @param T The type of the scalar value
    */
  template <class T> class ScalarLiveOut {
  protected:
    /** @private */
    Array<T,1,memDevice,styleC> data;

  public:
    /** @brief Default constructor allocates room on the device for one scalar of type `T` */
    YAKL_INLINE ScalarLiveOut() {
      data = Array<T,1,memDevice,styleC>("ScalarLiveOut_data",1);  // Create array
    }
    /** @brief [ASYNCHRONOUS] This constructor allocates room on the device for one scalar of type `T` and initializes it on device with the provided value. */
    explicit ScalarLiveOut(T val, Stream stream = Stream() ) {
      data = Array<T,1,memDevice,styleC>("ScalarLiveOut_data",1);  // Create array
      hostWrite(val,stream);                                // Copy to device
    }
    /** @brief Deallocates the scalar value on the device. */
    YAKL_INLINE ~ScalarLiveOut() {
      data = Array<T,1,memDevice,styleC>();
    }

    /** @brief Copies and moves are shallow, not deep copy. */
    YAKL_INLINE ScalarLiveOut            ( ScalarLiveOut const &rhs) { this->data = rhs.data; }
    /** @brief Copies and moves are shallow, not deep copy. */
    YAKL_INLINE ScalarLiveOut & operator=( ScalarLiveOut const &rhs) { this->data = rhs.data; return *this; }
    /** @brief Copies and moves are shallow, not deep copy. */
    YAKL_INLINE ScalarLiveOut            ( ScalarLiveOut      &&rhs) { this->data = rhs.data; }
    /** @brief Copies and moves are shallow, not deep copy. */
    YAKL_INLINE ScalarLiveOut & operator=( ScalarLiveOut      &&rhs) { this->data = rhs.data; return *this; }

    /** @brief Assign a value to the ScalarLiveOut object on the device. */
    template <class TLOC , typename std::enable_if< std::is_arithmetic<TLOC>::value , int >::type = 0>
    YAKL_INLINE T &operator= (TLOC rhs) const { data(0) = rhs; return data(0); }

    /** @brief Returns a modifiable reference to the underlying data on the device. */
    YAKL_INLINE T &operator() () const {
      return data(0);
    }

    /** @brief Returns a modifiable reference to the underlying data on the device. */
    YAKL_INLINE T get() const {
      return data(0);
    }

    /** @brief Returns a host copy of the data. This is blocking. */
    inline T hostRead(Stream stream = Stream()) const {
      return data.createHostCopy(stream)(0);
    }

    /** @brief [ASYNCHRONOUS] Writes a value to the device-resident underlying data */
    inline void hostWrite(T val, Stream stream = Stream()) {
      // Copy data to device
      auto &myData = this->data;
      c::parallel_for( c::Bounds<1>(1) , YAKL_LAMBDA (int dummy) {
        myData(0) = val;
      } , DefaultLaunchConfig().set_stream(stream) );
    }

  };
}


