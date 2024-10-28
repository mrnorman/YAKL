
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
    * parallel_for( Bounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
    *   // sum() obtains a modifiable reference to the data contained by the ScalarLiveOut object
    *   atomicAdd( sum() , 1. );
    * });
    * if (sum.hostRead() != ny*nx) yakl::Kokkos::abort("ERROR: Wrong sum");
    * ```
    * 
    * ```
    * ScalarLiveOut<bool> data_is_bad(false);
    * parallel_for( Bounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
    *   // operator= always assigns on the device
    *   if (density(j,i) < 0) data_is_bad = true;
    * });
    * if (data_is_bad.hostRead()) yakl::Kokkos::abort("ERROR: Data is bad");
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
    KOKKOS_INLINE_FUNCTION ScalarLiveOut() {
      data = Array<T,1,memDevice,styleC>("ScalarLiveOut_data",1);  // Create array
    }
    /** @brief [ASYNCHRONOUS] This constructor allocates room on the device for one scalar of type `T` and initializes it on device with the provided value. */
    explicit ScalarLiveOut(T val) {
      Array<T,1,memHost,styleC> host_data("ScalarLiveOut_data",1);  // Create array
      host_data(0) = val;
      data = host_data.createDeviceCopy();
    }
    /** @brief Deallocates the scalar value on the device. */
    KOKKOS_INLINE_FUNCTION ~ScalarLiveOut() {
      data = Array<T,1,memDevice,styleC>();
    }

    /** @brief Copies and moves are shallow, not deep copy. */
    KOKKOS_INLINE_FUNCTION ScalarLiveOut            ( ScalarLiveOut const &rhs) { this->data = rhs.data; }
    /** @brief Copies and moves are shallow, not deep copy. */
    KOKKOS_INLINE_FUNCTION ScalarLiveOut & operator=( ScalarLiveOut const &rhs) { this->data = rhs.data; return *this; }
    /** @brief Copies and moves are shallow, not deep copy. */
    KOKKOS_INLINE_FUNCTION ScalarLiveOut            ( ScalarLiveOut      &&rhs) { this->data = rhs.data; }
    /** @brief Copies and moves are shallow, not deep copy. */
    KOKKOS_INLINE_FUNCTION ScalarLiveOut & operator=( ScalarLiveOut      &&rhs) { this->data = rhs.data; return *this; }

    /** @brief Assign a value to the ScalarLiveOut object on the device. */
    template <class TLOC , typename std::enable_if< std::is_arithmetic<TLOC>::value , int >::type = 0>
    KOKKOS_INLINE_FUNCTION T &operator= (TLOC rhs) const { data(0) = rhs; return data(0); }

    /** @brief Returns a modifiable reference to the underlying data on the device. */
    KOKKOS_INLINE_FUNCTION T &operator() () const {
      return data(0);
    }

    /** @brief Returns a modifiable reference to the underlying data on the device. */
    KOKKOS_INLINE_FUNCTION T get() const {
      return data(0);
    }

    /** @brief Returns a host copy of the data. This is blocking. */
    inline T hostRead() const {
      return data.createHostCopy()(0);
    }

    /** @brief [ASYNCHRONOUS] Writes a value to the device-resident underlying data */
    inline void hostWrite(T val) {
      // Copy data to device
      YAKL_SCOPE( myData , this->data );
      c::parallel_for( YAKL_AUTO_LABEL() , c::Bounds<1>(1) , KOKKOS_LAMBDA (int dummy) {
        myData(0) = val;
      });
    }

  };
}


