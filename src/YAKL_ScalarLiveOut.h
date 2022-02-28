
#pragma once

// Handles the case where a scalar value is written to in a kernel and must be read on the host after the kernel
// completes.
// Passing a value to the constructor will initialize the device pointer with that data from the host
// Using the = operator will assign to the value inside a kernel. This is the most common operation
// To access the value as a reference that is writable, use the operator() (e.g., in an atomic operation perhaps)
// To read on the host afterward, using hostRead()
// To write after construction, use hostWrite()

template <class T> class ScalarLiveOut {
public:
  Array<T,1,memDevice,styleC> data;

  // Contructors
  YAKL_INLINE ScalarLiveOut() {
    data = Array<T,1,memDevice,styleC>("data",1);  // Create array
  }
  explicit ScalarLiveOut(T val) {
    data = Array<T,1,memDevice,styleC>("data",1);  // Create array
    hostWrite(val);                                // Copy to device
  }
  YAKL_INLINE ~ScalarLiveOut() {
    data = Array<T,1,memDevice,styleC>();
  }

  // Copy and move constructors, all shallow copies of the data
  YAKL_INLINE ScalarLiveOut            ( ScalarLiveOut const &rhs) { this->data = rhs.data; }
  YAKL_INLINE ScalarLiveOut & operator=( ScalarLiveOut const &rhs) { this->data = rhs.data; return *this; }
  YAKL_INLINE ScalarLiveOut            ( ScalarLiveOut      &&rhs) { this->data = rhs.data; }
  YAKL_INLINE ScalarLiveOut & operator=( ScalarLiveOut      &&rhs) { this->data = rhs.data; return *this; }

  // assignment on the GPU for integral types
  template <class TLOC , typename std::enable_if< std::is_arithmetic<TLOC>::value , int >::type = 0>
  YAKL_INLINE T &operator= (TLOC rhs) const { data(0) = rhs; return data(0); }

  // Access on the GPU
  YAKL_INLINE T &operator() () const {
    return data(0);
  }

  // Access on the GPU
  YAKL_INLINE T get() const {
    return data(0);
  }

  // Read on the host, createHostCopy automatically inserts a fence() operation
  inline T hostRead() const {
    return data.createHostCopy()(0);
  }

  inline void hostWrite(T val) {
    // Copy data to device
    auto &myData = this->data;
    c::parallel_for( c::Bounds<1>(1) , YAKL_LAMBDA (int dummy) {
      myData(0) = val;
    });
  }

};



