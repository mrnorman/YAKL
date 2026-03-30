
#pragma once
// Included by YAKL.h

namespace yakl {

  template <class T> class ScalarLiveOut {
  protected:
    Array<T *,yakl::DeviceSpace> data;

  public:
    KOKKOS_INLINE_FUNCTION ScalarLiveOut() {
      data = Array<T *,yakl::DeviceSpace>("ScalarLiveOut_data",1);  // Create array
    }

    explicit ScalarLiveOut(T val) {
      Array<T *,Kokkos::HostSpace> host_data("ScalarLiveOut_data",1);  // Create array
      host_data(0) = val;
      data = host_data.createDeviceCopy();
    }

    KOKKOS_INLINE_FUNCTION ~ScalarLiveOut() {
      data = Array<T *,yakl::DeviceSpace>();
    }

    KOKKOS_INLINE_FUNCTION ScalarLiveOut            ( ScalarLiveOut const &rhs) { this->data = rhs.data; }
    KOKKOS_INLINE_FUNCTION ScalarLiveOut & operator=( ScalarLiveOut const &rhs) { this->data = rhs.data; return *this; }
    KOKKOS_INLINE_FUNCTION ScalarLiveOut            ( ScalarLiveOut      &&rhs) { this->data = rhs.data; }
    KOKKOS_INLINE_FUNCTION ScalarLiveOut & operator=( ScalarLiveOut      &&rhs) { this->data = rhs.data; return *this; }

    template <class TLOC> requires std::is_arithmetic_v<TLOC>
    KOKKOS_INLINE_FUNCTION T &operator= (TLOC rhs) const { data(0) = rhs; return data(0); }

    KOKKOS_INLINE_FUNCTION T &operator() () const {
      return data(0);
    }

    KOKKOS_INLINE_FUNCTION T get() const {
      return data(0);
    }

    inline T hostRead() const {
      return data.createHostCopy()(0);
    }

    inline void hostWrite(T val) {
      // Copy data to device
      YAKL_SCOPE( myData , this->data );
      parallel_for( YAKL_AUTO_LABEL() , 1 , KOKKOS_LAMBDA (int dummy) {
        myData(0) = val;
      });
      if constexpr (yakl_auto_fence) Kokkos::fence();
    }

  };
}


