
#pragma once

namespace yakl {

  template <class T1, class T2>
  inline void memcpy_host_to_host(T1 *dst_p , T2 *src_p , size_t elems) {
    static_assert(std::is_same<typename std::remove_cv<T1>::type,typename std::remove_cv<T2>::type>::value);
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_host_to_host");
    #endif
    std::memcpy( dst_p , src_p , elems*sizeof(T1) );
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_host_to_host");
    #endif
  }

  inline void memcpy_host_to_host_void(void *dst_p , void *src_p , size_t bytes) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_host_to_host");
    #endif
    std::memcpy( dst_p , src_p , bytes );
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_host_to_host");
    #endif
  }

  template <class T1, class T2>
  inline void memcpy_device_to_host(T1 *dst_p , T2 *src_p , size_t elems ) {
    static_assert(std::is_same<typename std::remove_cv<T1>::type,typename std::remove_cv<T2>::type>::value);
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_device_to_host");
    #endif
    typedef typename std::remove_cv<T1>::type TNC;
    Kokkos::View<TNC *,Kokkos::LayoutRight,Kokkos::DefaultHostExecutionSpace::memory_space> dest(const_cast<TNC *>(dst_p),elems);
    Kokkos::View<TNC *,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space    > src (const_cast<TNC *>(src_p),elems);
    Kokkos::deep_copy( dest , src );
    #if defined(YAKL_AUTO_FENCE)
      Kokkos::fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_device_to_host");
    #endif
  }

  template <class T1, class T2>
  inline void memcpy_host_to_device(T1 *dst_p , T2 *src_p , size_t elems ) {
    static_assert(std::is_same<typename std::remove_cv<T1>::type,typename std::remove_cv<T2>::type>::value);
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_host_to_device");
    #endif
    typedef typename std::remove_cv<T1>::type TNC;
    Kokkos::View<TNC *,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space    > dest(const_cast<TNC *>(dst_p),elems);
    Kokkos::View<TNC *,Kokkos::LayoutRight,Kokkos::DefaultHostExecutionSpace::memory_space> src (const_cast<TNC *>(src_p),elems);
    Kokkos::deep_copy( dest , src );
    #if defined(YAKL_AUTO_FENCE)
      Kokkos::fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_host_to_device");
    #endif
  }

  template <class T1, class T2>
  inline void memcpy_device_to_device(T1 *dst_p , T2 *src_p , size_t elems ) {
    static_assert(std::is_same<typename std::remove_cv<T1>::type,typename std::remove_cv<T2>::type>::value);
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_device_to_device");
    #endif
    typedef typename std::remove_cv<T1>::type TNC;
    Kokkos::View<TNC *,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space> dest(const_cast<TNC *>(dst_p),elems);
    Kokkos::View<TNC *,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space> src (const_cast<TNC *>(src_p),elems);
    Kokkos::deep_copy( dest , src );
    #if defined(YAKL_AUTO_FENCE)
      Kokkos::fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_device_to_device");
    #endif
  }


  /**
   * @brief [USE AT YOUR OWN RISK]: memcpy the specified number of **bytes** on the device
   */
  inline void memcpy_device_to_device_void(void *dst_p , void *src_p , size_t bytes ) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_device_to_device");
    #endif
    Kokkos::View<char *,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space> dest((char *)dst_p,bytes/sizeof(char));
    Kokkos::View<char *,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space> src ((char *)src_p,bytes/sizeof(char));
    Kokkos::deep_copy( dest , src );
    #if defined(YAKL_AUTO_FENCE)
      Kokkos::fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_device_to_device");
    #endif
  }

}

