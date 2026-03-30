
#pragma once

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace yakl {
  class DeviceSpace {
    public:
    using memory_space    = yakl::DeviceSpace;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using device_type     = Kokkos::Device<execution_space,memory_space>;
    using size_type       = Kokkos::DefaultExecutionSpace::memory_space::size_type;
    DeviceSpace()                    = default; // default constructible
    DeviceSpace(const DeviceSpace &) = default; // copy constructible
    ~DeviceSpace()                   = default; // destructible
    static const char * name() { return "yakl::DeviceSpace"; }
    template <class Ex> void * allocate(Ex const & /*ex*/ , size_t const sz) const {
      return alloc_device(sz,"[Unlabeled]");
    }
    template <class Ex> void * allocate(Ex const & /*ex*/ , char const * label , size_t const sz , size_t const /*logical_sz*/=0) const {
      return alloc_device(sz,label);
    }
    void * allocate(size_t const sz) const {
      return alloc_device(sz,"[Unlabeled]");
    }
    void * allocate(char const * label , size_t const sz , size_t const /*logical_sz*/=0) const {
      return alloc_device(sz,label);
    }
    void deallocate(void * const ptr , size_t sz) const {
      free_device(ptr,"[Unlabeled]");
    }
    void deallocate(char const * label, void * const ptr , size_t const sz , size_t const /*logical_sz*/=0 ) const {
      free_device(ptr,label);
    }
  };
}


#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(yakl::DeviceSpace);
#else
KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_SPECIALIZATION(yakl::DeviceSpace);
#endif


namespace Kokkos {
  namespace Impl {
    // VerifyExecutionCanAccessMemorySpace appears to not be there anymore?

    #ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL

    template <> struct MemorySpaceAccess<HostSpace,yakl::DeviceSpace> {
      enum : bool { assignable = true };
      enum : bool { accessible = true };
      enum : bool { deepcopy   = true };
    };

    template <> struct MemorySpaceAccess<yakl::DeviceSpace,HostSpace> {
      enum : bool { assignable = true };
      enum : bool { accessible = true };
      enum : bool { deepcopy   = true };
    };

    #else

    template <> struct MemorySpaceAccess<HostSpace,yakl::DeviceSpace> {
      enum : bool { assignable = false };
      enum : bool { accessible = false };
      enum : bool { deepcopy   = true  };
    };

    template <> struct MemorySpaceAccess<yakl::DeviceSpace,HostSpace> {
      enum : bool { assignable = false };
      enum : bool { accessible = false };
      enum : bool { deepcopy   = true  };
    };

    template <> struct MemorySpaceAccess<Kokkos::DefaultExecutionSpace::memory_space,yakl::DeviceSpace> {
      enum : bool { assignable = true  };
      enum : bool { accessible = true  };
      enum : bool { deepcopy   = true  };
    };

    template <> struct MemorySpaceAccess<yakl::DeviceSpace,Kokkos::DefaultExecutionSpace::memory_space> {
      enum : bool { assignable = true  };
      enum : bool { accessible = true  };
      enum : bool { deepcopy   = true  };
    };
    #endif

    template <typename ExecSpace>
    struct DeepCopy<Kokkos::HostSpace,yakl::DeviceSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::HostSpace,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::HostSpace,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };

    template <typename ExecSpace>
    struct DeepCopy<yakl::DeviceSpace,Kokkos::HostSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::HostSpace,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::HostSpace,ExecSpace>(exec,dst,src,n);
      }
    };

    template <typename ExecSpace>
    struct DeepCopy<yakl::DeviceSpace,yakl::DeviceSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };

    #ifndef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
    template <typename ExecSpace>
    struct DeepCopy<yakl::DeviceSpace,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };

    template <typename ExecSpace>
    struct DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,yakl::DeviceSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };
    #endif

  }
}


