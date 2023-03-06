
#pragma once


__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  // YAKL's default allocation, free, mutex lock, and mutex unlock routines.
  /** @private */
  inline void yakl_mtx_lock  () { get_yakl_instance().yakl_mtx.lock  (); }
  /** @private */
  inline void yakl_mtx_unlock() { get_yakl_instance().yakl_mtx.unlock(); }
}
__YAKL_NAMESPACE_WRAPPER_END__


