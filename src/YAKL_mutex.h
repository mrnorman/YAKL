
#pragma once


namespace yakl {
  inline void yakl_mtx_lock  () { get_yakl_instance().yakl_mtx.lock  (); }
  inline void yakl_mtx_unlock() { get_yakl_instance().yakl_mtx.unlock(); }
}


