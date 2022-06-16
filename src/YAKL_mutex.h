
#pragma once


namespace yakl {
  // For thread safety in YAKL Array reference counters
  extern std::mutex yakl_mtx;

  // YAKL's default allocation, free, mutex lock, and mutex unlock routines.
  inline void yakl_mtx_lock  () { yakl_mtx.lock  (); }
  inline void yakl_mtx_unlock() { yakl_mtx.unlock(); }
}


