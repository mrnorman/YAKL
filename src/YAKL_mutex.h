
#pragma once


namespace yakl {
  // For thread safety in YAKL Array reference counters
  /** @private */
  extern std::mutex yakl_mtx;

  // YAKL's default allocation, free, mutex lock, and mutex unlock routines.
  /** @private */
  inline void yakl_mtx_lock  () { yakl_mtx.lock  (); }
  /** @private */
  inline void yakl_mtx_unlock() { yakl_mtx.unlock(); }
}


