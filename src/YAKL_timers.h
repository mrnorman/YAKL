/**
 * @file 
 *
 * Contains routines to use or override YAKL's timers
 */

#pragma once
// Included by YAKL.h
// Inside the yakl namespace

#include "YAKL_Toney.h"

namespace yakl {
  extern Toney timer;

  extern std::function<void ()>             timer_init_func;
  extern std::function<void ()>             timer_finalize_func;
  extern std::function<void (char const *)> timer_start_func;
  extern std::function<void (char const *)> timer_stop_func;

  /** @brief Initialize the YAKL timers */
  inline void timer_init    (               ) { timer_init_func    (   ); }

  /** @brief Finalize the YAKL timers */
  inline void timer_finalize(               ) { timer_finalize_func(   ); }

  /** @brief Start a timer with the given string label. NOTE: Timers must be perfectly nested */
  inline void timer_start   (char const *lab) { timer_start_func   (lab); }

  /** @brief Stop a timer with the given string label. NOTE: Timers must be perfectly nested */
  inline void timer_stop    (char const *lab) { timer_stop_func    (lab); }



  /** @brief Override YAKL's default timer initialization routine */
  inline void set_timer_init    ( std::function<void ()>             func ) { timer_init_func     = func; }

  /** @brief Override YAKL's default timer finalization routine */
  inline void set_timer_finalize( std::function<void ()>             func ) { timer_finalize_func = func; }

  /** @brief Override YAKL's default routine to start an individual timer */
  inline void set_timer_start   ( std::function<void (char const *)> func ) { timer_start_func    = func; }

  /** @brief Override YAKL's default routine to stop an individual timer */
  inline void set_timer_stop    ( std::function<void (char const *)> func ) { timer_stop_func     = func; }
}

