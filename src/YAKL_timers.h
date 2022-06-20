
#pragma once
// Included by YAKL.h
// Inside the yakl namespace

#include "YAKL_Toney.h"

namespace yakl {
  extern Toney timer;

  extern std::function<void ()>             timer_init;
  extern std::function<void ()>             timer_finalize;
  extern std::function<void (char const *)> timer_start;
  extern std::function<void (char const *)> timer_stop;

  inline void set_timer_init    ( std::function<void ()>             func ) { timer_init     = func; }
  inline void set_timer_finalize( std::function<void ()>             func ) { timer_finalize = func; }
  inline void set_timer_start   ( std::function<void (char const *)> func ) { timer_start    = func; }
  inline void set_timer_stop    ( std::function<void (char const *)> func ) { timer_stop     = func; }
}

