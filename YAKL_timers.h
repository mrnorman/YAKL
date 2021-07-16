
#pragma once

#include "gptl.h"


inline void timer_start(char const * label) {
  #ifdef YAKL_PROFILE
    fence();
    GPTLstart(label);
  #endif
}

inline void timer_stop(char const * label) {
  #ifdef YAKL_PROFILE
    fence();
    GPTLstop(label);
  #endif
}



