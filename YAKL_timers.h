
#pragma once

#include "gptl.h"


inline void timer_start(char const * label) {
  #if defined(YAKL_PROFILE) || defined(YAKL_AUTO_PROFILE)
    fence();
    GPTLstart(label);
  #endif
}

inline void timer_stop(char const * label) {
  #if defined(YAKL_PROFILE) || defined(YAKL_AUTO_PROFILE)
    fence();
    GPTLstop(label);
  #endif
}



