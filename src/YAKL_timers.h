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

  /** @brief Initialize the YAKL timers */
  inline void timer_init    (               ) { }

  /** @brief Finalize the YAKL timers */
  inline void timer_finalize(               ) { get_yakl_instance().timer.print_all_threads(); }

  /** @brief Start a timer with the given string label. NOTE: Timers must be perfectly nested */
  inline void timer_start   (std::string lab) { get_yakl_instance().timer.start(lab); }

  /** @brief Stop a timer with the given string label. NOTE: Timers must be perfectly nested */
  inline void timer_stop    (std::string lab) { get_yakl_instance().timer.stop (lab); }
}

