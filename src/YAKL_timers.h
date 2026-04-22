
#pragma once

namespace yakl {
  inline void   timer_start                   (std::string l) { Kokkos::fence(); get_yakl_instance().timer.start         (l); }
  inline void   timer_stop                    (std::string l) { Kokkos::fence(); get_yakl_instance().timer.stop          (l); }
  inline double timer_get_last_duration       (std::string l) { return get_yakl_instance().timer.get_last_duration       (l); }
  inline double timer_get_accumulated_duration(std::string l) { return get_yakl_instance().timer.get_accumulated_duration(l); }
  inline double timer_get_min_duration        (std::string l) { return get_yakl_instance().timer.get_min_duration        (l); }
  inline double timer_get_max_duration        (std::string l) { return get_yakl_instance().timer.get_max_duration        (l); }
  inline size_t timer_get_count               (std::string l) { return get_yakl_instance().timer.get_count               (l); }
  inline void   timer_print                   (             ) {        get_yakl_instance().timer.print_main              ( ); }
}

