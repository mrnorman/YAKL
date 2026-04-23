
#pragma once

namespace yakl {
  inline void   timer_start                   (std::string l) { if constexpr (yakl_profile) { Kokkos::fence(); get_yakl_instance().timer.start         (l); } }
  inline void   timer_stop                    (std::string l) { if constexpr (yakl_profile) { Kokkos::fence(); get_yakl_instance().timer.stop          (l); } }
  inline double timer_get_last_duration       (std::string l) { if constexpr( yakl_profile) { return get_yakl_instance().timer.get_last_duration       (l); }; return 0; }
  inline double timer_get_accumulated_duration(std::string l) { if constexpr( yakl_profile) { return get_yakl_instance().timer.get_accumulated_duration(l); }; return 0; }
  inline double timer_get_min_duration        (std::string l) { if constexpr( yakl_profile) { return get_yakl_instance().timer.get_min_duration        (l); }; return 0; }
  inline double timer_get_max_duration        (std::string l) { if constexpr( yakl_profile) { return get_yakl_instance().timer.get_max_duration        (l); }; return 0; }
  inline size_t timer_get_count               (std::string l) { if constexpr( yakl_profile) { return get_yakl_instance().timer.get_count               (l); }; return 0; }
  inline void   timer_print                   (             ) { if constexpr (yakl_profile) {        get_yakl_instance().timer.print_main              ( ); } }
}

