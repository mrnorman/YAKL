
#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <fstream>
#include <mutex>
#include <thread>

namespace yakl {

  // Yes, this is Toney the Timer
  struct Toney {
    typedef std::chrono::steady_clock          Clock    ;
    typedef std::chrono::duration<double>      Duration ;
    typedef std::chrono::time_point<Clock>     TimePoint;

    int static constexpr parent_index_just_created = -100;
    int static constexpr parent_index_main         = -1;
    int static constexpr label_print_length        = 50;

    std::hash<std::string> hasher;

    struct Timer {
      std::string         label;
      size_t              label_hash;
      size_t              hits;
      Duration            accumulated_duration;
      Duration            last_duration;
      Duration            max_duration;
      Duration            min_duration;
      std::vector<int>    child_indices;
      int                 parent_index;
      bool                multiple_parents;
    };

    struct ActiveStackEntry {
      int       timer_index;
      TimePoint start_time;
    };

    std::vector<Timer> timers;
    std::unordered_map<std::thread::id,std::vector<ActiveStackEntry>> active_stacks;
    mutable std::mutex mutex;


    void start(std::string label) {
      if (label.empty()) Kokkos::abort("ERROR: calling start() with an empty label");
      std::lock_guard<std::mutex> lock(mutex);
      auto const label_hash  = hasher( label );
      auto const timer_index = get_or_create_timer_index_locked( label , label_hash );
      auto &timer            = timers[timer_index];
      auto &active_stack = active_stacks[std::this_thread::get_id()];
      if ( ! active_stack.empty() ) {
        int const parent_timer_index = active_stack.back().timer_index;
        // Recursive use of one label is an invocation of the same timer, not a self-edge in the display hierarchy.
        if (parent_timer_index != timer_index) {
          auto &parent_timer        = timers[parent_timer_index];
          bool i_am_already_a_child = false;
          for ( auto const &child_index : parent_timer.child_indices ) {
            if ( child_index == timer_index ) { i_am_already_a_child = true; break; }
          }
          if ( ! i_am_already_a_child ) parent_timer.child_indices.push_back( timer_index );
          if (timer.parent_index == parent_index_just_created) timer.parent_index = parent_timer_index;
          if (timer.parent_index != parent_timer_index) timer.multiple_parents = true;
        }
      } else {
        if (timer.parent_index == parent_index_just_created) timer.parent_index = parent_index_main;
        if (timer.parent_index != parent_index_main) timer.multiple_parents = true;
      }
      active_stack.push_back( { timer_index , Clock::now() } );
      timer.hits++;
    }


    void stop(std::string label) {
      TimePoint const now = Clock::now();
      if (label.empty()) Kokkos::abort("ERROR: calling stop() with an empty label");
      std::lock_guard<std::mutex> lock(mutex);
      auto const thread_id = std::this_thread::get_id();
      auto stack_it = active_stacks.find(thread_id);
      if (stack_it == active_stacks.end() || stack_it->second.empty()) {
        Kokkos::abort("ERROR: calling stop() when no timer is active on this thread");
      }
      auto &active_stack = stack_it->second;
      if constexpr (kokkos_bounds_debug) {
        if (active_stack.back().timer_index < 0 ||
            static_cast<size_t>(active_stack.back().timer_index) >= timers.size()) {
          Kokkos::abort("ERROR: active timer index is out of bounds");
        }
      }
      if (timers[active_stack.back().timer_index].label != label) Kokkos::abort("ERROR: timers must be perfectly nested");
      auto &timer = timers[active_stack.back().timer_index];
      Duration const duration     = now - active_stack.back().start_time;
      timer.max_duration          = std::max( timer.max_duration , duration );
      timer.min_duration          = std::min( timer.min_duration , duration );
      timer.accumulated_duration += duration;
      timer.last_duration         = duration;
      // Retain the empty per-thread stack so repeated timing regions reuse its storage.
      active_stack.pop_back();
    }


    int get_or_create_timer_index( std::string label , size_t label_hash ) {
      std::lock_guard<std::mutex> lock(mutex);
      return get_or_create_timer_index_locked(label,label_hash);
    }


    private:
    int get_or_create_timer_index_locked( std::string const &label , size_t label_hash ) {
      for ( int i=0; i < timers.size(); i++) {
        if ( label == timers[i].label ) return i;
      }
      timers.push_back( { label , label_hash , 0 , Duration::zero() , Duration::zero() , Duration::zero() ,
                          Duration::max() , std::vector<int>() , parent_index_just_created , false } );
      return static_cast<int>(timers.size()-1);
    }


    public:
    int get_timer_id( size_t label_hash , bool die = true ) const {
      std::lock_guard<std::mutex> lock(mutex);
      return get_timer_id_locked(label_hash,die);
    }


    private:
    int get_timer_id_locked( size_t label_hash , bool die = true ) const {
      int timer_index = -1;
      for (int i=0; i < timers.size(); i++) {
        if (label_hash == timers[i].label_hash) {
          if (timer_index != -1) Kokkos::abort("ERROR: timer hash identifies more than one label");
          timer_index = i;
        }
      }
      if (timer_index != -1) return timer_index;
      if (die) Kokkos::abort("ERROR: label not found in timers");
      return -1;
    }


    public:
    int get_timer_id( std::string label , bool die = true ) const {
      if (label.empty()) Kokkos::abort("ERROR: calling get_last_duration() with an empty label");
      std::lock_guard<std::mutex> lock(mutex);
      return get_timer_id_locked(label,die);
    }


    private:
    int get_timer_id_locked( std::string const &label , bool die = true ) const {
      if (label.empty()) Kokkos::abort("ERROR: calling a timer query with an empty label");
      for (int i=0; i < timers.size(); i++) { if (label == timers[i].label) return i; }
      if (die) Kokkos::abort("ERROR: label not found in timers");
      return -1;
    }


    public:
    double get_last_duration(std::string label) const {
      std::lock_guard<std::mutex> lock(mutex);
      return timers[get_timer_id_locked(label)].last_duration.count();
    }


    double get_accumulated_duration(std::string label) const {
      std::lock_guard<std::mutex> lock(mutex);
      return timers[get_timer_id_locked(label)].accumulated_duration.count();
    }


    double get_min_duration(std::string label) const {
      std::lock_guard<std::mutex> lock(mutex);
      return timers[get_timer_id_locked(label)].min_duration.count();
    }


    double get_max_duration(std::string label) const {
      std::lock_guard<std::mutex> lock(mutex);
      return timers[get_timer_id_locked(label)].max_duration.count();
    }


    size_t get_count(std::string label) const {
      std::lock_guard<std::mutex> lock(mutex);
      return timers[get_timer_id_locked(label)].hits;
    }


    void clear() {
      std::lock_guard<std::mutex> lock(mutex);
      timers.clear();
      active_stacks.clear();
    }


    void print_main(std::ostream & os = std::cout) const {
      #ifndef HAVE_MPI
        print(os);
      #else
        int myrank;
        MPI_Comm_rank( MPI_COMM_WORLD , &myrank );
        if (myrank == 0) print(os);
      #endif
    }


    void print( std::ostream & os = std::cout ) const {
      std::lock_guard<std::mutex> lock(mutex);
      os << "******* Timers *******" << "\n";
      if (has_active_timers_locked()) {
        os << "WARNING: printing timers while some are still active. Results will be inaccurate\n";
      }
      std::vector<bool> printed( timers.size() , false );
      os << "________________________________________________________________________________________________________\n";
      os << std::setw(label_print_length) << std::left << "Timer label"
         << std::setw(12) << std::left << "# calls"
         << std::setw(15) << std::left << "Total time"
         << std::setw(15) << std::left << "Min time"
         << std::setw(15) << std::left << "Max time" << "\n";
      os << "________________________________________________________________________________________________________\n";
      for (int itimer = 0; itimer < timers.size(); itimer++) {
        int level = 0;
        if (! printed[itimer]) print_timer_and_children_locked( itimer , printed , level , os );
      }
      os << "________________________________________________________________________________________________________\n"
         << "The ~ character beginning a timer label indicates it has multiple parent timers.\n"
         << "Thus, those timers will likely not accumulate like you expect them to.\n";
      os << std::endl << std::endl;
    }


    void print_timer_and_children( int timer_index              ,
                                   std::vector<bool> & printed  ,
                                   int &level                   ,
                                   std::ostream & os            ) const {
      std::lock_guard<std::mutex> lock(mutex);
      print_timer_and_children_locked(timer_index,printed,level,os);
    }


    private:
    bool has_active_timers_locked() const {
      for (auto const &entry : active_stacks) {
        if (! entry.second.empty()) return true;
      }
      return false;
    }


    void print_timer_and_children_locked( int timer_index              ,
                                          std::vector<bool> & printed  ,
                                          int &level                   ,
                                          std::ostream & os            ) const {
      if constexpr (kokkos_bounds_debug) {
        if (timer_index < 0 || static_cast<size_t>(timer_index) >= timers.size() ||
            printed.size() != timers.size()) {
          Kokkos::abort("ERROR: invalid timer index or printed-vector size");
        }
      }
      auto & timer = timers[timer_index];
      if (! printed[timer_index]) {
        std::string label = timer.label;
        if (timer.multiple_parents) label = std::string("~") + label;
        for (int i=0; i < level; i++) { label = std::string("  ")+label; }
        label.resize( std::min(label_print_length-2,(int)label.size()) );
        os << std::setw(label_print_length) << std::left << label
           << std::setw(12) << std::left << timer.hits
           << std::setw(15) << std::left << std::scientific << std::setprecision(6) << timer.accumulated_duration.count()
           << std::setw(15) << std::left << std::scientific << std::setprecision(6) << timer.min_duration.count()
           << std::setw(15) << std::left << std::scientific << std::setprecision(6) << timer.max_duration.count() << "\n";
        printed[timer_index] = true;
        for (int ichild = 0; ichild < timer.child_indices.size(); ichild++) {
          int child_timer_index = timer.child_indices[ichild];
          int level_loc = level + 1;
          print_timer_and_children_locked( child_timer_index , printed , level_loc , os );
        }
      }
    }

  };

}
