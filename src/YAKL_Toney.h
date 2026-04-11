
#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <fstream>

namespace yakl {

  // Yes, this is Toney the Timer
  struct Toney {
    typedef std::chrono::high_resolution_clock Clock    ;
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
      TimePoint           previous_time_point;
      std::vector<size_t> child_hashes;
      int                 parent_index;
      bool                multiple_parents;
    };

    struct ActiveStackEntry {
      size_t label_hash;
      int    timer_index;
    };

    std::vector<Timer>            timers;
    std::vector<ActiveStackEntry> active_stack;


    void start(std::string label) {
      if (label.empty()) Kokkos::abort("ERROR: calling start() with an empty label");
      auto label_hash  = hasher( label );
      auto timer_index = get_or_create_timer_index( label , label_hash );
      auto &timer      = timers[timer_index];
      if ( ! active_stack.empty() ) {
        int  parent_timer_index   = active_stack.back().timer_index;
        auto &parent_timer        = timers[parent_timer_index];
        bool i_am_already_a_child = false;
        for ( auto &child_hash : parent_timer.child_hashes ) {
          if ( child_hash == label_hash ) { i_am_already_a_child = true; break; }
        }
        if ( ! i_am_already_a_child ) parent_timer.child_hashes.push_back( label_hash );
        if (timer.parent_index == parent_index_just_created) timer.parent_index = parent_timer_index;
        if (timer.parent_index != parent_timer_index) timer.multiple_parents = true;
      } else {
        if (timer.parent_index == parent_index_just_created) timer.parent_index = parent_index_main;
        if (timer.parent_index != parent_index_main) timer.multiple_parents = true;
      }
      active_stack.push_back( { label_hash , timer_index } );
      timer.hits++;
      timer.previous_time_point = Clock::now();
    }


    void stop(std::string label) {
      TimePoint now = Clock::now();
      if (label.empty()) Kokkos::abort("ERROR: calling stop() with an empty label");
      if ( hasher(label) != active_stack.back().label_hash ) Kokkos::abort("ERROR: timers must be perfectly nested");
      auto &timer = timers[active_stack.back().timer_index];
      Duration duration           = now - timer.previous_time_point;
      timer.max_duration          = std::max( timer.max_duration , duration );
      timer.min_duration          = std::min( timer.min_duration , duration );
      timer.accumulated_duration += duration;
      timer.last_duration         = duration;
      active_stack.pop_back();
    }


    int get_or_create_timer_index( std::string label , size_t label_hash ) {
      for ( int i=0; i < timers.size(); i++) {
        if ( label_hash == timers[i].label_hash ) return i;
      }
      timers.push_back( { label , label_hash , 0 , Duration::zero() , Duration::zero() , Duration::zero() ,
                          Duration::max() , TimePoint::min() , std::vector<size_t>() , parent_index_just_created ,
                          false } );
      return timers.size()-1;
    }


    int get_timer_id( size_t label_hash , bool die = true ) const {
      for (int i=0; i < timers.size(); i++) { if (label_hash == timers[i].label_hash) return i; }
      if (die) Kokkos::abort("ERROR: label not found in timers");
      return -1;
    }


    int get_timer_id( std::string label , bool die = true ) const {
      if (label.empty()) Kokkos::abort("ERROR: calling get_last_duration() with an empty label");
      auto label_hash = hasher( label );
      for (int i=0; i < timers.size(); i++) { if (label_hash == timers[i].label_hash) return i; }
      if (die) Kokkos::abort("ERROR: label not found in timers");
      return -1;
    }


    double get_last_duration(std::string label) const { return timers[get_timer_id(label)].last_duration.count(); }


    double get_accumulated_duration(std::string label) const { return timers[get_timer_id(label)].accumulated_duration.count(); }


    double get_min_duration(std::string label) const { return timers[get_timer_id(label)].min_duration.count(); }


    double get_max_duration(std::string label) const { return timers[get_timer_id(label)].max_duration.count(); }


    size_t get_count(std::string label) const { return timers[get_timer_id(label)].hits; }


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
      os << "******* Timers *******" << "\n";
      if (! active_stack.empty()) os << "WARNING: printing timers while some are still active. Results will be inaccurate\n";
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
        if (! printed[itimer]) print_timer_and_children( itimer , printed , level , os );
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
        for (int ichild = 0; ichild < timer.child_hashes.size(); ichild++) {
          int child_timer_index = get_timer_id( timer.child_hashes[ichild] );
          int level_loc = level + 1;
          print_timer_and_children( child_timer_index , printed , level_loc , os );
        }
      }
    }

  };

}

