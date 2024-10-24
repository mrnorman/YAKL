
#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <iomanip>
#include <vector>

namespace yakl {

  // Yes, this is Toney the Timer
  class Toney {
  public:
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

    struct ThreadData {
      std::thread::id               thread_id;
      std::vector<Timer>            timers;
      std::vector<ActiveStackEntry> active_stack;
    };

    std::vector<ThreadData> threads;  // List of timer lists (one per thread)


    void start(std::string label) {
      if (label.empty()) die("ERROR: calling start() with an empty label");
      // Get the timer and active stack for this thread and label
      auto thread_id     = std::this_thread::get_id();
      auto label_hash    = hasher( label );
      auto thread_index  = get_or_create_thread_index( thread_id );
      auto timer_index   = get_or_create_timer_index( thread_index , label , label_hash );
      auto &timer        = threads[thread_index].timers[timer_index];
      auto &active_stack = threads[thread_index].active_stack;
      // If the active stack isn't empty, then register my timer as a child of the stack top timer
      if ( ! active_stack.empty() ) {
        int parent_timer_index = active_stack.back().timer_index;
        auto &parent_timer = threads[thread_index].timers[parent_timer_index];
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
      // Push this timer onto the active stack
      active_stack.push_back( { label_hash , timer_index } );
      // Register the time point to start the timer
      timer.hits++;
      timer.previous_time_point = Clock::now();
    }


    void stop(std::string label) {
      TimePoint now = Clock::now();
      if (label.empty()) die("ERROR: calling stop() with an empty label");
      auto thread_id     = std::this_thread::get_id();
      auto label_hash    = hasher( label );
      auto thread_index  = get_or_create_thread_index( thread_id );
      auto &active_stack = threads[thread_index].active_stack;
      // Make sure the timers are properly nested to guarantee the active stack's back timer_index is appropriate
      if ( label_hash != active_stack.back().label_hash ) die("ERROR: timers must be perfectly nested");
      // Instead of searching for the timer index, use the one from the back of the active stack
      auto &timer        = threads[thread_index].timers[active_stack.back().timer_index];
      // Set the timer stats
      Duration duration           = now - timer.previous_time_point;
      timer.max_duration          = max( timer.max_duration , duration );
      timer.min_duration          = min( timer.min_duration , duration );
      timer.accumulated_duration += duration;
      timer.last_duration         = duration;
      // Remove this timer from the active stack
      active_stack.pop_back();
    }


    int get_or_create_thread_index( std::thread::id thread_id ) {
      for (int i=0; i < threads.size(); i++) {
        if (threads[i].thread_id == thread_id) return i;
      }
      // If we get here, the thread wasn't found, and we need to create one
      threads.push_back( { thread_id , std::vector<Timer>() , std::vector<ActiveStackEntry>() } );
      return threads.size()-1;
    }


    int get_or_create_timer_index( int thread_index , std::string label , size_t label_hash ) {
      auto &timers = threads[thread_index].timers;
      for ( int i=0; i < timers.size(); i++) {
        if ( label_hash == timers[i].label_hash ) return i;
      }
      // If we get here, the timer wasn't found, and we need to create one
      timers.push_back( { label , label_hash , 0 , Duration::zero() , Duration::zero() , Duration::zero() ,
                          Duration::max() , TimePoint::min() , std::vector<size_t>() , parent_index_just_created ,
                          false } );
      return timers.size()-1;
    }


    double get_last_duration(std::string label) {
      if (label.empty()) die("ERROR: calling get_last_duration() with an empty label");
      auto label_hash = hasher( label );
      auto &timers = threads[get_or_create_thread_index( std::this_thread::get_id() )].timers;
      int id = -1;
      for (int i=0; i < timers.size(); i++) { if (label_hash == timers[i].label_hash) id = i; }
      if (id == -1) die("ERROR: label not found in timers");
      return timers[id].last_duration.count();
    }


    double get_accumulated_duration(std::string label) {
      if (label.empty()) die("ERROR: calling get_last_duration() with an empty label");
      auto label_hash = hasher( label );
      auto &timers = threads[get_or_create_thread_index( std::this_thread::get_id() )].timers;
      int id = -1;
      for (int i=0; i < timers.size(); i++) { if (label_hash == timers[i].label_hash) id = i; }
      if (id == -1) die("ERROR: label not found in timers");
      return timers[id].accumulated_duration.count();
    }


    double get_min_duration(std::string label) {
      if (label.empty()) die("ERROR: calling get_last_duration() with an empty label");
      auto label_hash = hasher( label );
      auto &timers = threads[get_or_create_thread_index( std::this_thread::get_id() )].timers;
      int id = -1;
      for (int i=0; i < timers.size(); i++) { if (label_hash == timers[i].label_hash) id = i; }
      if (id == -1) die("ERROR: label not found in timers");
      return timers[id].min_duration.count();
    }


    double get_max_duration(std::string label) {
      if (label.empty()) die("ERROR: calling get_last_duration() with an empty label");
      auto label_hash = hasher( label );
      auto &timers = threads[get_or_create_thread_index( std::this_thread::get_id() )].timers;
      int id = -1;
      for (int i=0; i < timers.size(); i++) { if (label_hash == timers[i].label_hash) id = i; }
      if (id == -1) die("ERROR: label not found in timers");
      return timers[id].max_duration.count();
    }


    double get_count(std::string label) {
      if (label.empty()) die("ERROR: calling get_last_duration() with an empty label");
      auto label_hash = hasher( label );
      auto &timers = threads[get_or_create_thread_index( std::this_thread::get_id() )].timers;
      int id = -1;
      for (int i=0; i < timers.size(); i++) { if (label_hash == timers[i].label_hash) id = i; }
      if (id == -1) die("ERROR: label not found in timers");
      return timers[id].hits;
    }


    void print_my_thread( bool all_tasks = false ) {
      #ifndef HAVE_MPI
        print_thread( get_or_create_thread_index( std::this_thread::get_id() ) );
        return;
      #else
        int myrank;
        int nranks;
        MPI_Comm_rank( MPI_COMM_WORLD , &myrank );
        MPI_Comm_size( MPI_COMM_WORLD , &nranks );
        if (all_tasks) {
          for (int irank = 0; irank < nranks; irank++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (irank == myrank) {
              std::cout << "\n***************************************************\n";
              std::cout <<   "** FOR RANK " << myrank << "**\n";
              std::cout <<   "***************************************************\n";
              print_thread( get_or_create_thread_index( std::this_thread::get_id() ) );
              std::cout << "\n" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
          }
        } else {
          if (myrank == 0) {
            print_thread( get_or_create_thread_index( std::this_thread::get_id() ) );
          }
        }
      #endif
    }


    void print_all_threads( bool all_tasks = false ) {
      #ifndef HAVE_MPI
        for (int ithread = 0; ithread < threads.size(); ithread++) { print_thread(ithread); }
        return;
      #else
        int myrank;
        int nranks;
        MPI_Comm_rank( MPI_COMM_WORLD , &myrank );
        MPI_Comm_size( MPI_COMM_WORLD , &nranks );
        if (all_tasks) {
          for (int irank = 0; irank < nranks; irank++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (irank == myrank) {
              std::cout << "\n***************************************************\n";
              std::cout <<   "** FOR RANK " << myrank << "**\n";
              std::cout <<   "***************************************************\n";
              for (int ithread = 0; ithread < threads.size(); ithread++) { print_thread(ithread); }
              std::cout << "\n" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
          }
        } else {
          if (myrank == 0) {
            for (int ithread = 0; ithread < threads.size(); ithread++) { print_thread(ithread); }
          }
        }
      #endif
    }


    void print_thread( int ithread ) {
      std::cout << "******* Timers for thread " << ithread << " *******" << "\n";
      if (! threads[ithread].active_stack.empty())
        std::cout << "WARNING: printing timers while some are still active. Results will be inaccurate\n";
      auto &timers = threads[ithread].timers;
      std::vector<bool> printed ( timers.size() , false );
      std::cout << "________________________________________________________________________________________________________\n";
      std::cout << std::setw(label_print_length) << std::left << "Timer label"
                << std::setw(12) << std::left << "# calls"
                << std::setw(15) << std::left << "Total time"
                << std::setw(15) << std::left << "Min time"
                << std::setw(15) << std::left << "Max time" << "\n";
      std::cout << "________________________________________________________________________________________________________\n";
      // TODO: Find a way to detect timers with multiple parents
      for (int itimer = 0; itimer < timers.size(); itimer++) {
        int level = 0;
        if (! printed[itimer]) print_timer_and_children( ithread , itimer , printed , level );
      }
      std::cout << "________________________________________________________________________________________________________\n"
                << "The ~ character beginning a timer label indicates it has multiple parent timers.\n"
                << "Thus, those timers will likely not accumulate like you expect them to.\n";
      std::cout << std::endl << std::endl;
    }


    void print_timer_and_children( int thread_index , int timer_index , std::vector<bool> &printed  ,
                                                                        int &level ) {
      auto &timer = threads[thread_index].timers[timer_index];
      if (! printed[timer_index]) {
        std::string label = timer.label;
        if (timer.multiple_parents) label = std::string("~") + label;
        for (int i=0; i < level; i++) { label = std::string("  ")+label; }
        label.resize( std::min(label_print_length-2,(int)label.size()) );
        std::cout << std::setw(label_print_length) << std::left << label
                  << std::setw(12) << std::left << timer.hits
                  << std::setw(15) << std::left << std::scientific << std::setprecision(6) << timer.accumulated_duration.count()
                  << std::setw(15) << std::left << std::scientific << std::setprecision(6) << timer.min_duration.count()
                  << std::setw(15) << std::left << std::scientific << std::setprecision(6) << timer.max_duration.count() << "\n";
        printed[timer_index] = true;
        for (int ichild = 0; ichild < timer.child_hashes.size(); ichild++) {
          int child_timer_index = get_or_create_timer_index( thread_index , "" , timer.child_hashes[ichild] );
          int level_loc = level + 1;
          print_timer_and_children( thread_index , child_timer_index , printed , level_loc );
        }
      }
    }


    void die(std::string msg) { std::cerr << msg << std::endl; throw std::runtime_error(msg); }
  };

}

