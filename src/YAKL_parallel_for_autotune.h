
#pragma once

namespace yakl {
  namespace autotune {

    using ConfigListType = std::tuple<Config<0>,Config<64>,Config<128>,Config<256>,Config<512>,Config<1024>>;
    inline constexpr int configuration_count = std::tuple_size_v<ConfigListType>;

    struct AutotuneContext {
      int static constexpr tests_per_config = 5;
      int static constexpr total_tests      = tests_per_config*configuration_count;
      int tests_performed;
      int best_index;
      std::array<double,configuration_count> timings;
      std::array<int,configuration_count> sample_counts;
      AutotuneContext() {
        tests_performed = 0;
        best_index = -1;
        timings.fill(0);
        sample_counts.fill(0);
      }
      AutotuneContext(AutotuneContext const &)             = default;
      AutotuneContext(AutotuneContext &&)                  = default;
      AutotuneContext & operator=(AutotuneContext const &) = default;
      AutotuneContext & operator=(AutotuneContext &&)      = default;
      ~AutotuneContext()                                   = default;
    };



    inline std::unordered_map<std::string,AutotuneContext> autotune_contexts;



    template <int I> struct CurrInd {}; // For iterating through the launch_config function by parameter



    template <class F, int N, bool simple, class Style, int I=0>
    inline void launch_config( int                            index  ,
                               std::string                    str    ,
                               Bounds<N,Style,simple> const & bounds ,
                               F                      const & f      ,
                               CurrInd<I> = CurrInd<0>{} ) {
      if constexpr (kokkos_debug) {
        if (index < 0 || index >= configuration_count) {
          Kokkos::abort("ERROR: autotune configuration index out of bounds");
        }
      }
      if constexpr (I < std::tuple_size_v<ConfigListType>) {
        if (index == I) {
          using ConfigType = std::tuple_element_t<I,ConfigListType>;
          yakl::parallel_for( str , bounds , f , ConfigType() );
        } else {
          launch_config( index , str , bounds , f , CurrInd<I+1>{} );
        }
      }
    }



    template <class F, int N, bool simple, class Style>
    inline void parallel_for( std::string                    str    ,
                              Bounds<N,Style,simple> const & bounds ,
                              F                      const & f      ) {
      if (bounds.nIter == 0) {
        yakl::parallel_for(str,bounds,f);
        return;
      }
      auto lab = str+std::string(":");
      for (int d=0; d < N; d++) {
        uindex_t const dim = d == 0 ? bounds.nIter/bounds.offs[0] : bounds.offs[d-1]/bounds.offs[d];
        lab += std::to_string(dim);
        if (d != N-1) lab += "x";
      }
      lab += "_iterations";
      auto time_and_visit = [&] (int index , AutotuneContext & context) {
        if constexpr (kokkos_debug) {
          if (index < 0 || index >= configuration_count || context.tests_performed < 0 ||
              context.tests_performed >= AutotuneContext::total_tests) {
            Kokkos::abort("ERROR: invalid autotune context or configuration index");
          }
        }
        #if   defined(KOKKOS_ENABLE_CUDA)
          cudaEvent_t start, stop;
          if (cudaEventCreate(&start)  != cudaSuccess) Kokkos::abort("ERROR: failed event creation");
          if (cudaEventCreate(&stop)   != cudaSuccess) Kokkos::abort("ERROR: failed event creation");
          if (cudaEventRecord(start,0) != cudaSuccess) Kokkos::abort("ERROR: failed event record"  );
        #elif defined(KOKKOS_ENABLE_HIP)
          hipEvent_t start, stop;
          if (hipEventCreate(&start)  != hipSuccess) Kokkos::abort("ERROR: failed event creation");
          if (hipEventCreate(&stop)   != hipSuccess) Kokkos::abort("ERROR: failed event creation");
          if (hipEventRecord(start,0) != hipSuccess) Kokkos::abort("ERROR: failed event record"  );
        #else
          Kokkos::fence();
          auto const t1 = std::chrono::steady_clock::now();
        #endif
        launch_config(index,str,bounds,f);
        #if   defined(KOKKOS_ENABLE_CUDA)
          float time_loc = 0;
          if (cudaEventRecord(stop,0)                    != cudaSuccess) Kokkos::abort("ERROR: failed event record" );
          if (cudaEventSynchronize(stop)                 != cudaSuccess) Kokkos::abort("ERROR: failed event sync"   );
          if (cudaEventElapsedTime(&time_loc,start,stop) != cudaSuccess) Kokkos::abort("ERROR: failed event elapsed");
          if (cudaEventDestroy(start)                    != cudaSuccess) Kokkos::abort("ERROR: failed event destroy");
          if (cudaEventDestroy(stop)                     != cudaSuccess) Kokkos::abort("ERROR: failed event destroy");
        #elif defined(KOKKOS_ENABLE_HIP)
          float time_loc = 0;
          if (hipEventRecord(stop,0)                    != hipSuccess) Kokkos::abort("ERROR: failed event record" );
          if (hipEventSynchronize(stop)                 != hipSuccess) Kokkos::abort("ERROR: failed event sync"   );
          if (hipEventElapsedTime(&time_loc,start,stop) != hipSuccess) Kokkos::abort("ERROR: failed event elapsed");
          if (hipEventDestroy(start)                    != hipSuccess) Kokkos::abort("ERROR: failed event destroy");
          if (hipEventDestroy(stop)                     != hipSuccess) Kokkos::abort("ERROR: failed event destroy");
        #else
          Kokkos::fence();
          auto const t2 = std::chrono::steady_clock::now();
          auto time_loc = std::chrono::duration<double>(t2 - t1).count();
        #endif
        // Each configuration starts with a warmup. Keep it separate from measured samples so an incomplete tuning
        // cycle cannot select that zero-valued warmup as its best result.
        bool const is_warmup = context.tests_performed%AutotuneContext::tests_per_config == 0;
        if (!is_warmup) {
          context.timings[index] += time_loc;
          context.sample_counts[index]++;
          context.best_index = -1;
          for (int i=0; i < configuration_count; i++) {
            if (context.sample_counts[i] == 0) continue;
            if (context.best_index < 0 ||
                context.timings[i]/context.sample_counts[i] <
                context.timings[context.best_index]/context.sample_counts[context.best_index]) {
              context.best_index = i;
            }
          }
        }
        context.tests_performed++;
      };
      if (autotune_contexts.contains(lab)) {
        auto & context = autotune_contexts[lab];
        if (context.tests_performed == AutotuneContext::total_tests) {
          if constexpr (kokkos_debug) {
            if (context.best_index < 0 || context.best_index >= configuration_count) {
              Kokkos::abort("ERROR: completed autotune context has no valid configuration");
            }
          }
          launch_config(context.best_index,str,bounds,f);
        } else {
          int index = context.tests_performed / AutotuneContext::tests_per_config;
          time_and_visit(index,context);
        }
      } else {
        autotune_contexts[lab] = AutotuneContext();
        time_and_visit(0,autotune_contexts[lab]);
      }
    }

    template <class F>
    inline void parallel_for( std::string str , std::integral auto bnd , F const & f ) {
      yakl::autotune::parallel_for( str , Bounds<1,CStyle,true>(bnd) , f );
    }



    template <class F, int N, bool simple>
    inline void parallel_for_F( std::string str , Bounds<N,FStyle,simple> const & bounds , F const & f ) {
      parallel_for<F,N,simple,FStyle>( str , bounds , f );
    }

    template <class F>
    inline void parallel_for_F( std::string str , std::integral auto bnd , F const & f ) {
      yakl::autotune::parallel_for<F,1,true,FStyle>( str , Bounds<1,FStyle,true>(bnd) , f );
    }



    template <int I=0>
    inline int get_config(int index) {
      if constexpr (kokkos_debug) {
        if (index < 0 || index >= configuration_count) {
          Kokkos::abort("ERROR: autotune configuration index out of bounds");
        }
      }
      if constexpr (I < std::tuple_size_v<ConfigListType>) {
        if (index == I) return std::tuple_element_t<I,ConfigListType>::Thr;
        else            return get_config<I+1>(index);
      } else { return 0; }
    }



    inline void print_best() {
      if (! yakl::autotune::autotune_contexts.empty()) {
        #ifndef HAVE_MPI
          int myrank = 0;
        #else
          int myrank;
          MPI_Comm_rank( MPI_COMM_WORLD , &myrank );
        #endif
        if (myrank == 0) std::cout << "\n*** AUTOTUNE RESULTS ***\n";
        for (auto const & [key,c] : autotune_contexts) {
          if constexpr (kokkos_debug) {
            if (c.tests_performed < 0 || c.tests_performed > AutotuneContext::total_tests) {
              Kokkos::abort("ERROR: invalid autotune launch count");
            }
            for (int i=0; i < configuration_count; i++) {
              if (c.sample_counts[i] < 0 || c.sample_counts[i] >= AutotuneContext::tests_per_config ||
                  !std::isfinite(c.timings[i]) || c.timings[i] < 0) {
                Kokkos::abort("ERROR: invalid autotune timing state");
              }
            }
          }
          if (c.best_index < 0) {
            if (myrank == 0) {
              std::cout << key << " : Tuning incomplete (" << c.tests_performed << "/"
                        << AutotuneContext::total_tests << " launches); no timed samples completed" << std::endl;
            }
            continue;
          }
          if constexpr (kokkos_debug) {
            if (c.best_index >= configuration_count || c.sample_counts[c.best_index] == 0) {
              Kokkos::abort("ERROR: invalid autotune best configuration");
            }
          }
          int const maxThreads = get_config(c.best_index);
          double const default_time = c.sample_counts[0] > 0 ? c.timings[0]/c.sample_counts[0] : 0;
          double const best_time    = c.timings[c.best_index]/c.sample_counts[c.best_index];
          if (myrank == 0) {
            std::cout << key << " : Config<" << maxThreads << "> , Speedup: ";
            if (c.sample_counts[0] > 0 && best_time > 0) {
              std::cout << default_time/best_time;
            } else {
              std::cout << "unavailable";
            }
            if (c.tests_performed < AutotuneContext::total_tests) {
              std::cout << " , Tuning incomplete (" << c.tests_performed << "/" << AutotuneContext::total_tests << " launches)";
            }
            std::cout << std::endl;
          }
        }
      }
    }

  }
}
