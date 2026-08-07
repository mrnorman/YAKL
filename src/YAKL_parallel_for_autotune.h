
#pragma once

namespace yakl {
  namespace autotune {

    using ConfigListType = std::tuple<Config<0>,Config<64>,Config<128>,Config<256>,Config<512>>;
    inline constexpr std::array<size_t,4> tile_sizes = {1,2,4,8};
    inline constexpr int configuration_count = std::tuple_size_v<ConfigListType>*tile_sizes.size();

    struct AutotuneContext {
      int static constexpr tests_per_config = 5;
      int static constexpr total_tests      = tests_per_config*configuration_count;
      int tests_performed;
      int best_index;
      std::array<double,configuration_count> timings;
      AutotuneContext() {
        tests_performed = 0;
        best_index = 0;
        timings.fill(std::numeric_limits<double>::max());
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
        int const configIndex = index/static_cast<int>(tile_sizes.size());
        int const tileIndex   = index%static_cast<int>(tile_sizes.size());
        if (configIndex == I) {
          using ConfigType = std::tuple_element_t<I,ConfigListType>;
          yakl::parallel_for( str , bounds , f , ConfigType(tile_sizes[tileIndex]) );
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
        size_t const dim = d == 0 ? bounds.nIter/bounds.offs[0] : bounds.offs[d-1]/bounds.offs[d];
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
          auto t1 = std::chrono::high_resolution_clock::now();
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
          auto t2 = std::chrono::high_resolution_clock::now();
          auto time_loc = std::chrono::duration<double>(t2 - t1).count();
        #endif
        // Ignore first run, then sum the rest of the runs
        if (context.timings[index] == std::numeric_limits<double>::max()) { context.timings[index]  = 0;        }
        else                                                              { context.timings[index] += time_loc; }
        auto & v = context.timings;
        context.best_index = std::distance( v.begin() , std::min_element(v.begin(),v.end()) );
        context.tests_performed++;
      };
      if (autotune_contexts.contains(lab)) {
        auto & context = autotune_contexts[lab];
        if (context.tests_performed == AutotuneContext::total_tests) {
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
    inline std::pair<int,int> get_config(int index) {
      if constexpr (kokkos_debug) {
        if (index < 0 || index >= configuration_count) {
          Kokkos::abort("ERROR: autotune configuration index out of bounds");
        }
      }
      if constexpr (I < std::tuple_size_v<ConfigListType>) {
        int const configIndex = index/static_cast<int>(tile_sizes.size());
        int const tileIndex   = index%static_cast<int>(tile_sizes.size());
        if (configIndex == I) return std::make_pair(std::tuple_element_t<I,ConfigListType>::Thr,
                                                    static_cast<int>(tile_sizes[tileIndex]));
        else                  return get_config<I+1>(index);
      } else { return std::make_pair(0,0); }
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
            if (c.best_index < 0 || c.best_index >= configuration_count ||
                !std::isfinite(c.timings[0]) || !std::isfinite(c.timings[c.best_index]) ||
                c.timings[c.best_index] <= 0) {
              Kokkos::abort("ERROR: invalid autotune result");
            }
          }
          auto config = get_config(c.best_index);
          if (myrank == 0) std::cout << key << " : Config<" << std::get<0>(config) << ">{" << std::get<1>(config)
                                            << "} , Speedup: " << c.timings[0]/c.timings[c.best_index] << std::endl;
        }
      }
    }

  }
}
