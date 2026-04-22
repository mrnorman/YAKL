
#pragma once

namespace yakl {
  namespace autotune {

    using ConfigListType = std::tuple<Config<0,1>,
                                      Config<64,1 >,Config<128,1 >,Config<256,1 >,Config<512,1 >,Config<1024,1 >,
                                      Config<64,2 >,Config<128,2 >,Config<256,2 >,Config<512,2 >,Config<1024,2 >,
                                      Config<64,4 >,Config<128,4 >,Config<256,4 >,Config<512,4 >,Config<1024,4 >,
                                      Config<64,8 >,Config<128,8 >,Config<256,8 >,Config<512,8 >,Config<1024,8 >,
                                      Config<64,16>,Config<128,16>,Config<256,16>,Config<512,16>,Config<1024,16>,
                                      Config<64,32>,Config<128,32>,Config<256,32>,Config<512,32>,Config<1024,32>,
                                      Config<64,64>,Config<128,64>,Config<256,64>,Config<512,64>,Config<1024,64>>;

    struct AutotuneContext {
      int static constexpr tests_per_config = 3;
      int static constexpr total_tests      = tests_per_config*std::tuple_size_v<ConfigListType>;
      int tests_performed;
      int best_index;
      std::array<double,std::tuple_size_v<ConfigListType>> timings;
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
      if constexpr (I < std::tuple_size_v<ConfigListType>) {
        if (index == I) { yakl::parallel_for( str , bounds , f , std::get<I>(ConfigListType{}) ); }
        else            { launch_config( index , str , bounds , f , CurrInd<I+1>{} ); }
      }
    }



    template <class F, int N, bool simple, class Style>
    inline void parallel_for( std::string                    str    ,
                              Bounds<N,Style,simple> const & bounds ,
                              F                      const & f      ) {
      auto lab = str+std::string(":")+std::to_string(bounds.nIter)+std::string("_iterations");
      auto time_and_visit = [&] (int index , AutotuneContext & context) {
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
        context.timings[index] = std::min( context.timings[index] , (double)time_loc );
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
      parallel_for( str , Bounds<1,CStyle,true>(bnd) , f );
    }



    template <class F, int N, bool simple>
    inline void parallel_for_F( std::string str , Bounds<N,FStyle,simple> const & bounds , F const & f ) {
      parallel_for<F,N,simple,FStyle>( str , bounds , f );
    }

    template <class F>
    inline void parallel_for_F( std::string str , std::integral auto bnd , F const & f ) {
      parallel_for<F,1,true,FStyle>( str , Bounds<1,FStyle,true>(bnd) , f );
    }



    template <int I=0>
    inline std::pair<int,int> get_config(int index) {
      if constexpr (I < std::tuple_size_v<ConfigListType>) {
        if (index == I) { return std::make_pair(std::tuple_element_t<I,ConfigListType>::Thr,
                                                std::tuple_element_t<I,ConfigListType>::Str); }
        else            { return get_config<I+1>(index); }
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
          auto config = get_config(c.best_index);
          if (myrank == 0) std::cout << key << " : Config<" << std::get<0>(config) << "," << std::get<1>(config)
                                            << "> , Speedup: " << c.timings[0]/c.timings[c.best_index] << std::endl;
        }
      }
    }

  }
}


