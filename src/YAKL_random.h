#pragma once
// Included by YAKL.h

namespace yakl {

  /** @brief Allocation-free Philox4x32-10 pseudo-random number generator for host and device code.
    *
    * Philox is a counter-based generator. The seed selects an experiment, the stream ID selects a logical
    * consumer within that experiment, and successive calls advance within that stream. Different streams do not
    * share mutable state, so creating one Random object per kernel iteration requires no allocation, locking, or
    * atomics. This independent implementation uses the Philox4x32-10 constants and permutation specified by
    * Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3," SC11.
    *
    * This generator is intended for simulation, not cryptography.
    */
  class Random {
  protected:
    using u4 = uint32_t;
    using u8 = uint64_t;

    u4 static constexpr multiplier0 = 0xd2511f53U;
    u4 static constexpr multiplier1 = 0xcd9e8d57U;
    u4 static constexpr weyl0       = 0x9e3779b9U;
    u4 static constexpr weyl1       = 0xbb67ae85U;

    struct State {
      u4 key0, key1;
      u8 stream, block;
      u8 output0, output1;
      int next;
    };

    State state;

    KOKKOS_INLINE_FUNCTION void generate_block() {
      u4 counter0 = static_cast<u4>(state.block);
      u4 counter1 = static_cast<u4>(state.block >> 32);
      u4 counter2 = static_cast<u4>(state.stream);
      u4 counter3 = static_cast<u4>(state.stream >> 32);
      u4 key0 = state.key0;
      u4 key1 = state.key1;

      for (int round = 0; round < 10; round++) {
        u8 const product0 = static_cast<u8>(multiplier0) * counter0;
        u8 const product1 = static_cast<u8>(multiplier1) * counter2;
        u4 const high0 = static_cast<u4>(product0 >> 32);
        u4 const high1 = static_cast<u4>(product1 >> 32);
        u4 const low0  = static_cast<u4>(product0);
        u4 const low1  = static_cast<u4>(product1);

        u4 const next0 = high1 ^ counter1 ^ key0;
        u4 const next1 = low1;
        u4 const next2 = high0 ^ counter3 ^ key1;
        u4 const next3 = low0;
        counter0 = next0;
        counter1 = next1;
        counter2 = next2;
        counter3 = next3;

        if (round != 9) {
          key0 += weyl0;
          key1 += weyl1;
        }
      }

      state.output0 = (static_cast<u8>(counter0) << 32) | counter1;
      state.output1 = (static_cast<u8>(counter2) << 32) | counter3;
      state.block++;
      state.next = 0;
    }

  public:
    /** @brief Creates a deterministic stream for the given experiment seed and logical stream ID. */
    KOKKOS_INLINE_FUNCTION Random(u8 seed, u8 stream_id) { set_seed(seed,stream_id); }
    KOKKOS_DEFAULTED_FUNCTION Random(Random const &) = default;
    KOKKOS_DEFAULTED_FUNCTION Random(Random &&) = default;
    KOKKOS_DEFAULTED_FUNCTION Random &operator=(Random const &) = default;
    KOKKOS_DEFAULTED_FUNCTION Random &operator=(Random &&) = default;

    /** @brief Restarts the generator at the beginning of a deterministic seed and stream pair. */
    KOKKOS_INLINE_FUNCTION void set_seed(u8 seed, u8 stream_id) {
      state.key0    = static_cast<u4>(seed);
      state.key1    = static_cast<u4>(seed >> 32);
      state.stream  = stream_id;
      state.block   = 0;
      state.output0 = 0;
      state.output1 = 0;
      state.next    = 2;
    }

    /** @brief Generates a uniformly distributed integer over the complete uint64_t range. */
    KOKKOS_INLINE_FUNCTION u8 gen() {
      if (state.next == 2) generate_block();
      if (state.next++ == 0) return state.output0;
      return state.output1;
    }

    /** @brief Generates a uniformly distributed floating-point value in [0,1). */
    template <class T> requires std::is_floating_point_v<T>
    KOKKOS_INLINE_FUNCTION T genFP() {
      int constexpr digits = std::numeric_limits<T>::digits;
      static_assert(digits <= 64,"Random::genFP supports floating-point types with at most 64 mantissa bits");
      if constexpr (digits < 64) {
        return static_cast<T>(gen() >> (64-digits)) / static_cast<T>(u8(1) << digits);
      } else {
        return static_cast<T>(gen()) * static_cast<T>(0x1p-64L);
      }
    }

    /** @brief Generates a uniformly distributed floating-point value in [lb,ub). */
    template <class T> requires std::is_floating_point_v<T>
    KOKKOS_INLINE_FUNCTION T genFP(T lb, T ub) {
      if constexpr (kokkos_debug) {
        if (ub < lb) Kokkos::abort("ERROR: Random::genFP upper bound is less than lower bound");
      }
      if (ub == lb) return lb;
      T const value = genFP<T>() * (ub-lb) + lb;
      return value < ub ? value : Kokkos::nextafter(ub,lb);
    }
  };

}
