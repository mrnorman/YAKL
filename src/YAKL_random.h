
#pragma once
// Included by YAKL.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  // Simple, low-overhead random numbers
  // https://burtleburtle.net/bob/rand/smallprng.html

  /** @brief Non-cryptographic pseudo-random number generator with a very small internal state.
    * 
    * Based on the algorithm here: https://burtleburtle.net/bob/rand/smallprng.html
    * The author wrote the following statement on the website (Accessed Aug 1 2022): 
    * > I wrote this PRNG. I place it in the public domain.
    * > Same goes for at least the implementation of all those tests linked to above.
    * 
    * **IMPORTANT**: When creating a Random object inside a `parallel_for` kernel, please ensure each
    *                thread's object has a **different and unique** initial seed.
    */
  class Random {
  protected:
    /** @private */
    typedef unsigned long long u8;
    /** @private */
    u8 static constexpr rot(u8 x, u8 k) { return (((x)<<(k))|((x)>>(64-(k)))); }
    /** @private */
    struct State { u8 a, b, c, d; };
    /** @private */
    State state;

  public:

    /** @brief Initializes a prng object with the seed 1368976481. Warm-up of 20 iterations. */
    YAKL_INLINE Random()                            { set_seed(1368976481L); } // I made up this number
    /** @brief Initializes a prng object with the specified seed. Warm-up of 20 iterations. */
    YAKL_INLINE Random(u8 seed)                     { set_seed(seed); }
    /** @brief Copies a Random object */
    YAKL_INLINE Random(Random const            &in) { this->state = in.state; }
    /** @brief Moves a Random object */
    YAKL_INLINE Random(Random                 &&in) { this->state = in.state; }
    /** @brief Copies a Random object */
    YAKL_INLINE Random &operator=(Random const &in) { this->state = in.state; return *this; }
    /** @brief Moves a Random object */
    YAKL_INLINE Random &operator=(Random      &&in) { this->state = in.state; return *this; }

    /** @brief Assigns a seed. Warm-up of 20 iterations. */
    YAKL_INLINE void set_seed(u8 seed) {
      state.a = 0xf1ea5eed;  state.b = seed;  state.c = seed;  state.d = seed;
      for (int i=0; i<20; ++i) { gen(); }
    }

    /** @brief Generates a random unsigned integer between zero and `std::numeric_limits<u8>::max() - 1` */
    YAKL_INLINE u8 gen() {
      u8 e    = state.a - rot(state.b, 7);
      state.a = state.b ^ rot(state.c,13);
      state.b = state.c + rot(state.d,37);
      state.c = state.d + e;
      state.d = e       + state.a;
      return state.d;
    }

    /** @brief Generates a random floating point value between `0` and `1`
      * @param T The type of the floating point number */
    template <class T> YAKL_INLINE T genFP() {
      return static_cast<T>(gen()) / static_cast<T>(std::numeric_limits<u8>::max());
    }

    /** @brief Generates a random floating point value between `lb` and `ub`
      * @param T  The type of the floating point number
      * @param lb Lower bound of the random number
      * @param ub Upper bound of the random number*/
    template <class T> YAKL_INLINE T genFP(T lb, T ub) {
      return genFP<T>() * (ub-lb) + lb;
    }

  };

}
__YAKL_NAMESPACE_WRAPPER_END__


