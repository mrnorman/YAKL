
#pragma once

// Simple, low-overhead random numbers
// https://burtleburtle.net/bob/rand/smallprng.html

class Random {
public:
  #define rot(x,k) (((x)<<(k))|((x)>>(64-(k))))

  typedef unsigned long long u8;

  struct State { u8 a, b, c, d; };

  State state;

  YAKL_INLINE Random()                            { set_seed(1368976481L); } // I made up this number
  YAKL_INLINE Random(u8 seed)                     { set_seed(seed); }
  YAKL_INLINE Random(Random const            &in) { this->state = in.state; }
  YAKL_INLINE Random(Random                 &&in) { this->state = in.state; }
  YAKL_INLINE Random &operator=(Random const &in) { this->state = in.state; return *this; }
  YAKL_INLINE Random &operator=(Random      &&in) { this->state = in.state; return *this; }

  // Warmup (probably not needed, but whatever)
  YAKL_INLINE void set_seed(u8 seed) {
    state.a = 0xf1ea5eed;  state.b = seed;  state.c = seed;  state.d = seed;
    for (int i=0; i<20; ++i) { gen(); }
  }

  // Return a random unsigned 64-bit integer
  YAKL_INLINE u8 gen() {
    u8 e    = state.a - rot(state.b, 7);
    state.a = state.b ^ rot(state.c,13);
    state.b = state.c + rot(state.d,37);
    state.c = state.d + e;
    state.d = e       + state.a;
    return state.d;
  }

  // Return floating point value: domain \in [0,1]
  template <class T> YAKL_INLINE T genFP() {
    return static_cast<T>(gen()) / static_cast<T>(std::numeric_limits<u8>::max());
  }

  // Return a floating point value with custom bounds
  template <class T> YAKL_INLINE T genFP(T lb, T ub) {
    return genFP<T>() * (ub-lb) + lb;
  }

};

