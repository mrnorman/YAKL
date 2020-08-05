
#pragma once

////////////////////////////////////////
// Simple, low-overhead random numbers
////////////////////////////////////////

class Random {
public:
  size_t m = 4294967296UL;
  size_t a = 1664525UL;
  size_t c = 1013904223UL;

  size_t state;

  // No seed (1.5 billion-th prime number)
  YAKL_INLINE Random()                            { this->state = (size_t) 3121238909UL;}
  YAKL_INLINE Random(size_t state)                { this->state = state*3121238909UL; warmup(2); }
  YAKL_INLINE Random(Random const            &in) { this->state = in.state; }
  YAKL_INLINE Random(Random                 &&in) { this->state = in.state; }
  YAKL_INLINE Random &operator=(Random const &in) { this->state = in.state; return *this; }
  YAKL_INLINE Random &operator=(Random      &&in) { this->state = in.state; return *this; }

  // Warmup (probably not needed, but whatever)
  YAKL_INLINE void warmup(int ncycles=10) {
    for (int i=0; i<ncycles; i++) { gen(); }
  }

  // Return a random unsigned 32-bit integer
  YAKL_INLINE uint32_t gen() {
    state = (a*state + c) % m;
    return state;
  }

  // Return floating point value: domain \in (0,1]
  template <class T> YAKL_INLINE T genFP() {
    return ( (T) gen() + 1 ) / ( (T) 4294967296UL );
  }

  // Return a floating point value with custom bounds
  template <class T> YAKL_INLINE T genFP(T lb, T ub) {
    return  genFP<T>() * (ub-lb) + lb;
  }

  // Return floating point value: domain \in (0,1]
  template <class T> YAKL_INLINE void fillArray(T *data, int n) {
    for (int i=0; i<n; i++) {
      data[i] = genFP<T>();
    }
  }

};


