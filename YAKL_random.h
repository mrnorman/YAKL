
#pragma once

////////////////////////////////////////////////////////////////////////////////
// Simple, low-overhead random numbers (only a 64-bit internal state)
// 
// Adapted from:
// Thomas, D. B. "The MWC64X Random Number Generator.(2011)."
// URL: http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
// html (2011).
////////////////////////////////////////////////////////////////////////////////

class Random {
public:
  uint64_t state;

  // No seed (1.5 billion-th prime number)
  YAKL_INLINE Random() {
    state = (uint64_t) 3121238909U;
  }
  // Seed, default warmup
  YAKL_INLINE Random(uint64_t state) {
    this->state = state;
    warmup();
  }
  // Seed, custom warmup
  YAKL_INLINE Random(uint64_t state, int ncycles) {
    this->state = state;
    warmup(ncycles);
  }

  // Warmup (probably not needed, but whatever)
  YAKL_INLINE void warmup(int ncycles=10) {
    for (int i=0; i<ncycles; i++) { gen(); }
  }

  // Return a random unsigned 32-bit integer
  YAKL_INLINE uint32_t gen() {
      uint32_t c = state>>32;
      uint32_t x = state&0xFFFFFFFF;
      state = x*((uint64_t)4294883355U) + c;
      return x^c;
  }

  // Return floating point value: domain \in (0,1]
  template <class T> YAKL_INLINE T genFP() {
    return ( (T) gen() + 1 ) / ( (T) 4294967295U );
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


