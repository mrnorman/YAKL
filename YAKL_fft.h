
#pragma once


// Lower-level routine for FFTs
template<unsigned N, typename T=double> class DanielsonLanczos {
public:
  static YAKL_INLINE void constexpr apply(T* data) {
    DanielsonLanczos<N/2,T>::apply(data  );
    DanielsonLanczos<N/2,T>::apply(data+N);
 
    // The compiler should have enough information to compute the sine
    // and cosine at compile time. Use "nm objectfile.o | grep -i sin"
    // to check is sin or cos are linked in. If you see them in the 
    // output of nm, then this was *not* computed at compile time, and
    // it will probably be expensive during runtime
    T wtemp,tempr,tempi,wr,wi,wpr,wpi;
    wtemp = sin(M_PI/N);
    wpr = -2.0*wtemp*wtemp;
    wpi = -sin(2*M_PI/N);
    wr = 1.0;
    wi = 0.0;
    for (unsigned i=0; i<N; i+=2) {
      tempr = data[i+N]*wr - data[i+N+1]*wi;
      tempi = data[i+N]*wi + data[i+N+1]*wr;
      data[i+N  ] = data[i  ]-tempr;
      data[i+N+1] = data[i+1]-tempi;
      data[i  ] += tempr;
      data[i+1] += tempi;
 
      wtemp = wr;
      wr += wr*wpr - wi   *wpi;
      wi += wi*wpr + wtemp*wpi;
    }
  }
};
template<typename T> class DanielsonLanczos<1,T> {
public:
  static YAKL_INLINE void constexpr apply(T* data) { }
};


template <class T> YAKL_INLINE constexpr void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}


// Pre-processing for complex FFTs
template <class T> YAKL_INLINE void scramble(T *data , unsigned N ) {
  unsigned n = N<<1;
  unsigned j=1;
  for (unsigned i=1; i<n; i+=2) {
    if (j>i) {
      swap(data[j-1], data[i-1]);
      swap(data[j  ], data[i  ]);
    }
    unsigned m = N;
    while (m>=2 && j>m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  };
}


// Process complex FFT output to compute the FFTs of real data
template <unsigned I, unsigned N, class T> class ProcessRealFFT {
public:
  static YAKL_INLINE void constexpr process(T *data, T *tmp) {
    T xrp = (tmp[2*I  ] + tmp[2*(N-I)  ])*0.5;
    T xrm = (tmp[2*I  ] - tmp[2*(N-I)  ])*0.5;
    T xip = (tmp[2*I+1] + tmp[2*(N-I)+1])*0.5;
    T xim = (tmp[2*I+1] - tmp[2*(N-I)+1])*0.5;
    // These should be computed at compile time
    data[2*I  ] = ( xrp + cos(M_PI*I/N)*xip - sin(M_PI*I/N)*xrm )/(2*N);
    data[2*I+1] = ( xim - sin(M_PI*I/N)*xip - cos(M_PI*I/N)*xrm )/(2*N);
    ProcessRealFFT<I-1,N,T>::process(data,tmp);
  }
};
template <unsigned N, class T> class ProcessRealFFT<1,N,T> {
public:
  static YAKL_INLINE void constexpr process(T *data, T *tmp) {
    unsigned constexpr I = 1;
    T xrp = (tmp[2*I  ] + tmp[2*(N-I)  ])*0.5;
    T xrm = (tmp[2*I  ] - tmp[2*(N-I)  ])*0.5;
    T xip = (tmp[2*I+1] + tmp[2*(N-I)+1])*0.5;
    T xim = (tmp[2*I+1] - tmp[2*(N-I)+1])*0.5;
    // These should be computed at compile time
    data[2*I  ] = ( xrp + cos(M_PI*I/N)*xip - sin(M_PI*I/N)*xrm )/(2*N);
    data[2*I+1] = ( xim - sin(M_PI*I/N)*xip - cos(M_PI*I/N)*xrm )/(2*N);
  }
};


// Pre-process real FFTs for inverting to real data
template <unsigned I, unsigned N, class T> class ProcessRealInverseFFT {
public:
  static YAKL_INLINE void constexpr process(T *data, T *tmp) {
    T xrp = (data[2*I  ] + data[2*(N-I)  ]);
    T xrm = (data[2*I  ] - data[2*(N-I)  ]);
    T xip = (data[2*I+1] + data[2*(N-I)+1]);
    T xim = (data[2*I+1] - data[2*(N-I)+1]);
    // These should be computed at compile time
    tmp[2*I  ] = xrp - cos(M_PI*I/N)*xip - sin(M_PI*I/N)*xrm;
    tmp[2*I+1] = xim - sin(M_PI*I/N)*xip + cos(M_PI*I/N)*xrm;
    ProcessRealInverseFFT<I-1,N,T>::process(data,tmp);
  }
};
template <unsigned N, class T> class ProcessRealInverseFFT<0,N,T> {
public:
  static YAKL_INLINE void constexpr process(T *data, T *tmp) {
    unsigned constexpr I = 0;
    T xrp = (data[2*I  ] + data[2*(N-I)  ]);
    T xrm = (data[2*I  ] - data[2*(N-I)  ]);
    T xip = (data[2*I+1] + data[2*(N-I)+1]);
    T xim = (data[2*I+1] - data[2*(N-I)+1]);
    // These should be computed at compile time
    tmp[2*I  ] = xrp - cos(M_PI*I/N)*xip - sin(M_PI*I/N)*xrm;
    tmp[2*I+1] = xim - sin(M_PI*I/N)*xip + cos(M_PI*I/N)*xrm;
  }
};


// Calculate the next power of two for a given unsigned integer
YAKL_INLINE constexpr unsigned nextPowerOfTwo(unsigned n) {
  unsigned count = 0;  
  // If n is zero or n is a power of 2, then return it
  if (n && !(n & (n - 1))) { return n; }
  while( n != 0) {
    n >>= 1;  
    count += 1;  
  }
  return 1 << count; 
}


///////////////////////////////////////////////////////////////////////////////////////
// Class to compute Fast Fourier Transforms on real data with a length that matches a
// power of two (e.g., 4, 16, 32, 64, ...). FFTs are serializeda in the dimension they
// are performed. User must allocate temporary data themselves because it might need
// to be valid in GPU memory
//
// Example usage:
//     unsigned constexpr N = 64; // FFTs are restricted to powers of 2 for now
//     FFT<N,double> fft;
//     double data[N+2];
//     double tmp[N];
//     // Initialize data
//     fft.forward(data,tmp);
//     // Manipulate data in Fourier space
//     fft.inverse(data,tmp);
//
// forward(T *data, T *tmp):
//       data: 1-D array of type T with SIZE+2 elements allocated. It is expected to
//             contain SIZE real values on input; and it will contain SIZE/2+1 complex
//             Fourier modes with complex numbers represented as:
//             [real,imag , real,imag , ...]
//       tmp: Temporary storage, 1-D array of type T with SIZE elements allocated
//
//       Computes a forward transform of SIZE real values and stores it into SIZE/2+1
//       complex Fourier modes. The transform is performed in place. The 0th and
//       (SIZE/2)th modes have no imaginary component (i.e., data[1] and
//       data[2*SIZE+1] are both zero)
//
//       Consider this the equivalent of:
//       fft[k] = sum( data[m]*exp(-I*2*pi*k*m/SIZE) , m=0..SIZE-1 ) / SIZE
//
//       Note the scaling by SIZE in the forward transform rather than in the inverse
//       transform. Also, only SIZE/2 transforms need to be computed because the rest
//       can be computed by symmetry since this is using an entirely real signal
//
// inverse(T *data, T *tmp):
//       data: 1-D array of type T with SIZE+2 elements allocated. It is expected to
//             contain SIZE/2+1 complex Fourier modes consuming all SIZE+2 indices on
//             input; and it will contain SIZE real values on output
//       tmp: Temporary storage, 1-D array of type T with SIZE elements allocated
//
//       Computes an inverse transform of SIZE/2+1 complex Fourier modes into SIZE
//       real values. The transform is performed in place. 
//
//       Consider this the equivalent of:
//       data[k] = sum( fft[n]*exp(I*2*pi*k*m/SIZE) , m=0..SIZE-1 )
//
//       Note that the scaling is done in the forward transform so it isn't needed in
//       the inverse. 
///////////////////////////////////////////////////////////////////////////////////////
template<unsigned SIZE, typename T=double> class FFT {
  static unsigned constexpr N = nextPowerOfTwo(SIZE)/2;
  static_assert(SIZE-nextPowerOfTwo(SIZE) == 0,"ERROR: Running FFT with a non-power-of-two-size");
  YAKL_INLINE void forwardComplex(T* data) const {
    scramble(data,N);
    DanielsonLanczos<N,T>::apply(data);
  }
  YAKL_INLINE void inverseComplex(T* data) const {
    // Multiply complex components by -1
    for (unsigned i=0; i<2*N; i+=2) { data[i+1] = -data[i+1]; }
    forwardComplex(data);
    // Multiply complex components by -1
    for (unsigned i=0; i<2*N; i+=2) { data[i+1] = -data[i+1]; }
  }
public:
  YAKL_INLINE void forward(T *data, T *tmp) const {
    // Copy to temporary buffer
    for (unsigned i=0; i<2*N; i++) {
      tmp[i] = data[i];
    }
    // Compute FFT assuming complex #s are even,I*odd; even,I*odd
    forwardComplex(tmp);
    data[0    ] = (tmp[0] + tmp[1])/(2*N);
    data[1    ] = 0;
    data[2*N  ] = (tmp[0] - tmp[1])/(2*N);
    data[2*N+1] = 0;
    // Transform the FFT into the true FFT for the real sequence
    ProcessRealFFT<N-1,N,T>::process(data,tmp);
  }
  YAKL_INLINE void inverse(T* data, T *tmp) const {
    // Transform FFTs into something whose inverse reproduces the original real signal
    ProcessRealInverseFFT<N-1,N,T>::process(data,tmp);
    inverseComplex(tmp);
    for (unsigned i=0; i<2*N; i++) {
      data[i] = tmp[i];
    }
  }
};




