
#pragma once


  // Unpack 2D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int &i1, int &i2) {
    i1 = (iGlob/(n2))     ;
    i2 = (iGlob     ) % n2;
  }
  // Unpack 3D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int &i1, int &i2, int &i3) {
    i1 = (iGlob/(n3*n2))     ;
    i2 = (iGlob/(n3   )) % n2;
    i3 = (iGlob        ) % n3;
  }
  // Unpack 4D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int &i1, int &i2, int &i3, int &i4) {
    i1 = (iGlob/(n4*n3*n2))     ;
    i2 = (iGlob/(n4*n3   )) % n2;
    i3 = (iGlob/(n4      )) % n3;
    i4 = (iGlob           ) % n4;
  }
  // Unpack 5D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int &i1, int &i2, int &i3, int &i4, int &i5) {
    i1 = (iGlob/(n5*n4*n3*n2))     ;
    i2 = (iGlob/(n5*n4*n3   )) % n2;
    i3 = (iGlob/(n5*n4      )) % n3;
    i4 = (iGlob/(n5         )) % n4;
    i5 = (iGlob              ) % n5;
  }
  // Unpack 6D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6) {
    i1 = (iGlob/(n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n6*n5*n4      )) % n3;
    i4 = (iGlob/(n6*n5         )) % n4;
    i5 = (iGlob/(n6            )) % n5;
    i6 = (iGlob                 ) % n6;
  }
  // Unpack 7D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int n7, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7) {
    i1 = (iGlob/(n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n7*n6*n5         )) % n4;
    i5 = (iGlob/(n7*n6            )) % n5;
    i6 = (iGlob/(n7               )) % n6;
    i7 = (iGlob                    ) % n7;
  }
  // Unpack 8D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7, int &i8) {
    i1 = (iGlob/(n8*n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n8*n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n8*n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n8*n7*n6*n5         )) % n4;
    i5 = (iGlob/(n8*n7*n6            )) % n5;
    i6 = (iGlob/(n8*n7               )) % n6;
    i7 = (iGlob/(n8                  )) % n7;
    i8 = (iGlob                       ) % n8;
  }
