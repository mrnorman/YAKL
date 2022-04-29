
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int rank, int myStyle>
    inline Array<T,1,memHost,myStyle> pack( Array<T,rank,memHost,myStyle> const &arr ,
                                            Array<bool,rank,memHost,myStyle> const &mask = Array<bool,rank,memHost,myStyle>() ) {
      if (allocated(mask)) {
        if (mask.totElems() != arr.totElems()) {
          yakl_throw("Error: pack: arr and mask have a different number of elements");
        }
        // count the number of true elements
        int numTrue = count( mask );
        Array<T,1,memHost,myStyle> ret("packReturn",numTrue);
        int slot = 0;
        for (int i=0; i < arr.totElems(); i++) {
          if (mask.data()[i]) { ret.data()[slot] = arr.data()[i]; slot++; }
        }
        return ret;
      } else {
        Array<T,1,memHost,myStyle> ret("packReturn",arr.totElems());
        for (int i=0; i < arr.totElems(); i++) {
          ret.data()[i] = arr.data()[i];
        }
        return ret;
      }
    }

  }
}

