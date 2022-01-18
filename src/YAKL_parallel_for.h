
#pragma once

template <class T> constexpr T fastmod(T a , T b) {
  return a < b ? a : a-b*(a/b);
}

#include "YAKL_parallel_for_c.h"

#include "YAKL_parallel_for_fortran.h"



