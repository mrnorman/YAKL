
#pragma once

#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>
#include "stdlib.h"
#include "YAKL.h"

#ifdef ARRAY_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif

namespace yakl {

template <class T, int rank, int myMem=memDefault, int myStyle=styleDefault> class Array;

#include "CArray.h"

#include "FArray.h"

#include "SArray.h"

#include "FSArray.h"

}

