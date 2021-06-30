
#pragma once 

#include <iostream>
#include <iomanip>
#include <time.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include "YAKL_Gator.h"
#include "stdlib.h"

#ifdef YAKL_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif

#include "YAKL_defines.h"

#ifdef _OPENMP45
#include <omp.h>
#endif

#ifdef YAKL_ARCH_OPENMP45
#include <omp.h>
#endif

#ifdef _OPENACC
#include "openacc.h"
#endif

#include "YAKL_error.h"
#include "YAKL_alloc_free.h"

