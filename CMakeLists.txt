
set(SRC BuddyAllocator.cpp YAKL.cpp gator_mod.F90)

set_source_files_properties(BuddyAllocator.cpp YAKL.cpp PROPERTIES LANGUAGE CUDA)

add_library(yakl STATIC ${SRC})


