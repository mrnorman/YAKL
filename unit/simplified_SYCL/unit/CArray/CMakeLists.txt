cmake_minimum_required(VERSION 3.0)
project(yakl_unit)
enable_testing()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)

# YAKL library
add_subdirectory(.. ./yakl)

# Unit test directories
add_subdirectory(CArray)
