#!/bin/bash

./main-crayclang-debug.sh  
./main-crayclang-opt.sh  
./main-hipcc-debug.sh  
./main-hipcc-opt.sh
# Comment out below if Valgrind is not needed
# ./main-hipcc-opt-valgrind.sh
./main-hipcc-debug-valgrind.sh
