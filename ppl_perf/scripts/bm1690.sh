#!/bin/bash
set -e
{
    PERF_DIR=$PWD/profiling_bm1690
    mkdir -p $PERF_DIR
    for DIR in */; do
        if [ -f "./$DIR/autotune_test" ]; then
            echo "LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH PPL_KERNEL_PATH=$PWD/$DIR/lib/libkernel.so $PWD/$DIR/autotune_test $DIR/test_case $PERF_DIR "
            LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH PPL_KERNEL_PATH=$PWD/$DIR/lib/libkernel.so ./$DIR/autotune_test $PWD/$DIR/test_case $PERF_DIR
        fi
    done
    tar -czf ${PERF_DIR}.tar.gz ./profiling_bm1690
    mv ${PERF_DIR}.tar.gz $PWD/../
}