#!/bin/bash
set -e
{
    _ROOT=$PWD
    PERF_DIR=$_ROOT/profiling_bm1684x
    mkdir -p $PERF_DIR
    for DIR in */; do
        if [ -f "./$DIR/test_case" ]; then
            mkdir -p $PERF_DIR/$DIR
            pushd $PERF_DIR/$DIR > /dev/null
            BMLIB_ENABLE_ALL_PROFILE=1 PPL_FILE_NAME=0 PPL_DATA_PATH=$_ROOT/$DIR/data LD_LIBRARY_PATH=$_ROOT:$LD_LIBRARY_PATH PPL_KERNEL_PATH=$_ROOT/$DIR/lib/libkernel.so $_ROOT/$DIR/test_case
            popd > /dev/null
        fi
    done
    tar -czf ${PERF_DIR}.tar.gz ./profiling_bm1684x
    mv ${PERF_DIR}.tar.gz $PWD/../
}