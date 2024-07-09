#!/bin/bash

usage() {
    echo "Usage: $0 [-k] <iterations>"
    exit 1
}

flag_set=false

while getopts ":k" opt; do
    case $opt in
        k)
            flag_set=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

shift $((OPTIND -1))

if [ "$#" -ne 1 ]; then
    usage
fi


if $flag_set; then
    echo "Flag is set. Setting up one large kernel version"
    path='accelwattch_validation_one_kernel/rodinia-3.1'
    make_target='_one_kernel'
    onek='_one_kernel'
else
    echo "Flag is not set. Setting up many kernel call version"
    path='rodinia/3.1'
    make_target='hw_power'
    onek=''
fi

ITERS=${1}
#change the UINT64 to what you entered
grep "UINT64_MAX" -rl "src/cuda/${path}/cuda/" | xargs sed -i "s/UINT64_MAX/${ITERS}/g"

#Recompile
make rodinia-3.1${make_target} -C src

#Rename the benchmarks
mv $BINDIR/release/backprop_k1${onek}   $BINDIR/release/backprop_k1${onek}_${ITERS}iter
mv $BINDIR/release/backprop_k2${onek}   $BINDIR/release/backprop_k2${onek}_${ITERS}iter
mv $BINDIR/release/btree_k2${onek}      $BINDIR/release/btree_k2${onek}_${ITERS}iter
mv $BINDIR/release/btree_k1${onek}      $BINDIR/release/btree_k1${onek}_${ITERS}iter
mv $BINDIR/release/hotspot_k1${onek}    $BINDIR/release/hotspot_k1${onek}_${ITERS}iter
mv $BINDIR/release/kmeans_k1${onek}     $BINDIR/release/kmeans_k1${onek}_${ITERS}iter
mv $BINDIR/release/pathfinder_k1${onek} $BINDIR/release/pathfinder_k1${onek}_${ITERS}iter
mv $BINDIR/release/srad_v1_k1${onek}    $BINDIR/release/srad_v1_k1${onek}_${ITERS}iter
 
#Generate the config file
# sed "s|REPLACEITER|${ITERS}|g" "config_rodinia_kernel_template.cfg" > "../flop_counting/config_rodinia_kernel_${ITERS}.cfg"

#Restore the benchmarks back again
git restore "src/cuda/rodinia/${path}/cuda/*"
