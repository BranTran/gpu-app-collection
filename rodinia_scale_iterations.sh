#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <iterations>"
    exit 1
fi

ITERS=${1}
#change the UINT64 to what you entered
grep "UINT64_MAX" -rl "src/cuda/rodinia/3.1/cuda/" | xargs sed -i "s/UINT64_MAX/${ITERS}/g"

#Recompile
make rodinia-3.1_hw_power -C src -j8

#Rename the benchmarks
mv $BINDIR/release/backprop_k1 $BINDIR/release/backprop_k1_${ITERS}iter
mv $BINDIR/release/backprop_k2 $BINDIR/release/backprop_k2_${ITERS}iter
mv $BINDIR/release/btree_k2 $BINDIR/release/btree_k2_${ITERS}iter
mv $BINDIR/release/btree_k1 $BINDIR/release/btree_k1_${ITERS}iter
mv $BINDIR/release/hotspot_k1 $BINDIR/release/hotspot_k1_${ITERS}iter
mv $BINDIR/release/kmeans_k1 $BINDIR/release/kmeans_k1_${ITERS}iter
mv $BINDIR/release/pathfinder_k1 $BINDIR/release/pathfinder_k1_${ITERS}iter
mv $BINDIR/release/srad_v1_k1 $BINDIR/release/srad_v1_k1_${ITERS}iter
 
#Generate the config file
# sed "s|REPLACEITER|${ITERS}|g" "config_rodinia_kernel_template.cfg" > "../flop_counting/config_rodinia_kernel_${ITERS}.cfg"

#Restore the benchmarks back again
git restore "src/cuda/rodinia/3.1/cuda/*"
