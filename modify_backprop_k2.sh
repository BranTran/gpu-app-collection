#!/bin/bash

usage() {
    echo "Usage: $0 <iterations>"
    exit 1
}
path='accelwattch_validation_one_kernel/rodinia-3.1/cuda/backprop_k2'
onek='mod'

ITERS=${1}

#change the iterations and defines
grep "UINT64_MAX" -rl "src/cuda/${path}" | xargs sed -i "s/UINT64_MAX/${ITERS}/g"
grep "#define ETA" -rl "src/cuda/${path}" | xargs sed -i "s/ETA 0.3/ETA 0.3f/g"
grep "#define MOMENTUM" -rl "src/cuda/${path}" | xargs sed -i "s/MOMENTUM 0.3/MOMENTUM 0.3f/g"

#Recompile
make bt_backprop_mod -C src

#Rename the benchmarks
# Loop over all files in $BINDIR/release that end with ${onek}
for file in $BINDIR/release/*${onek}; do
  # Extract the base name of the file (without the directory part)
  base_name=$(basename "$file")
  # Rename the file
  mv "$file" "$BINDIR/release/${base_name}_${ITERS}iter"
done
 

#Restore the benchmarks back again
git restore "src/cuda/${path}/"
