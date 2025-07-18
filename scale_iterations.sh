#!/bin/bash

usage() {
    echo "Usage: $0 <iterations>"
    exit 1
}

# Check if the number of arguments is exactly 1
if [ "$#" -ne 1 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

path='accelwattch_validation_one_kernel'
onek='_one_kernel'

ITERS=${1}
#change the UINT64 to what you entered
grep "UINT64_MAX" -rl "src/cuda/${path}" | xargs sed -i "s/UINT64_MAX/${ITERS}/g"

#Recompile
make bt_val -C src

#Rename the benchmarks
# Loop over all files in $BINDIR/release that end with ${onek}
for file in $BINDIR/$BINSUBDIR/*${onek}; do
  # Extract the base name of the file (without the directory part)
  base_name=$(basename "$file")
  # Rename the file
  mv "$file" "$BINDIR/$BINSUBDIR/${base_name}_${ITERS}iter"
done
 

#Restore the benchmarks back again
git restore "src/cuda/${path}/"
