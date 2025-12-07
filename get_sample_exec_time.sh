#!/bin/bash

OUTPUT_DIR=${1}
BIN_DIR=$BINDIR/$BINSUBDIR #"/scratch/08944/brantran/wattchmen-sc25-artifact/gpu-app-collection/bin/12.0/v100_release"
NARGS=100000
NAMESPACE="occupancy"

for exe in "$BIN_DIR"/*; do
  # Skip if not executable
  [[ -x "$exe" && ! -d "$exe" ]] || continue

  # Get basename (e.g., BAR)
  base_name=$(basename "$exe")

  # Output file
  output_file="${base_name}_${NARGS}.txt"
  if [ -f $OUTPUT_DIR/$output_file ]; then
	  continue
  fi

  $exe "$NARGS" > "h100_new_ops_exec_times/$output_file"

  echo "Wrote $output_file"
done

