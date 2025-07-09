#!/bin/bash

BIN_DIR="/scratch/08944/brantran/wattchmen-sc25-artifact/gpu-app-collection/bin/12.0/v100_release"
NARGS=100000
NAMESPACE="occupancy"

for exe in "$BIN_DIR"/*; do
  # Skip if not executable
  [[ -x "$exe" && ! -d "$exe" ]] || continue

  # Get basename (e.g., BAR)
  base_name=$(basename "$exe")

  # Output file
  output_file="${base_name}_${NARGS}_${NAMESPACE}.csv"

  # Run ncu and filter quoted lines
#  ncu --section Occupancy --csv "$exe" "$NARGS" 2>/dev/null | grep '^"' > "$output_file"
ncu --section Occupancy --csv --log-file "${output_file}" "$exe" "$NARGS"

  echo "Wrote $output_file"
done

