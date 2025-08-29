#!/bin/bash
BINDIR="$BINDIR/$BINSUBDIR" # Path to compiled benchmark binaries
CONFIG_FILE=${1}
#"/scratch/08944/brantran/wattchmen-sc25-artifact/gpu-app-collection/bin/12.0/v100_release"
while IFS= read -r line; do
  # Skip if not executable
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  bm_name="${line// /_}"
  # Get basename (e.g., BAR)

  # Output file
  output_file="${bm_name}.txt"
  $BINDIR/$line > $output_file

  echo "Wrote $output_file"
done < "$CONFIG_FILE"

