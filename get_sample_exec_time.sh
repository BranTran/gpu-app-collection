#!/bin/bash

OUTPUT_DIR=${1}
BIN_DIR=$BINDIR/$BINSUBDIR #"/scratch/08944/brantran/wattchmen-sc25-artifact/gpu-app-collection/bin/12.0/v100_release"
NARGS=100000
NAMESPACE="occupancy"

# wgmma takes <kernel_shape> <iterations> instead of the single <iterations>
# argument every other ubench accepts. Sample every valid m64nNk16 shape (N a
# multiple of 8, 8-256; see tensor_benchmarks/wgmma/examples/bt/generate_matmul_kernel.py),
# not just the subset already tuned in flop_counting/config_h100_wgmma_exec_times_tuned.cfg.
WGMMA_KERNELS="8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256"

for exe in "$BIN_DIR"/*; do
  # Skip if not executable
  [[ -x "$exe" && ! -d "$exe" ]] || continue

  # Get basename (e.g., BAR)
  base_name=$(basename "$exe")

  if [ "$base_name" == "wgmma" ]; then
    for kernel in $WGMMA_KERNELS; do
      output_file="${base_name}_${kernel}_${NARGS}.txt"
      if [ -f $OUTPUT_DIR/$output_file ]; then
        continue
      fi

      $exe "$kernel" "$NARGS" > "h100_new_ops_exec_times/$output_file"

      echo "Wrote $output_file"
    done
    continue
  fi

  # Output file
  output_file="${base_name}_${NARGS}.txt"
  if [ -f $OUTPUT_DIR/$output_file ]; then
	  continue
  fi

  $exe "$NARGS" > "h100_new_ops_exec_times/$output_file"

  echo "Wrote $output_file"
done

