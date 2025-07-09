#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 [--ptx|-p] [-o] <path-to-binary>"
    exit 1
}

# Default values
dump_mode="sass"
output_suffix="_dump"
use_output_dir=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ptx|-p)
            dump_mode="ptx"
            output_suffix="_ptx_dump"
            shift
            ;;
        -o|--output)
            use_output_dir=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            binary_path="$1"
            shift
            ;;
    esac
done

# Check if binary path is provided
if [ -z "$binary_path" ]; then
    usage
fi

# Check if the file exists
if [ ! -f "$binary_path" ]; then
    echo "Error: '$binary_path' does not exist."
    exit 1
fi

# Check if the file is executable
if [ ! -x "$binary_path" ]; then
    echo "Warning: '$binary_path' is not marked executable. Proceeding anyway."
fi

# Extract just the basename without the path and extension
binary_name="$(basename "$binary_path")"
binary_base="${binary_name%.*}"  # remove extension if any
output_file="${binary_base}${output_suffix}"

# Output handling
if [ "$use_output_dir" = true ]; then
    output_dir="SASS_PTX_${CUDA_VERSION:-unknown}"
    mkdir -p "$output_dir"
    output_path="$output_dir/$output_file"
else
    output_path="$output_file"
fi

# Run cuobjdump
if [ "$dump_mode" == "sass" ]; then
    cuobjdump -sass "$binary_path" > "$output_path"
else
    cuobjdump --dump-ptx "$binary_path" > "$output_path"
fi

echo "Finished with $output_path"

# View in less if not writing to output dir
if [ "$use_output_dir" = false ]; then
    less "$output_path"
fi

