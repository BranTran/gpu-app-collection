#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 [--ptx] <path-to-binary>"
    exit 1
}

# Parse optional flag
dump_mode="sass"
output_suffix="_dump"

if [ "$1" == "--ptx" ]; then
    dump_mode="ptx"
    output_suffix="_ptx_dump"
    shift
fi

# Check if binary path is provided
if [ -z "$1" ]; then
    usage
fi

binary_path="$1"

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

# Output file
output_file="${binary_base}${output_suffix}"

# Output dir

output_dir="SASS_PTX_${CUDA_VERSION}"

if [ ! -d $output_dir ]; then
	mkdir -p $output_dir
fi

# Run cuobjdump
#rm -f "$output_dir/$output_file"

if [ "$dump_mode" == "sass" ]; then
    cuobjdump -sass "$binary_path" > "$output_dir/$output_file"
else
    cuobjdump --dump-ptx "$binary_path" > "$output_dir/$output_file"
fi

echo Finished with $output_file

# Open the output in less
#less "$output_file"

