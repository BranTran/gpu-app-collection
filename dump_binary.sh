#!/bin/bash

# Check if binary path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-binary>"
    exit 1
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
output_file="${binary_base}_dump"

# Run cuobjdump
rm -f "$output_file"
cuobjdump -sass "$binary_path" > "$output_file"

# Open the output in less
less "$output_file"

