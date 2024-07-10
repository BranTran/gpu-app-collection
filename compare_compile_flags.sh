#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <dir_0> <dir_1>"
  exit 1
fi

# Define the directories
DIR_O0=${1}
DIR_O2=${2}

found_diff=false
# Iterate over each file in DIR_O0
for file in "$DIR_O0"/*; do
    filename=$(basename "$file")
    file_O2="$DIR_O2/$filename"

    # Check if the file exists in DIR_O2
    if [[ -e "$file_O2" ]]; then
        echo "Testing $filename"
        # Execute cuobjdump -sass for both files
        cuobjdump -sass "$file" > test_A
        cuobjdump -sass "$file_O2" > test_B
        
        # Take the diff between test_A and test_B
        if ! diff test_A test_B > /dev/null; then
            echo "FOUND DIFFERENCE! $filename"
            found_diff=true
        fi
        
        # Clean up temporary files
        rm -f test_A test_B
    fi
done

if $found_diff; then
    echo "We found a difference, check it out"
fi
