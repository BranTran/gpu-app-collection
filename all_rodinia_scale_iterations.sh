#!/bin/bash

usage() {
    echo "Usage: $0 [-k]"
    exit 1
}

flag_set=false

while getopts ":k" opt; do
    case $opt in
        k)
            flag_set=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

# List of iterations
ITERS=(
    100
    1000
    5000000
    10000000
    15000000
    30000000
)

# Loop through each iteration and run the command
for i in "${ITERS[@]}"; do
    if $flag_set; then
        ./rodinia_scale_iterations.sh -k $i
    else
        ./rodinia_scale_iterations.sh $i
    fi
done
