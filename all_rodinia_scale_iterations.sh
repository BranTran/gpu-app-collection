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

# Check if no arguments were provided
if [ "$OPTIND" -eq 1 ]; then
    echo "No arguments were provided. Please use -k if you want one kernel rodinia"
    read -p "Do you want to proceed with compiling many kernel rodinia? (y/n): " confirm
    case $confirm in
        [Yy]* ) 
            echo "Proceeding with compiling many kernel rodinia"
            ;;
        [Nn]* ) 
            echo "Exiting."
            exit 1
            ;;
        * ) 
            echo "Please answer yes or no."
            exit 1
            ;;
    esac
fi

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

