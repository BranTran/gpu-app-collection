#!/bin/bash

# List of iterations
ITERS=(
    1000
    10000
    100000
    5000000
    10000000
    15000000
    30000000
)

# Loop through each iteration and run the command
for i in "${ITERS[@]}"; do
        ./scale_iterations.sh $i
        ./modify_backprop_k2.sh $i
done

