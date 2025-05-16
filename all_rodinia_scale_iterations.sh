#!/bin/bash

# List of iterations
ITERS=(
    1000
    10000
    100000
    500000
    10000000
    15000000
)

# Loop through each iteration and run the command
for i in "${ITERS[@]}"; do
    ./rodinia_scale_iterations.sh $i
done


