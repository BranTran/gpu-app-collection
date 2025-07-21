#!/bin/bash

# Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
# Northwestern University, Purdue University, The University of British Columbia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer;
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution;
# 3. Neither the names of Northwestern University, Purdue University,
#    The University of British Columbia nor the names of their contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# --- SYNTHESIZED SCRIPT ---

# Usage: ./profile_frequency_and_ubench_power.sh <config_file>
# The config_file should list benchmarks and their arguments, one per line.
# Example config_file.txt:
# BE_DP_FP_ADD 20000000
# hotspot_k1 1024 2 2 /data/temp_1024 /data/power_1024 output.out
# ... (other benchmarks and their arguments)

# --- Configuration ---
sleep_time=5            # Seconds waiting between pinging for power
max_wait_iterations=10  # 5 minutes waiting in total for power to drop
threshold_power_diff=1  # Threshold above starting power allowed for idle, adjust as needed

# Define the array of frequencies to test
frequencies=(345 495 660 810 975 1125 1290 1440 1605 1755)

# Number of runs for each benchmark at each frequency
NUM_RUNS=3 
# Fixed GPU Device ID (as per your request to set DEVID=1)
FIXED_DEVID=1

# --- Pathing and Setup ---
SCRIPT_DIR="$GPUAPPS_ROOT/../flop_counting"
BINDIR="$BINDIR/$BINSUBDIR" # Path to compiled benchmark binaries
RODINIA_DATADIR="$GPUAPPS_ROOT/data_dirs/cuda/rodinia/3.1"
PARBOIL_DATADIR="$GPUAPPS_ROOT/data_dirs/parboil" # You might not need this if your benchmarks are Rodinia-based
OUT_BASE_DIR="$SCRIPT_DIR/h100_data" # Base output directory
PROFILER="$SCRIPT_DIR/profiler/dumpGpuPowerAllClocks" # Profiler binary

# Check if the profiler binary exists, compile if not
if [ ! -f "$PROFILER" ]; then
    echo "WARNING: Could not find profiler binary at $PROFILER, attempting to compile now."
    make -C "$SCRIPT_DIR/profiler"
    if [ ! -f "$PROFILER" ]; then
        echo "ERROR: Failed to compile profiler. Exiting."
        exit 1
    fi
    echo "Profiler compiled successfully."
fi

# Checking system info
HOSTNAME=$(hostname)
UUID_list=($(nvidia-smi -L | awk '{print $NF}' | tr -d '[)]')) # Get UUIDs of all GPUs

# --- Input Argument Check ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "The config_file should list benchmarks and their arguments, one per line."
    exit 1
fi

config_file="$1"

if [ ! -f "$config_file" ]; then
    echo "Error: Config file '$config_file' not found."
    exit 1
fi

## Main Execution Loop


# Loop over the frequencies first
for freq in "${frequencies[@]}"; do
    echo "--- Setting GPU Graphics Clock to ${freq} MHz ---"
    # Set the GPU clock frequency. This requires sudo.
    # Ensure your user has permissions for this or use `sudo -E` if environment variables are needed.
    sudo nvidia-smi -i "$FIXED_DEVID" -lgc "$freq","$freq"
    
    # Read the config file line by line for benchmarks
    while IFS= read -r line; do
        # Skip empty lines or lines starting with #
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        
        # Parse benchmark name and its arguments from the config line
        bm=$(echo "$line" | awk '{print $1}')
        args=$(echo "$line" | cut -d' ' -f2-) # All arguments after the benchmark name

        if [ -z "$args" ]; then
            bm_name="${bm// /_}" # Replace spaces with underscores for directory names
        else
            bm_name="${bm// /_}_${args// /_}"
        fi

        echo "--- Processing Benchmark: ${bm_name} at ${freq} MHz ---"

        bm_my_input="${bm//-/_}_r" # Replace hyphens in benchmark name for valid var name

        # Replace ./data with $RODINIA_DATADIR/$bm/data in the args if applicable
        # This handles Rodinia-specific data paths
        parsed_args="${args//.\/data/$RODINIA_DATADIR/$bm/data}"

        # Determine the actual benchmark execution command
        bm_exec_cmd=""
        # Check if the variable contains a hyphen, indicating a generic binary name
        if [[ "$bm" == *"-"* ]]; then
            bm_exec_cmd="$BINDIR/$bm $parsed_args"
        else
            # Check if a custom variable (like backprop_k1_r) exists and is not empty
            if [ -n "${!bm_my_input:-}" ]; then
                # Use the value of the custom variable, ensure args from config are used if not embedded
                bm_exec_cmd="${!bm_my_input}"
                
                case "$bm" in
                    "backprop_k1") bm_exec_cmd="$BINDIR/backprop_k1_${parsed_args}iter 65536" ;;
                    "backprop_k2") bm_exec_cmd="$BINDIR/backprop_k2_${parsed_args}iter 65536" ;;
                    "btree_k1") bm_exec_cmd="$BINDIR/btree_k1_${parsed_args}iter file $RODINIA_DATADIR/b+tree-rodinia-3.1/data/mil.txt command $RODINIA_DATADIR/b+tree-rodinia-3.1/data/command.txt" ;;
                    "btree_k2") bm_exec_cmd="$BINDIR/btree_k2_${parsed_args}iter file $RODINIA_DATADir/b+tree-rodinia-3.1/data/mil.txt command $RODINIA_DATADIR/b+tree-rodinia-3.1/data/command.txt" ;;
                    "hotspot_k1") bm_exec_cmd="$BINDIR/hotspot_k1_${parsed_args}iter 1024 2 2 $RODINIA_DATADIR/hotspot-rodinia-3.1/data/temp_1024 $RODINIA_DATADIR/hotspot-rodinia-3.1/data/power_1024 output.out" ;;
                    "kmeans_k1") bm_exec_cmd="$BINDIR/kmeans_k1_${parsed_args}iter -o -i $RODINIA_DATADIR/kmeans-rodinia-3.1/data/819200.txt" ;;
                    "pathfinder_k1") bm_exec_cmd="$BINDIR/pathfinder_k1_${parsed_args}iter 100000 100 20 " ;;
                    "srad_v1_k1") bm_exec_cmd="$BINDIR/srad_v1_k1_${parsed_args}iter 100 0.5 502 458" ;;
                    *) bm_exec_cmd="$BINDIR/$bm $parsed_args" ;; # Default if no specific mapping
                esac
            else
                bm_exec_cmd="$BINDIR/$bm $parsed_args" # Fallback if no specific var or custom logic
            fi
        fi

        # Make sure bm_exec_cmd is not empty
        if [ -z "$bm_exec_cmd" ]; then
            echo "ERROR: Could not construct execution command for benchmark '$bm'. Skipping."
            continue
        fi

        # Create output directories for the current frequency and benchmark
        mkdir -p "$OUT_BASE_DIR/ubench_profile_output/${freq}_${bm_name}/${HOSTNAME}"
        mkdir -p "$OUT_BASE_DIR/ubench_execval_output/${freq}_${bm_name}/${HOSTNAME}"
        
        # --- Loop for multiple runs ---
        for run in $(seq 0 $((NUM_RUNS - 1))); do
            DEVID="$FIXED_DEVID" # Use the fixed device ID
            GPU_UUID=${UUID_list[${DEVID}]} # Get UUID for the fixed device

            echo "Starting profiling of $bm_name on $HOSTNAME gpu${DEVID} (${GPU_UUID}) at ${freq}MHz, run ${run}"
            
            # Get initial idle power before launching profiler/benchmark
            # This helps in the adaptive sleeping later
            idle_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id="${DEVID}")

            # Start the power profiler in the background
            "$PROFILER" -r 100 -n -1 -d "${DEVID}" > "$OUT_BASE_DIR/ubench_profile_output/${freq}_${bm_name}/${HOSTNAME}/${HOSTNAME}_gpu${DEVID}_${GPU_UUID}_${bm_name}_${run}.txt" 2>&1 &
            pid_profiler=$!
            
            sleep 10 # Give profiler time to start

            # Start the benchmark in the background
            CUDA_VISIBLE_DEVICES="$DEVID" $bm_exec_cmd > "$OUT_BASE_DIR/ubench_execval_output/${freq}_${bm_name}/${HOSTNAME}/${HOSTNAME}_gpu${DEVID}_${GPU_UUID}_${bm_name}_${run}.txt" 2>&1 &
            pid_benchmark=$!

            # Wait for the benchmark to finish
            wait "$pid_benchmark"
            echo "Benchmark $bm_name (PID $pid_benchmark) finished. Sleeping before killing profiler."
            sleep 10 # Give benchmark time to fully complete and power draw to settle slightly

            # --- Kill Profiler Logic (Robust multiple attempts) ---
            kill -SIGINT "$pid_profiler" # Send graceful termination signal
            echo "Kill signal (SIGINT) sent to profiler PID $pid_profiler."
            sleep "$sleep_time" # Wait for profiler to respond
            
            # Check if profiler is still running and send SIGKILL if needed
            if ps -p "$pid_profiler" > /dev/null; then
                kill -SIGKILL "$pid_profiler"
                echo "Profiler PID $pid_profiler still active, sent SIGKILL."
                sleep "$sleep_time"
            fi

            # Final check via ps, useful if PID was reused or initial kill failed
            bg_pid=$(ps -ef | grep "$PROFILER" | grep -v grep | awk '{print $2}')
            if [ -n "$bg_pid" ]; then
                echo "Warning: Profiler still running according to ps, killing PID $bg_pid."
                kill -SIGKILL "$bg_pid"
                sleep "$sleep_time"
            fi
            echo "Profiling run ${run} on gpu${DEVID} for ${bm_name} at ${freq}MHz concluded."

            # --- Adaptive Sleeping Mechanism ---
            echo "Waiting for GPU power to return to idle levels..."
            wait_counter=0
            while true; do
                current_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id="${DEVID}")
                # Check if current power is still significantly above idle power
                if (( $(echo "$current_power > ($idle_power + $threshold_power_diff)" | bc -l) )); then
                    echo "  Power still high (${current_power}W > ${idle_power}W + ${threshold_power_diff}W). Sleeping for ${sleep_time}s..."
                    sleep "$sleep_time"
                    ((wait_counter++))
                else
                    echo "  Power returned to idle levels (${current_power}W). Continuing."
                    break # Power dropped, safe to proceed
                fi
                
                # Check if max wait iterations are reached (timeout)
                if [ "$wait_counter" -ge "$max_wait_iterations" ]; then
                    echo "  Max wait iterations (${max_wait_iterations}) reached. Force continuing."
                    break # Timeout, move on
                fi
            done
            echo "Finished adaptive sleep for run ${run}."
        done # End of run loop
    done < "$config_file" # End of benchmark loop
done # End of frequency loop

#Need to reset once we are done
sudo nvidia-smi -i "$FIXED_DEVID" -rgc
echo "--- All profiling complete ---"
