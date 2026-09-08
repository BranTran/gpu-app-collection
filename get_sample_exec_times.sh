#!/bin/bash
set -euo pipefail

# --- Configuration & Inputs ---
BIN_DIR="${1:-}"
OUT_DIR="${2:-./exec_time_outputs}"
NARGS=100000
TIMEOUT_SEC=30        # Maximum allowed runtime per binary (seconds)
MIN_TIME_MS=1         # Minimum runtime warning threshold (milliseconds)

# --- 1. Validation & Environment Checks ---
if [[ -z "$BIN_DIR" ]]; then
    echo "Error: Missing binary directory argument." >&2
    echo "Usage: $0 <BIN_DIR> [OUT_DIR]" >&2
    exit 1
fi

if [[ ! -d "$BIN_DIR" ]]; then
    echo "Error: Directory '$BIN_DIR' does not exist." >&2
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Check if 'timeout' utility is present on the system
HAS_TIMEOUT=false
if command -v timeout &> /dev/null; then
    HAS_TIMEOUT=true
else
    echo "Warning: 'timeout' command not found. Executables will run without a execution limit." >&2
fi

echo "================================================================="
echo " Starting Executable Profiling Suite"
echo " Target Directory : $BIN_DIR"
echo " Output Directory : $OUT_DIR"
echo " NARGS            : $NARGS"
echo " Max Timeout      : ${TIMEOUT_SEC}s"
echo "================================================================="

found_executable=false

# --- 2. Main Execution Loop ---
for exe in "$BIN_DIR"/*; do
    # Skip if not executable or if it's a directory
    [[ -x "$exe" && ! -d "$exe" ]] || continue
    found_executable=true

    base_name=$(basename "$exe")
    output_file="${OUT_DIR}/${base_name}_${NARGS}.txt"

    echo -n "Running ${base_name}... "

    # High-resolution start time (nanoseconds)
    start_ns=$(date +%s%N 2>/dev/null || echo "0")

    # Run with timeout guardrail if available
    if $HAS_TIMEOUT; then
        timeout "${TIMEOUT_SEC}s" "$exe" "$NARGS" > "$output_file" 2>&1
        exit_code=$?
    else
        "$exe" "$NARGS" > "$output_file" 2>&1
        exit_code=$?
    fi

    # High-resolution end time (nanoseconds)
    end_ns=$(date +%s%N 2>/dev/null || echo "0")

    # --- 3. Status Handling & High-Precision Timing ---
    if [[ $exit_code -eq 124 ]]; then
        echo "TIMED OUT (Exceeded ${TIMEOUT_SEC}s limit)"
        echo -e "\n[WARNING]: Execution terminated by timeout (${TIMEOUT_SEC}s)." >> "$output_file"
    elif [[ $exit_code -ne 0 ]]; then
        echo "FAILED (Exit Code: $exit_code)"
    else
        # Calculate elapsed time in milliseconds if sub-second precision is supported
        if [[ "$start_ns" != "0" && "$end_ns" != "0" && "$end_ns" -ge "$start_ns" ]]; then
            elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

            if [[ $elapsed_ms -lt $MIN_TIME_MS ]]; then
                echo "DONE (${elapsed_ms} ms) -> [WARNING: Execution < ${MIN_TIME_MS}ms, verify output]"
            else
                echo "DONE (${elapsed_ms} ms)"
            fi
        else
            echo "DONE"
        fi
    fi
done

if ! $found_executable; then
    echo "Error: No executable files found in '$BIN_DIR'." >&2
    exit 1
fi

echo "================================================================="
echo " Profiling complete. All logs saved to '$OUT_DIR'."
echo "================================================================="
