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

FALLBACK_NARGS=1000

# --- 2. Main Execution Loop ---
for exe in "$BIN_DIR"/*; do
    # Skip if not executable or if it's a directory
    [[ -x "$exe" && ! -d "$exe" ]] || continue
    found_executable=true

    base_name=$(basename "$exe")
    current_nargs=$NARGS
    output_file="${OUT_DIR}/${base_name}_${current_nargs}.txt"

    # Full newline print ensures output is immediately flushed to the console
    echo "Running ${base_name} (NARGS=${current_nargs})..."

    start_ns=$(date +%s%N 2>/dev/null || echo "0")
    exit_code=0

    # Execute primary attempt (NARGS=100000)
    if $HAS_TIMEOUT; then
        timeout "${TIMEOUT_SEC}s" "$exe" "$current_nargs" > "$output_file" 2>&1 || exit_code=$?
    else
        "$exe" "$current_nargs" > "$output_file" 2>&1 || exit_code=$?
    fi
    end_ns=$(date +%s%N 2>/dev/null || echo "0")

    # --- Backoff Retry Logic (If timed out with code 124) ---
    if [[ $exit_code -eq 124 ]]; then
        echo "  └─ [TIMED OUT] Exceeded ${TIMEOUT_SEC}s at NARGS=${current_nargs}."
        echo "  └─ Retrying with backed-off NARGS=${FALLBACK_NARGS}..."
        
        # Annotate primary output file before switching
        echo -e "\n[WARNING]: Timed out at NARGS=${current_nargs} (${TIMEOUT_SEC}s limit)." >> "$output_file"

        # Switch parameters to fallback run
        current_nargs=$FALLBACK_NARGS
        output_file="${OUT_DIR}/${base_name}_${current_nargs}.txt"

        start_ns=$(date +%s%N 2>/dev/null || echo "0")
        exit_code=0

        if $HAS_TIMEOUT; then
            timeout "${TIMEOUT_SEC}s" "$exe" "$current_nargs" > "$output_file" 2>&1 || exit_code=$?
        else
            "$exe" "$current_nargs" > "$output_file" 2>&1 || exit_code=$?
        fi
        end_ns=$(date +%s%N 2>/dev/null || echo "0")
    fi

    # --- Final Status Reporting ---
    if [[ $exit_code -eq 124 ]]; then
        echo "  └─ [FAILED] Timed out again at fallback NARGS=${current_nargs}."
        echo -e "\n[WARNING]: Timed out at fallback NARGS=${current_nargs} (${TIMEOUT_SEC}s limit)." >> "$output_file"
    elif [[ $exit_code -ne 0 ]]; then
        echo "  └─ [FAILED] Exit code: $exit_code"
    else
        # Calculate wall time
        if [[ "$start_ns" != "0" && "$end_ns" != "0" && "$end_ns" -ge "$start_ns" ]]; then
            elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
            if [[ $elapsed_ms -lt $MIN_TIME_MS ]]; then
                echo "  └─ [DONE] NARGS=${current_nargs} (${elapsed_ms} ms) -> [WARNING: Execution < ${MIN_TIME_MS}ms]"
            else
                echo "  └─ [DONE] NARGS=${current_nargs} (${elapsed_ms} ms)"
            fi
        else
            echo "  └─ [DONE] NARGS=${current_nargs}"
        fi
    fi
    echo ""
done

if ! $found_executable; then
    echo "Error: No executable files found in '$BIN_DIR'." >&2
    exit 1
fi

echo "================================================================="
echo " Profiling complete. All logs saved to '$OUT_DIR'."
echo "================================================================="
