#!/bin/bash

# =====================================================
# Script to run edisonn simulation using an input file
# and capture output in a log
# =====================================================

# Start MKL Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Usage:
# ./run_edisonn.sh <inputFile>

# Check for input file argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <inputFile>"
    exit 1
fi

inputFile=$1

# Get folder of input file and change directory there
inputDir=$(dirname "$inputFile")
inputName=$(basename "$inputFile")
cd "$inputDir" || { echo "[ERROR] Could not enter folder $inputDir"; exit 1; }

# Optional: parse simName and meshType from input file
simName=$(grep "simName" "$inputName" | awk '{print $3}')
meshType=$(grep "meshType" "$inputName" | awk '{print $3}')
version=$(grep "version" "$inputName" | awk '{print $3}')

simType="${simName}_${meshType}-"
logfile="${simType}${version}_simulation_log.txt"

# Echo starting info
echo "[INFO] Running edisonn with input file: $inputName"
echo "[INFO] simName: $simName, meshType: $meshType, version: $version"

# Run the executable with input file and redirect output
./edisonn "$inputName" > full_run_log.txt 2>&1

# Check exit status
if [ $? -eq 0 ]; then
    echo "[INFO] Simulation completed successfully."
    if [ -f "$logfile" ]; then
        echo "[INFO] Simulation log generated: $logfile"
        grep "Total computation time" full_run_log.txt
    else
        echo "[WARNING] Simulation completed but log file not found."
    fi
else
    echo "[ERROR] Simulation encountered an error. Check full_run_log.txt for details."
fi
