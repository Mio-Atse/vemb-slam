#!/bin/bash
# Parameter sweep script for vEMB-SLAM
# This script demonstrates how to run multiple experiments with different configurations

# Update calibration file path in config if needed
CALIBRATION_FILE="data/calibration/CalibrationMatrix_college_cpt.npz"

# Models to test
MODELS=("zoe" "midas" "metric3d")

echo "Starting parameter sweep..."
echo "============================"

# Test each model
for MODEL in "${MODELS[@]}"; do
    echo "Testing model: $MODEL"
    
    python scripts/run_slam.py \
        --config config/default_config.yaml \
        --model "$MODEL"
    
    echo "$MODEL completed."
    echo "----------------------------"
done

echo "Parameter sweep complete!"
