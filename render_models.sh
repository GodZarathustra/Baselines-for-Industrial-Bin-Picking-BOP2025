#!/bin/bash

# Base command
BLENDER_PATH="/home/xyz/blender-3.1.1-linux-x64"
SCRIPT_PATH="./Render/render_custom_templates.py"
MODELS_DIR="./Data/XYZ/models"  # Replace with your models directory
BASE_OUTPUT_DIR="./Data/XYZ/templates"               # Base output directory name

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

# Find all .ply files and store their paths
mapfile -t CAD_PATHS < <(find "$MODELS_DIR" -name "*.ply")

# Check if any .ply files were found
if [ ${#CAD_PATHS[@]} -eq 0 ]; then
    echo "Error: No .ply files found in $MODELS_DIR"
    exit 1
fi

# Create output directories based on model names and store their paths
OUTPUT_DIRS=()
for cad_path in "${CAD_PATHS[@]}"; do
    # Extract filename without extension and create output dir
    model_name=$(basename "$cad_path" .ply)
    output_dir="$BASE_OUTPUT_DIR/$model_name"
    mkdir -p "$output_dir"
    OUTPUT_DIRS+=("$output_dir")
done

# Loop through the arrays and run BlenderProc for each pair
for i in "${!CAD_PATHS[@]}"; do
    echo "Processing model: $(basename "${CAD_PATHS[i]}")"
    echo "Output directory: ${OUTPUT_DIRS[i]}"
    
    blenderproc run \
        --custom-blender-path "$BLENDER_PATH" \
        "$SCRIPT_PATH" \
        --output_dir "${OUTPUT_DIRS[i]}" \
        --cad_path "${CAD_PATHS[i]}" \
        --colorize True
    
    # Add a small delay between runs (optional)
    sleep 2
    
    echo "Completed $(($i + 1)) of ${#CAD_PATHS[@]} models"
    echo "----------------------------------------"
done

echo "All rendering jobs completed!"