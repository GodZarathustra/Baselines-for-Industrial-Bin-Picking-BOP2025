#!/usr/bin/env python3
import subprocess
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Templates render")
parser.add_argument("--dataset_name", nargs="?", help="Dataset name")
args = parser.parse_args()

models_info_path = f"Data/{args.dataset_name}/models/models_info.json"
with open(models_info_path, "r") as f:
    models_info = json.load(f)
# List of object IDs to process
object_ids = [int(item) for item in models_info.keys()]

# Blender path
blender_path = "/home/tum/blender-3.1.1-linux-x64"

for obj_id in object_ids:
    # Format paths with the object ID
    output_dir = f"Data/{args.dataset_name}/templates/obj_{obj_id:06d}"
    cad_path = f"Data/{args.dataset_name}/models/obj_{obj_id:06d}.ply"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build and run command
    cmd = [
        "blenderproc", "run",
        "--custom-blender-path", blender_path,
        "Render/render_custom_templates.py",
        "--output_dir", output_dir,
        "--cad_path", cad_path,
        "--colorize", "True"
    ]
    
    print(f"Processing object ID: {obj_id}")
    subprocess.run(cmd)

print("All objects processed!")

#---------------------------------------------------------
# # List of camera indices to process
# cam_indices = range(42)
# obj_id = 14 
# blender_path = "/home/tum/blender-3.1.1-linux-x64"

# for cam_idx in cam_indices:
#     # Format paths
#     output_dir = f"Data/IPD/templates/obj_{obj_id:06d}"
#     cad_path = f"Data/IPD/models/obj_{obj_id:06d}.ply"
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Build and run command
#     cmd = [
#         "blenderproc", "run",
#         "--custom-blender-path", blender_path,
#         "Render/render_templates_custom.py",
#         "--output_dir", output_dir,
#         "--cad_path", cad_path,
#         "--colorize", "True",
#         "--cam_idx", str(cam_idx)
#     ]
    
#     # Run a completely new process for each camera
#     subprocess.run(cmd)
#     print(f"Processed camera {cam_idx}/{len(cam_indices)}")
#---------------------------------------------------------