# Read the json files that are the output of the infer.py script.
import os
import json
from collections import defaultdict
import numpy as np

def load_json(path: str, keys_to_int: bool = False):
    """Loads the content of a JSON file.

    Args:
        path: The path to the input JSON file.
        keys_to_int: Whether to convert keys to integers.
    Returns:
        The loaded content (typically a dictionary).
    """

    def convert_keys_to_int(x):
        return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()}

    with open(path, "r") as f:
        if keys_to_int:
            return json.load(f, object_hook=convert_keys_to_int)
        else:
            return json.load(f)

# Load the estimated poses from the json file
object_dataset = "IPD"
output_dir = f"Data/{object_dataset}/sam6d_outputs"

detection_time_per_image = {}
run_time_per_image = defaultdict(float)
total_run_time = defaultdict(float)

scene_lids = sorted(os.listdir(output_dir))

# BOP 19 format
lines = ["scene_id,im_id,obj_id,score,R,t,time"]
for scene_lid in scene_lids:
    image_ids = sorted(os.listdir(os.path.join(output_dir, scene_lid)))
    for image_id in image_ids:
        # Load the estimated poses from the json file
        results_path = os.path.join(output_dir, str(scene_lid), str(image_id), "detection_pem.json")
        estimated_poses = load_json(results_path)

        for estimated_pose_data in estimated_poses:
            scene_id = estimated_pose_data["scene_id"]
            img_id = estimated_pose_data["im_id"]
            obj_id = estimated_pose_data["obj_id"]
            score = estimated_pose_data["score"]
            R = estimated_pose_data["R"]
            t = estimated_pose_data["t"]
            run_time = estimated_pose_data["time"]

            lines.append(
                "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                    scene_id=scene_id,
                    im_id=img_id,
                    obj_id=obj_id,
                    score=score,
                    R=" ".join(map(str, np.array(R).flatten().tolist())),
                    t=" ".join(map(str, np.array(t).flatten().tolist())),
                    time=run_time,
                )
            )

bop_path = os.path.join(output_dir, f"{object_dataset}-estimated-poses.csv")
with open(bop_path, "wb") as f:
    f.write("\n".join(lines).encode("utf-8"))
