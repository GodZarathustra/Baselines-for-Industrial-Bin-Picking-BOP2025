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
object_dataset = "ipd"
detection_time_per_image = {}
run_time_per_image = defaultdict(float)
total_run_time = defaultdict(float)

# BOP 19 format
lines = ["scene_id,im_id,obj_id,score,R,t,time"]
# Load the estimated poses from the json file
results_tpath = "debug/{object_dataset}_sam6d/{scene_id:06d}/{object_dataset}_res.json"

scene_lids = os.listdir(f"debug/{object_dataset}_sam6d")
for scene_lid in scene_lids: 
    results_path = results_tpath.format(object_dataset=object_dataset, scene_id=int(scene_lid))
    if not os.path.exists(results_path):
        print(f"Skipping {results_path}")
        continue
    estimated_poses = load_json(results_path)
    for estimated_pose_data in estimated_poses:
        scene_id = estimated_pose_data["scene_id"]
        img_id = estimated_pose_data["img_id"]
        obj_id = estimated_pose_data["obj_id"]
        score = estimated_pose_data["score"]

        R = estimated_pose_data["R"]
        t = estimated_pose_data["t"]
        time = estimated_pose_data["time"]

        lines.append(
            "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                scene_id=scene_id,
                im_id=img_id,
                obj_id=obj_id,
                score=score,
                R=" ".join(map(str, np.array(R).flatten().tolist())),
                t=" ".join(map(str, np.array(t).flatten().tolist())),
                time=time,
            )
        )

bop_path = os.path.join(f"coarse_{object_dataset}-estimated-poses.csv")
with open(bop_path, "wb") as f:
    f.write("\n".join(lines).encode("utf-8"))
