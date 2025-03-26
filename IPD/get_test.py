import os
import os.path as osp
import shutil

ipd_test_base = '/media/jiaqi/Extreme SSD/ipd/test' 
scene_folders = sorted(os.listdir(ipd_test_base))
output_base = '/media/jiaqi/Extreme SSD/ipd_test'
os.makedirs(output_base, exist_ok=True)

target_item = ["depth_photoneo", "rgb_photoneo", "scene_camera_photoneo.json"]
for scene_folder in scene_folders:
    scene_folder_path = osp.join(ipd_test_base, scene_folder)
    scene_folder_des = osp.join(output_base, scene_folder)
    os.makedirs(scene_folder_des, exist_ok=True)

    for item in target_item:
        dest_path = osp.join(scene_folder_des, item)
        if item == "scene_camera_photoneo.json":
            dest_file = osp.join(scene_folder_des, "scene_camera.json")
            if not osp.exists(dest_file):
                shutil.copy2(osp.join(scene_folder_path, item), dest_file)
        elif item == "rgb_photoneo":
            dest_folder = osp.join(scene_folder_des, "gray")
            if not osp.exists(dest_folder):
                shutil.copytree(osp.join(scene_folder_path, item), dest_folder)
        elif item == "depth_photoneo":
            dest_folder = osp.join(scene_folder_des, "depth")
            if not osp.exists(dest_folder):
                shutil.copytree(osp.join(scene_folder_path, item), dest_folder)
        else:
            raise KeyError
    
    