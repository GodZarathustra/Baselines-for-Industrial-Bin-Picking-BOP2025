## Getting Started

### 1. Preparation
Please follow the [SAM6D](https://github.com/JiehongLin/SAM-6D) to set up the environment.

### 2. Evaluation on the IPD&XYZ dataset

#### Your data should follow the structure
```
SAM-6D
|---Data/
|   |---IPD/
|       |---models/
|       |---camera.json
|       |---ipd_mask_sam6d.json
|       |---test_targets_multiview_bop25.json
|       |---test/
|           |--000000/
|                |--gray
|                |--depth
|                |--scene_camera.json
|           |--000001/
|           |--...
|           |--0000014/
|   |---XYZ/
|       |---models/
|       |---camera.json
|       |---xyz_mask_sam6d.json
|       |---test_targets_multiview_bop25.json
|       |---test/
|           |--000001/
|               |--gray
|               |--depth
|               |--scene_camera.json
|           |--000002/
|           |--...
|           |--0000074/
```

#### Run the template render
```
python Render/render_templates.py --dataset_name IPD
```

```
python Render/render_templates.py --dataset_name XYZ
```

#### Run the pose estimation model
```
python Pose_Estimation_Model/run_inference_detections.py --dataset_name IPD --output_dir Data/IPD/sam6d_outputs --input_dir Data/IPD/test --template_dir Data/IPD/templates --cad_dir Data/IPD/models --detection_path Data/IPD/ipd_mask_sam6d.json --det_score_thresh 0.4
```

```
python Pose_Estimation_Model/run_inference_detections.py --dataset_name XYZ --output_dir Data/XYZ/sam6d_outputs --input_dir Data/XYZ/test --template_dir Data/XYZ/templates --cad_dir Data/XYZ/models --detection_path Data/XYZ/xyz_mask_sam6d.json --det_score_thresh 0.4
```