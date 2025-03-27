## Getting Started
### 1. Preparation
Please follow the [SAM6D](https://github.com/JiehongLin/SAM-6D) to set up the environment.
### 2. Evaluation on the IPD dataset

#### Your data should follow the structure:
```
SAM-6D
|---Data/
|   |---IPD/
|       |---models/
|       |---test/
|       |---camera.json
|       |---ipd_mask_sam6d.json
|       |---test_targets_multiview_bop25.json
```

#### Run the template render
```
python Render/render_templates.py
```

#### Run the pose estimation model
```
python Pose_Estimation_Model/run_inference_detections.py --output_dir Data/IPD/sam6d_outputs --input_dir Data/IPD/test --template_dir Data/IPD/templates --cad_dir Data/IPD/models --detection_path Data/IPD/ipd_mask_sam6d.json --det_score_thresh 0.4
```
