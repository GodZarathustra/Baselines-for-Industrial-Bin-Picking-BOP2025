## Getting Started

### 1. Preparation
Please follow the [Foundationpose](https://github.com/NVlabs/FoundationPose) to set up the environment.

### 2. Evaluation on the IPD&XYZ dataset

#### Your data should follow the structure
```
FoundationPose
|---DATASET/
|   |---ipd/
|       |---models/
|       |---test/
|       |---camera.json
|       |---ipd_mask_sam6d.json
|       |---test_targets_multiview_bop25.json
|   |---xyz/
|       |---models/
|       |---test/
|       |---camera.json
|       |---xyz_mask_sam6d.json
|       |---test_targets_multiview_bop25.json
```

#### Run the pose estimation model
```
python inference.py --dataset_dir DATASET/ipd --dataset_name ipd --use_reconstructed_mesh 0 --mask_dir DATASET/ipd/ipd_mask_sam6d.json --debug 1 --debug_dir debug --test_targets_path DATASET/ipd/test_targets_multiview_bop25.json
```

```
python inference.py --dataset_dir DATASET/xyz --dataset_name xyz --use_reconstructed_mesh 0 --mask_dir DATASET/xyz/xyz_mask_sam6d.json --debug 1 --debug_dir debug --test_targets_path DATASET/xyz/test_targets_multiview_bop25.json
```