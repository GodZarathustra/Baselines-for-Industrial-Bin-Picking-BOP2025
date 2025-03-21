xyzibd
ipd

Change 
    in bop.yaml
    max_num_scenes=100 or more in bop_pbr.py
    in utils
        add 
            ipd_object_ids = np.array(
                        [0,1,4,8,10,11,14,18,19,20]
                    )  
            xyzibd_object_ids = np.array(
                [
                    1,
                    2,
                    4,
                    5,
                    6,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17
                ]
            )  # object ID of occlusionLINEMOD is different
        correct as instead of lmo
            "category_id": self.object_ids + 1
                if dataset_name != "xyzibd"

    any error with bop_pbr as out of index- make sure train_pbr only 000 folder and can remove the train_pbr.csv and do it again
python -m src.scripts.render_template_with_pyrender

export DATASET_NAME=xyzibd
HYDRA_FULL_ERROR=1 python run_inference.py dataset_name=$DATASET_NAME

HYDRA_FULL_ERROR=1 python run_inference.py dataset_name=$DATASET_NAME model=cnos_fast