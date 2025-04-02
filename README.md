# Baselines for [OpenCV Perception Challenge for Bin-picking](https://bpc.opencv.org/) and [BOP-Industrial Challenge 2025](https://bop.felk.cvut.cz/challenges/)

In this repo, we implemented the baselines for both the object segmentation and pose estimation tasks as a starting point for participants in the BOP Challenge 2025 as well as the OpenCV Bin-Picking Challenge.

The baselines are built upon recent state-of-the-art methods and repositories, including [FoundationPose](https://github.com/NVlabs/FoundationPose), [SAM6D](https://github.com/Kudo510/Sam6D) , and [CNOS](https://github.com/nv-nguyen/cnos). We are continuously updating the collection with new methods. These baselines are integrated with the new BOP Industrial Track datasets—IPD and XYZ-IBD, and have been evaluated using the official BOP evaluation system, achieving reasonable results on the [leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-detection-unseen-bop24/bop-industrial/).

Please check each **Branch** for the corresponding task and baseline method for the **IPD** and **XYZ-IBD** dataset. 


## Pose Estimation for Unseen Objects
The pose estimation baselines have integrated with both IPD and XYZ-IBD datasets. 
| Method |Branch|
|--------|------|
| [FoundationPose](https://github.com/NVlabs/FoundationPose)  | [Pose_FoundationPose](https://github.com/GodZarathustra/Baselines-for-Industrial-Bin-Picking-BOP2025/tree/Pose_FoundationPose)    |
| [SAM6D](https://github.com/Kudo510/Sam6D)                   | [Pose_SAM6D](https://github.com/GodZarathustra/Baselines-for-Industrial-Bin-Picking-BOP2025/tree/Pose_SAM6D)             |

## Detection/Segmentation for Unseen Objects
The detection/segmentation baselines are currently implemented separately for each dataset in different branches. We will merge them soon, but you can still check for the following branches. 
| Dataset  | Method |Branch|
|----------|--------|------|
| IPD      | [CNOS](https://github.com/nv-nguyen/cnos) | Seg_CNOS_IPD |
| IPD      | [SAM6D](https://github.com/Kudo510/Sam6D)  | Seg_SAM6D_IPD |
| XYZ-IBD  | [CNOS](https://github.com/nv-nguyen/cnos)  | Seg_CNOS_XYZ-IBD |
| XYZ-IBD  | [SAM6D](https://github.com/Kudo510/Sam6D)  | Seg_SAM6D_XYZ-IBD |


