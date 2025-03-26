### Create conda environment
conda env create -f environment.yaml
conda activate sam6d

### Install pointnet2
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
cd ../../../

### Download ISM pretrained model
cd Instance_Segmentation_Model
# which python
# sudo /path/to/your/python download_sam.py
# python download_sam.py
wget -O ~/Downloads/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth --no-check-certificate
mv '/home/xyz/Downloads/sam_vit_h_4b8939.pth' '/media/xyz/Extreme Pro/Industry_BOP/Benchmark/SAM-6D/SAM-6D/Instance_Segmentation_Model/checkpoints/segment-anything'
# python download_fastsam.py
mkdir -p checkpoints/FastSAM && gdown --no-cookies --no-check-certificate -O 'checkpoints/FastSAM/FastSAM-x.pt' 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv

# python download_dinov2.py
wget -O ~/Downloads/dinov2_vitl14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth --no-check-certificate
mv '/home/xyz/Downloads/dinov2_vitl14_pretrain.pth' '/media/xyz/Extreme Pro/Industry_BOP/Benchmark/SAM-6D/SAM-6D/Instance_Segmentation_Model/checkpoints/dinov2'
cd ../

### Download PEM pretrained model
cd Pose_Estimation_Model
mkdir -p checkpoints && gdown --no-cookies --no-check-certificate -O 'checkpoints/sam-6d-pem-base.pth' 1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7
# python download_sam6d-pem.py
_ext_src