# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os,sys,kornia
import time
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../')
from learning.datasets.h5_dataset import *
from learning.models.score_network import *
from learning.datasets.pose_dataset import *
from Utils import *
from datareader import *


def vis_batch_data_scores(pose_data, ids, scores, pad_margin=5):
  assert len(scores)==len(ids)
  canvas = []
  for id in ids:
    rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
    rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
    H,W = rgbA_vis.shape[:2]
    zmin = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).min()
    zmax = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).max()
    depthA_vis = depth_to_vis(pose_data.depthAs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    depthB_vis = depth_to_vis(pose_data.depthBs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    if pose_data.normalAs is not None:
      pass
    pad = np.ones((rgbA_vis.shape[0],pad_margin,3))*255
    if pose_data.normalAs is not None:
      pass
    else:
      row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis], axis=1)
    s = 100/row.shape[0]
    row = cv2.resize(row, fx=s, fy=s, dsize=None)
    row = cv_draw_text(row, text=f'id:{id}, score:{scores[id]:.3f}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
    canvas.append(row)
    pad = np.ones((pad_margin, row.shape[1], 3))*255
    canvas.append(pad)
  canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
  return canvas

def vis_batch_data_scores_with_mask(pose_data, ids, scores, mask, pad_margin=5):
    canvas = []
    for id in ids:
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        H,W = rgbA_vis.shape[:2]
        
        # Get mask for current ID
        curr_mask = mask[id].cpu().numpy().reshape(H,W)
        
        # Create mask overlay (green color)
        mask_overlay = np.zeros_like(rgbB_vis)
        mask_overlay[curr_mask > 0] = [0, 255, 0]  # Green color in RGB format since input images are in RGB
        
        # Blend mask with rgbB_vis
        alpha = 0.3  # Transparency of the overlay
        rgbB_masked = cv2.addWeighted(rgbB_vis, 1, mask_overlay, alpha, 0)
        
        zmin = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).min()
        zmax = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).max()
        depthA_vis = depth_to_vis(pose_data.depthAs[id].data.cpu().numpy().reshape(H,W), 
                                 zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(pose_data.depthBs[id].data.cpu().numpy().reshape(H,W), 
                                 zmin=zmin, zmax=zmax, inverse=False)
        
        if pose_data.normalAs is not None:
            pass
            
        pad = np.ones((rgbA_vis.shape[0], pad_margin, 3))*255
        
        if pose_data.normalAs is not None:
            pass
        else:
            row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_masked, pad, depthB_vis], axis=1)
            
        s = 100/row.shape[0]
        row = cv2.resize(row, fx=s, fy=s, dsize=None)
        row = cv_draw_text(row, text=f'id:{id}, score:{scores[id]:.3f}', 
                          uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
        canvas.append(row)
        pad = np.ones((pad_margin, row.shape[1], 3))*255
        canvas.append(pad)
        
    canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
    return canvas

@torch.no_grad()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, mask, K, crop_ratio, normal_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None):
  # logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]

  args = []
  method = 'box_3d'
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
  # logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)

  for b in range(0,len(ob_in_cams),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    xyz_map_rs.append(extra['xyz_map'])

  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  # logging.info("render done")

  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(mask, dtype=torch.float, device='cuda')[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    rgbAs = rgb_rs
    depthAs = depth_rs

  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs

  normalAs = None
  normalBs = None

  Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter

  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  # pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)
  pose_data = process_in_batches(dataset, pose_data, H, W, num_batches=4)

  # logging.info("pose batch data done")

  return pose_data, maskBs

def process_in_batches(dataset, pose_data, H, W, num_batches=4):
    # Get the total batch size
    total_size = len(pose_data.rgbAs)
    
    # Initialize lists to store processed results
    processed_rgbAs, processed_rgbBs = [], []
    processed_depthAs, processed_depthBs = [], []
    processed_poseA, processed_xyz_mapAs = [], []
    processed_xyz_mapBs = []
    processed_tf_to_crops, processed_Ks = [], []
    processed_mesh_diameters = []
    processed_normalAs, processed_normalBs = [], []
    
    # Calculate batch sizes - distribute remainder across first few batches
    base_size = total_size // num_batches
    remainder = total_size % num_batches
    
    # Calculate batch boundaries
    batch_boundaries = [0]
    for i in range(num_batches):
        # Add one extra item to the first 'remainder' batches
        this_batch_size = base_size + (1 if i < remainder else 0)
        batch_boundaries.append(batch_boundaries[-1] + this_batch_size)
    
    # Process each batch separately
    for i in range(num_batches):
        start_idx = batch_boundaries[i]
        end_idx = batch_boundaries[i+1]
        
        # Skip empty batches
        if start_idx == end_idx:
            continue
        
        # Create a batch with just the data for this slice
        batch = BatchPoseData(
            rgbAs=pose_data.rgbAs[start_idx:end_idx],
            rgbBs=pose_data.rgbBs[start_idx:end_idx],
            depthAs=pose_data.depthAs[start_idx:end_idx],
            depthBs=pose_data.depthBs[start_idx:end_idx],
            normalAs=None if pose_data.normalAs is None else pose_data.normalAs[start_idx:end_idx],
            normalBs=None if pose_data.normalBs is None else pose_data.normalBs[start_idx:end_idx],
            poseA=pose_data.poseA[start_idx:end_idx],
            xyz_mapAs=pose_data.xyz_mapAs[start_idx:end_idx],
            tf_to_crops=pose_data.tf_to_crops[start_idx:end_idx],
            Ks=pose_data.Ks[start_idx:end_idx],
            mesh_diameters=pose_data.mesh_diameters[start_idx:end_idx]
        )
        
        # Process this batch
        batch_processed = dataset.transform_batch(batch, H_ori=H, W_ori=W, bound=1)
        
        # Append results to our lists
        processed_rgbAs.append(batch_processed.rgbAs)
        processed_rgbBs.append(batch_processed.rgbBs)
        processed_depthAs.append(batch_processed.depthAs)
        processed_depthBs.append(batch_processed.depthBs)
        processed_poseA.append(batch_processed.poseA)
        processed_xyz_mapAs.append(batch_processed.xyz_mapAs)
        processed_xyz_mapBs.append(batch_processed.xyz_mapBs)
        processed_tf_to_crops.append(batch_processed.tf_to_crops)
        processed_Ks.append(batch_processed.Ks)
        processed_mesh_diameters.append(batch_processed.mesh_diameters)
        
        # Handle normalAs and normalBs depending on whether they're None
        if batch_processed.normalAs is not None:
            processed_normalAs.append(batch_processed.normalAs)
        if batch_processed.normalBs is not None:
            processed_normalBs.append(batch_processed.normalBs)
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Add verification to check lengths before concatenation
    total_processed = sum(tensor.shape[0] for tensor in processed_rgbAs)
    if total_processed != total_size:
        logging.warning(f"Expected to process {total_size} items but processed {total_processed}")
    
    # Combine all processed batches
    combined = BatchPoseData(
        rgbAs=torch.cat(processed_rgbAs, dim=0),
        rgbBs=torch.cat(processed_rgbBs, dim=0),
        depthAs=torch.cat(processed_depthAs, dim=0),
        depthBs=torch.cat(processed_depthBs, dim=0),
        poseA=torch.cat(processed_poseA, dim=0),
        xyz_mapAs=torch.cat(processed_xyz_mapAs, dim=0),
        xyz_mapBs=torch.cat(processed_xyz_mapBs, dim=0),
        tf_to_crops=torch.cat(processed_tf_to_crops, dim=0),
        Ks=torch.cat(processed_Ks, dim=0),
        mesh_diameters=torch.cat(processed_mesh_diameters, dim=0),
        # Handle normalAs and normalBs
        normalAs=None if not processed_normalAs else torch.cat(processed_normalAs, dim=0),
        normalBs=None if not processed_normalBs else torch.cat(processed_normalBs, dim=0)
    )
    
    # Final verification
    assert len(combined.rgbAs) == total_size, f"Expected {total_size} items but got {len(combined.rgbAs)}"
    
    # logging.info(f"Combined batch processing complete. Processed {len(combined.rgbAs)} items.")
    return combined

# @torch.no_grad()
# def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, normal_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None):
#   logging.info("Welcome make_crop_data_batch")
#   H,W = depth.shape[:2]

#   args = []
#   method = 'box_3d'
#   tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
#   logging.info("make tf_to_crops done")

#   B = len(ob_in_cams)
#   poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

#   bs = 512
#   rgb_rs = []
#   depth_rs = []
#   xyz_map_rs = []

#   bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
#   bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)

#   for b in range(0,len(ob_in_cams),bs):
#     extra = {}
#     rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
#     rgb_rs.append(rgb_r)
#     depth_rs.append(depth_r[...,None])
#     xyz_map_rs.append(extra['xyz_map'])

#   rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
#   depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
#   xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
#   logging.info("render done")

#   rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
#   depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
#   if rgb_rs.shape[-2:]!=cfg['input_resize']:
#     rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
#     depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
#   else:
#     rgbAs = rgb_rs
#     depthAs = depth_rs

#   if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
#     xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
#   else:
#     xyz_mapAs = xyz_map_rs

#   normalAs = None
#   normalBs = None

#   Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
#   mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter

#   pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
#   pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)

#   logging.info("pose batch data done")

#   return pose_data


class ScorePredictor:
  def __init__(self, amp=True):
    self.amp = amp
    self.run_name = "2024-01-11-20-02-45"

    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = np.inf
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2

    # logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)
    self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

    # logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)

    self.model.cuda().eval()
    # logging.info("init done")


  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, mask, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None):
    '''
    @rgb: np array (H,W,3)
    '''
    # logging.info(f"ob_in_cams:{ob_in_cams.shape}")
    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

    # logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    # logging.info("making cropped data")

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)

    rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)

    pose_data, mask_vis = make_crop_data_batch(self.cfg.input_resize, ob_in_cams, mesh, rgb, depth, mask, K, crop_ratio=self.cfg['crop_ratio'], glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, cfg=self.cfg, mesh_diameter=mesh_diameter)

    def find_best_among_pairs(pose_data:BatchPoseData):
      # logging.info(f'pose_data.rgbAs.shape[0]: {pose_data.rgbAs.shape[0]}')
      ids = []
      scores = []
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        if pose_data.normalAs is not None:
          A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
          B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)
        with torch.cuda.amp.autocast(enabled=self.amp):
          output = self.model(A, B, L=len(A))
        scores_cur = output["score_logit"].float().reshape(-1)
        ids.append(scores_cur.argmax()+b)
        scores.append(scores_cur)
      ids = torch.stack(ids, dim=0).reshape(-1)
      scores = torch.cat(scores, dim=0).reshape(-1)
      return ids, scores

    pose_data_iter = pose_data
    global_ids = torch.arange(len(ob_in_cams), device='cuda', dtype=torch.long)
    scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device='cuda')

    while 1:
      ids, scores = find_best_among_pairs(pose_data_iter)
      if len(ids)==1:
        scores_global[global_ids] = scores + 100
        break
      global_ids = global_ids[ids]
      pose_data_iter = pose_data.select_by_indices(global_ids)

    scores = scores_global

    # logging.info(f'forward done')
    torch.cuda.empty_cache()

    if get_vis:
      # logging.info("get_vis...")
      canvas = []
      ids = scores.argsort(descending=True)
      # canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores)
      assert len(ids) == len(scores)
      vis_ids = ids[:5]
      # canvas = vis_batch_data_scores(pose_data, ids=vis_ids, scores=scores)
      canvas = vis_batch_data_scores_with_mask(pose_data, ids=vis_ids, scores=scores, mask=mask_vis)
      return scores, canvas

    return scores, None

