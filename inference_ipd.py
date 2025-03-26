# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,os,sys
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import pycocotools.mask as cocomask
import imageio.v2 as imageio

def get_detection(masks, scene_id, image_id, obj_id):
    dets_ = []
    for det in masks:
        if det['scene_id'] == int(scene_id) and det['image_id'] == int(image_id) and det['category_id'] == obj_id:
            dets_.append(det)
    return dets_

def get_masks(reader, scene_id, idx, ob_id, masks):
  image_id = reader.id_strs[idx]
  dets = get_detection(masks, scene_id, image_id, ob_id)
  if len(dets) == 0:
     return None, None

  depth_path = os.path.join(reader.base_dir, 'depth', f'{image_id}.png')
  whole_depth = imageio.imread(depth_path).astype(np.float32) * reader.bop_depth_scale / 1000.0
  
  masks = []
  mask_files = []
  for inst_idx, inst in enumerate(dets):
      seg = inst['segmentation']
      # mask
      h,w = seg['size']
      try:
          rle = cocomask.frPyObjects(seg, h, w)
      except:
          rle = seg
      mask = cocomask.decode(rle)
      mask = np.logical_and(mask > 0, whole_depth > 0)
      if np.sum(mask) < 1000:
          continue
      mask_file = os.path.join(reader.base_dir, 'mask', f'{scene_id}_{int(image_id)}_{ob_id}_{inst_idx}.png')
      masks.append(mask)
      mask_files.append(mask_file)

  masks = np.array(masks)
  return masks, mask_files

def run_pose_estimation_worker(reader, indices, est:FoundationPose=None, debug=0, scene_id=None, im_id=None, ob_id=None, device='cuda:0', masks=None):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()

  for i, index in enumerate(indices):
    # logging.info(f"scene_id:{scene_id}, frame:{im_id}, ob_id:{ob_id}")
    color = reader.get_color(index)
    depth = reader.get_depth(index)
    id_str = reader.id_strs[index]
    
    ob_masks, ob_mask_files = get_masks(reader, scene_id, index, ob_id, masks)
    if ob_masks is None:
      # logging.info("ob_mask not found, skip")
      return None

    for inst_idx, (ob_mask, ob_mask_file) in enumerate(zip(ob_masks, ob_mask_files)):
      pose = est.register_ipd(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_mask_file=ob_mask_file, ob_id=ob_id)
      # logging.info(f"pose:\n{pose}")
      result[ob_id][id_str][inst_idx] = pose

  return result


def run_pose_estimation():
  scene_ids = list(range(15))
  wp.force_load(device='cuda')
  if opt.dataset_name == "ipd":
    reader_tmp = IPDReader(f'{opt.dataset_dir}/test/000001', split=None)
  elif opt.dataset_name == "xyz":
    reader_tmp = XYZReader(f'{opt.dataset_dir}/test/000001', split=None)

  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir
  mask_path = opt.mask_dir

  with open(mask_path, 'r') as f:
    sam6d_masks = json.load(f)

  test_targets_path = opt.test_targets_path
  with open(test_targets_path, "r") as f:
     test_targets = json.load(f)

  res = {}
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  for target in test_targets:
    scene_id = target["scene_id"]
    im_id = target["im_id"][0][1]
    logging.info(f"scene_id:{scene_id}, frame:{im_id}")
    if scene_id not in scene_ids:
      continue
    det_masks = [item for item in sam6d_masks if item['scene_id'] == scene_id and item['image_id'] == im_id and item['score'] > 0.4]
    if len(det_masks) == 0:             
      continue

    det_obj_ids = set([item['category_id'] for item in det_masks])
    for ob_id in det_obj_ids:
      if use_reconstructed_mesh:
        mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
      else:
        mesh = reader_tmp.get_gt_mesh(ob_id)
      symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]

      args = []

      video_dir = f'{opt.dataset_dir}/test/{scene_id:06d}'
      reader = IPDReader(video_dir, split=None)
      est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)

      for i in range(len(reader.color_files)):
        if int(os.path.splitext(os.path.basename(reader.color_files[i]))[0]) == im_id:
          args.append((reader, [i], est, debug, scene_id, im_id, ob_id, "cuda:0", det_masks))

      outs = []
      for arg in args:
        out = run_pose_estimation_worker(*arg)
        if out is not None:
          outs.append(out)

      for out in outs:
        for id_str in out[ob_id]:
          for inst_id in out[ob_id][id_str]:
            pose = out[ob_id][id_str][inst_id]
            res[scene_id].append(
              {
                  "scene_id": str(scene_id),
                  "img_id": str(im_id),
                  "obj_id": str(ob_id),
                  "score": "0",
                  "R": pose[:3,:3].tolist(),
                  "t": (pose[:3,3]*1000).tolist(),
                  "time": 0, 
              }
              )

    with open(f'{opt.debug_dir}/{opt.dataset_name}_sam6d/{scene_id:06d}/{opt.dataset_name}_res.json','w') as file:
      json.dump(res[scene_id], file, indent=2)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--dataset_dir', type=str, default="DATASET/ipd", help="IPD root dir")
  parser.add_argument('--dataset_name', type=str)
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  parser.add_argument('--mask_dir', type=str, default="DATASET/ipd/ipd_mask_sam6d.json", help="mask root dir")
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--test_targets_path', type=str)

  opt = parser.parse_args()
  set_seed(0)

  run_pose_estimation()

