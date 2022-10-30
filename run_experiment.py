import subprocess,os,sys,cv2,joblib,copy
sys.path.append(f"/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/github/HashNeRF-pytorch/BundleTrack/scripts/")
from data_reader import *
import numpy as np
import glob,imageio,pyexr,yaml
import subprocess
import glob
import os


def make_data_one_video(reader, out_dir):
  if os.path.exists(f"{out_dir}/color"):
    tmp = glob.glob(f"{out_dir}/color/*")
    if len(tmp)==len(reader.color_files):
      print(f"make_data_one_video skip {reader.video_dir}")
      return

  os.system(f"rm -rf {out_dir} && mkdir -p {out_dir}/color {out_dir}/depth {out_dir}/pose {out_dir}/output")
  code_dir = os.path.dirname(os.path.realpath(__file__))

  for i in range(len(reader.color_files)):
    color = imageio.imread(reader.color_files[i])
    mask = reader.get_mask(i)
    depth = reader.get_depth(i)
    color[mask==0] = 0
    depth[mask==0] = -1
    imageio.imwrite(f'{out_dir}/color/{reader.id_strs[i]}.jpg', color)
    imageio.imwrite(f'{out_dir}/depth/{reader.id_strs[i]}.exr', depth.astype(np.float32))
    ob_in_cam = reader.get_gt_pose(i)
    if ob_in_cam is None:
      ob_in_cam = np.eye(4)
    np.savetxt(f'{out_dir}/pose/{reader.id_strs[i]}.txt', np.linalg.inv(ob_in_cam))



def run_one_video(reader, data_dir):
  # subprocess.call(
  #   [
  #   "OPENCV_IO_ENABLE_OPENEXR=1",
  #   "python", "run.py", "configs/Demo/demo_bundlenerf.yaml",
  #   "--input_folder",f'{data_dir}',
  #   "--output",f"{data_dir}/output/"
  #   ]
  # )
  os.system(f"rm -rf {data_dir}/output && mkdir -p {data_dir}/output")

  tmp = imageio.imread(reader.color_files[0])
  H,W = tmp.shape[:2]

  code_dir = os.path.dirname(os.path.realpath(__file__))
  cfg = yaml.load(open(f"{code_dir}/configs/Demo/demo_bundlenerf.yaml",'r'))
  cfg['downscale_ratio'] = 0.5
  cfg['data']['input_folder'] = f"{data_dir}"
  cfg['data']['output'] = f"{data_dir}/output"
  cfg['cam']['H'] = int(H)
  cfg['cam']['W'] = int(W)
  cfg['cam']['fx'] = float(reader.K[0,0])
  cfg['cam']['fy'] = float(reader.K[1,1])
  cfg['cam']['cx'] = float(reader.K[0,2])
  cfg['cam']['cy'] = float(reader.K[1,2])
  gt_mesh = reader.get_gt_mesh()
  max_xyz = gt_mesh.vertices.max(axis=0) + 0.5
  min_xyz = gt_mesh.vertices.min(axis=0) - 0.5
  cfg['mapping']['bound'] = np.stack((min_xyz,max_xyz), axis=-1).reshape(3,2).tolist()
  cfg['mapping']['marching_cubes_bound'] = copy.deepcopy(cfg['mapping']['bound'])
  yaml.dump(cfg, open(f"{data_dir}/cfg.yml",'w'))

  code_dir = os.path.dirname(os.path.realpath(__file__))
  os.system(f"OPENCV_IO_ENABLE_OPENEXR=1 python {code_dir}/run.py --config {data_dir}/cfg.yml")


def run_ho3d():
  video_dirs = sorted(glob.glob(f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/*'))

  args = []
  args1 = []
  for video_dir in video_dirs:
    reader = Ho3dReader(video_dir)
    out_dir = f'/home/bowen/debug/ho3d_nice_slam/{os.path.basename(video_dir)}'
    args.append((reader, out_dir))
    args1.append((reader, out_dir,))
    break

  joblib.Parallel(n_jobs=1)(joblib.delayed(make_data_one_video)(*arg) for arg in args)
  joblib.Parallel(n_jobs=1)(joblib.delayed(run_one_video)(*arg) for arg in args1)



if __name__=="__main__":
  os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

  run_ho3d()