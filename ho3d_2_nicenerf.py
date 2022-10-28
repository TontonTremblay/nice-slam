import numpy as np 
import cv2 
import argparse
import glob 
import os 
import imageio 

parser = argparse.ArgumentParser()
parser.add_argument('--pimgs',default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/Downloads/ho3d_eval/SM1/rgb/")
parser.add_argument('--pmasks',default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/Downloads/ho3d_eval/masks_XMem/SM1/")
parser.add_argument('--outf',default="Datasets/ho3d/SM1/")
opt = parser.parse_args()

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
imgs = sorted(glob.glob(opt.pimgs+"*.jpg"))
masks = sorted(glob.glob(opt.pmasks+"/*.png"))
depths = sorted(glob.glob(opt.pimgs.replace("rgb","depth")+"/*.png"))
poses = sorted(glob.glob(opt.pimgs.replace("rgb","meta")+"/*.pkl"))

# print(opt.pimgs)
# print(opt.pmasks)
# print(opt.outf)

img_outf = f"{opt.outf}/color/"
if not os.path.exists(img_outf):
    os.mkdir(img_outf)

depth_outf = f"{opt.outf}/depth/"
if not os.path.exists(depth_outf):
    os.mkdir(depth_outf)

pose_outf = f"{opt.outf}/pose/"
if not os.path.exists(pose_outf):
    os.mkdir(pose_outf)

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])
i_adding = 1
for i_img, img in enumerate(imgs):

    import pickle
    import pyrr

    with open(poses[i_img], "rb") as f:
        meta = pickle.load(f)

    ob_in_cam_gt = np.eye(4)
    ob_in_cam_gt[:3,3] = meta['objTrans']
    if meta['objRot'] is None: 
        continue
    ob_in_cam_gt[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
    ob_in_cam_gt = glcam_in_cvcam@ob_in_cam_gt
    mat = pyrr.Matrix44(ob_in_cam_gt)
    mat = mat.inverse
    s_out = ""
    for m in mat: 
        s_out += str(m) + "\n"
    # print(s_out)
    with open(f"{pose_outf}/{str(i_adding).zfill(4)}.txt",'w') as f: 
        f.write(s_out.replace("[",'').replace("]",''))

    im = cv2.imread(img)
    mask = cv2.imread(masks[i_img])
    # print(mask.shape)
    output = np.zeros((im.shape[0],im.shape[1],3))
    output[:,:,:3] = im
    # output[:,:,3] = 255
    output[:,:][mask[:,:,0]<1] = 0 
    # print(output[:,:,3].min(),output[:,:,3].max(),output[:,:,3].mean())
    # output
    cv2.imwrite(f"{img_outf}/{str(i_adding).zfill(4)}.jpg",output)
    
    # make depth
    depth_scale = 0.00012498664727900177
    depth = cv2.imread(depths[i_img], -1)
    depth = (depth[...,2]+depth[...,1]*256)*depth_scale
    # depth_data = cv2.imread(depths[i_img], cv2.IMREAD_UNCHANGED)
    # depth_data = depth_data[...,2] + depth_data[...,1]*256

    black = np.ones((im.shape[0],im.shape[1]))*-1
    black[mask[:,:,0]>0] = depth[mask[:,:,0]>0]

    output_name = f"{depth_outf}/{str(i_adding).zfill(4)}.exr"
    # print(black.min(),black.max(),black.mean())
    imageio.imwrite(output_name, black.astype("float32"))
    

    
    i_adding += 1

    # raise()