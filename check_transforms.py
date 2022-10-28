import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf

import pyrr 
import simplejson as json

import numpy as np 
import glob 
import cv2

# These functions were provided by Lucas Manuelli
def create_visualizer(clear=True, zmq_url='tcp://127.0.0.1:6000'):
    """
    If you set zmq_url=None it will start a server
    """

    print('Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a server')
    vis = meshcat.Visualizer(zmq_url=zmq_url)
    if clear:
        vis.delete()
    return vis

def make_frame(vis, name, T=None, h=0.15, radius=0.001, o=1.0):
    """Add a red-green-blue triad to the Meschat visualizer.
    Args:
    vis (MeshCat Visualizer): the visualizer
    name (string): name for this frame (should be unique)
    h (float): height of frame visualization
    radius (float): radius of frame visualization
    o (float): opacity
    """
    vis[name]['x'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xff0000, reflectivity=0.8, opacity=o))
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]['x'].set_transform(rotate_x)

    vis[name]['y'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00ff00, reflectivity=0.8, opacity=o))
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]['y'].set_transform(rotate_y)

    vis[name]['z'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000ff, reflectivity=0.8, opacity=o))
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]['z'].set_transform(rotate_z)

    if T is not None:
        # print(T)
        vis[name].set_transform(T)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_slam',
        default="Datasets/Demo/frames/pose/*.txt",
        help = "folder of images"
    )

    parser.add_argument(
        '--path_ho3d',
        default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/Downloads/ho3d_eval/SM1/meta/*.pkl",
        help = "folder of images"
    )
    opt = parser.parse_args()


    vis = create_visualizer()

    for i_f, file in enumerate(glob.glob(opt.path_slam)):
        with open(file, 'r') as f:
            lines = f.readlines()
        a = []
        for l in lines:
            a.append(eval("["+",".join(l.replace("\n","").split(" "))+"]"))

        trans = np.array(a)
        make_frame(vis,file,trans)
        if i_f > 20: 
            break
    glcam_in_cvcam = np.array([[1,0,0,0],
                              [0,-1,0,0],
                              [0,0,-1,0],
                              [0,0,0,1]])
    import pickle
    import pyrr
    all_poses = []
    for i_f, file in enumerate(glob.glob(opt.path_ho3d)):
        with open(file, 'rb') as f:
            meta = pickle.load(f)
        
        ob_in_cam_gt = np.eye(4)
        ob_in_cam_gt[:3,3] = meta['objTrans']
        if meta['objRot'] is None: 
            continue
        ob_in_cam_gt[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
        ob_in_cam_gt = glcam_in_cvcam@ob_in_cam_gt
        mat = pyrr.Matrix44(ob_in_cam_gt)
        mat = mat.inverse
        make_frame(vis,file,mat)
        # print(mat)
        # print(mat[0,-1])
        # print(mat[1,-1])
        # print(mat[2,-1])
        all_poses.append([mat[0,-1],mat[1,-1],mat[2,-1]])
        # print(all_poses[-1])
        # raise()
        # if i_f > 20: 
        #     break
    print(np.array(all_poses).max())
    print(np.array(all_poses).min())
    # print(all_poses[-1])