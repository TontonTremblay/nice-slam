import os

import torch
import numpy as np


class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, slam
                 ):
        self.verbose = slam.verbose
        self.ckptsdir = slam.ckptsdir
        self.shared_c = slam.shared_c
        self.gt_c2w_list = slam.gt_c2w_list
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'c': self.shared_c,
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            # 'keyframe_dict': keyframe_dict, # to save keyframe_dict into ckpt, uncomment this line
            'selected_keyframes': selected_keyframes,
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)
        # print(self.estimate_c2w_list.shape)
        # print()
        name_out =self.ckptsdir.replace('ckpts','poses')
        if not os.path.exists(name_out):
            os.mkdir(name_out)

        d = self.estimate_c2w_list.cpu().numpy()
        for i in range(idx):
            if np.sum(d[i])== 0: 
                break
            if os.path.exists(f'{name_out}/{str(i).zfill(4)}.txt'):
                continue
            np.savetxt(f'{name_out}/{str(i).zfill(4)}.txt', d[i] )
        # raise()
        if self.verbose:
            print('Saved checkpoints at', path)
