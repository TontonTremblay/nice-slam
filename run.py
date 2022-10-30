import argparse,os,sys
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()
    print('args.config',args.config)

    code_dir = os.path.dirname(os.path.realpath(__file__))
    cfg = config.load_config(args.config, f'{code_dir}/configs/nice_slam.yaml' if args.nice else f'{code_dir}/configs/imap.yaml')
    if 'downscale_ratio' in cfg:
        cfg['cam']['fx'] *= cfg['downscale_ratio']
        cfg['cam']['fy'] *= cfg['downscale_ratio']
        cfg['cam']['cx'] *= cfg['downscale_ratio']
        cfg['cam']['cy'] *= cfg['downscale_ratio']
        cfg['cam']['H'] = int(cfg['cam']['H']*cfg['downscale_ratio'])
        cfg['cam']['W'] = int(cfg['cam']['W']*cfg['downscale_ratio'])
    else:
        cfg['downscale_ratio'] = 1

    slam = NICE_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
