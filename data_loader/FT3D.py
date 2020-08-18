import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import os.path as osp
import numpy as np
from io_utils import *
import glob

# FT3D Mapping
FT3D_LETTER_LISTS = {
    'A': list(filter(
        lambda x: x not in [12, 18, 96, 132, 186, 441, 456, 483, 653, 676, 728] + [60, 91, 169, 179, 364, 398, 518, 521,
                                                                                   658], range(746))),  # 730
    'B': list(filter(lambda x: x not in [18, 172, 316, 400, 459] + [53, 189, 424, 668], range(746))),  # 741
    'C': list(filter(lambda x: x not in [31, 80, 140, 260, 323, 398, 419, 651], range(746))),  # 742
}

class SceneFlow:
    def __init__(self, root, mode):
        self.root = root
        self.split = mode
        if mode == 'TRAIN' or mode == 'VAL':
            self.split = 'TRAIN'

        self.mode = mode
        self.frames = self.make_dataset()
        if len(self.frames) == 0:
            raise (ValueError("No files in subfolders of: " + self.root + ".\nPlease add the correct path of your data set."))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        scene = self.frames[index][0]
        fname = self.frames[index][1]
        fname2 = "%04d" % (int(fname) + 1)

        # 2D
        image_10 = read_ft3d(osp.join(self.root, 'frames_finalpass', self.split, scene, 'left', fname  + '.png'))[:,:,0:3] / 255.
        image_11 = read_ft3d(osp.join(self.root, 'frames_finalpass', self.split, scene, 'left', fname2 + '.png'))[:,:,0:3] / 255.
        disp_10 = read_ft3d(osp.join(self.root, 'disparity', self.split, scene, 'left', fname + '.pfm'))
        disp_10_change = read_ft3d(osp.join(self.root, 'disparity_change', self.split, scene, 'into_future', 'left', fname + '.pfm'))
        disp_11 = read_ft3d(osp.join(self.root, 'disparity', self.split, scene, 'left', fname2 + '.pfm'))
        flow = read_ft3d(osp.join(self.root, 'optical_flow', self.split, scene, 'into_future', 'left', 'OpticalFlowIntoFuture_' + fname + '_L.pfm'))
        sf_2d = np.stack([flow[:,:,0], flow[:,:,1], disp_10, disp_10+disp_10_change], axis=2)
        valid_d_10 = np.ones(disp_10.shape, dtype=np.bool)
        valid_d_11 = np.ones(disp_11.shape, dtype=np.bool)
        return image_10, image_11, disp_10, disp_11, valid_d_10, valid_d_11, sf_2d

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of frames: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def make_dataset(self):
        root = osp.realpath(osp.expanduser(self.root))
        if not os.path.exists(root):
            raise(ValueError("No scene flow data set in the given path: " + self.root + ".\nPlease add the correct path of your data set."))
        root = sorted(glob.glob(osp.join(root, 'disparity_change', self.split, '*')))
        dirs_A = np.sort(os.listdir(root[0]))
        dirs_B = np.sort(os.listdir(root[1]))
        dirs_C = np.sort(os.listdir(root[2]))

        if self.mode == 'TRAIN' or self.mode == 'VAL':
            dirs_A = np.sort(os.listdir(root[0]))[FT3D_LETTER_LISTS['A']]
            dirs_B = np.sort(os.listdir(root[1]))[FT3D_LETTER_LISTS['B']]
            dirs_C = np.sort(os.listdir(root[2]))[FT3D_LETTER_LISTS['C']]
            if self.mode == 'TRAIN':
                dirs_A = dirs_A[:-50]
                dirs_B = dirs_B[:-50]
                dirs_C = dirs_C[:-50]
            else:
                dirs_A = dirs_A[-50:]
                dirs_B = dirs_B[-50:]
                dirs_C = dirs_C[-50:]

        useful_paths_A = [[osp.join('A', item), str(im).zfill(4)] for item in dirs_A for im in range(6,15)]
        useful_paths_B = [[osp.join('B', item), str(im).zfill(4)] for item in dirs_B for im in range(6,15)]
        useful_paths_C = [[osp.join('C', item), str(im).zfill(4)] for item in dirs_C for im in range(6,15)]
        useful_paths = useful_paths_A + useful_paths_B + useful_paths_C
        if self.split == 'TRAIN':
            np.random.shuffle(useful_paths)
        return useful_paths

if __name__ == '__main__':
    dataset = SceneFlow(
        root='./FlyingThings3D',
        mode='VAL')
    p = len(dataset) # 3933
    for idx in range(0, len(dataset)):
        data = dataset[idx]