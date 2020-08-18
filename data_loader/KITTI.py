import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import os.path as osp
import numpy as np
from io_utils import *

class SceneFlow:
    def __init__(self, root, mode):
        self.root = root
        self.split = 'training'
        self.mode = mode
        self.frames = self.make_dataset()
        if len(self.frames) == 0:
            raise (ValueError("No files in subfolders of: " + self.root + ".\n Please add the correct path of your data set."))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        image_10 = load_kitti_image(self.root, self.split, n=self.frames[index], view='left', frame=0)
        image_11 = load_kitti_image(self.root, self.split, n=self.frames[index], view='left', frame=1)
        disp_10, valid_d_10 = load_kitti_disparity(self.root, self.split, n=self.frames[index], frame=0, occ='occ')
        disp_11, valid_d_11 = load_kitti_disparity_t(self.root, self.split, n=self.frames[index], frame=1, occ='occ')
        sf_2d = load_kitti_sf(self.root, self.split, n=self.frames[index])
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

        root = osp.join(root, self.split, 'flow_occ')
        files = np.sort(os.listdir(root))
        all_paths = [item.split('_10.')[0] for item in files]
        mapping_path = osp.join(osp.dirname(__file__), 'KITTI_test.txt')

        if self.mode == 'TRAIN':
            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in all_paths if lines[int(osp.split(path)[-1])] == '']
            np.random.shuffle(useful_paths)
        else:
            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in all_paths if lines[int(osp.split(path)[-1])] != '']
        return useful_paths

if __name__ == '__main__':
    dataset = SceneFlow(root = './data_scene_flow', mode = 'TRAIN')
    p = len(dataset)
    for idx in range(0,len(dataset)):
        data = dataset[idx]