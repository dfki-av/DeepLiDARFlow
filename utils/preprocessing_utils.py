import numpy as np

def dense_to_sparse(depth_mask, num_samples):
    n_keep = float(np.count_nonzero(depth_mask))
    prob = num_samples / n_keep
    matrix = np.random.uniform(low=0., high=1., size=(depth_mask.shape)) < prob
    matrix = matrix.astype(np.float32)
    matrix *= depth_mask
    return matrix


def get_rgb_mean(data_type):
    if data_type == 'KITTI':
        mean_pixel = [0.3791041, 0.39846687, 0.38367166]
    else:
        mean_pixel = [0.424101, 0.40341005, 0.36796424]
    return mean_pixel


def get_disp_mean_std(data_type):
    if data_type == 'KITTI':
        MEAN_DISP = 28.1591
        STD_DISP = 19.9899
    else:
        MEAN_DISP = 41.232
        STD_DISP = 35.423
    return MEAN_DISP, STD_DISP

def get_cropped(crop_shape, left_10, left_11, sf_2d):
    def_shape = np.array(crop_shape)
    offset = (np.array(left_10.shape)[0:2] - def_shape) // 2
    cropped_left_10 = left_10[offset[0]:offset[0] + def_shape[0], offset[1]:offset[1] + def_shape[1], :]
    cropped_left_11 = left_11[offset[0]:offset[0] + def_shape[0], offset[1]:offset[1] + def_shape[1], :]
    cropped_sf_2d = sf_2d[offset[0]:offset[0] + def_shape[0], offset[1]:offset[1] + def_shape[1], :]
    return np.array([cropped_left_10, cropped_left_11]), cropped_sf_2d, crop_shape