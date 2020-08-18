import numpy as np
from matplotlib import pyplot as plt

ABS_THRESH = 3
REL_THRESH = 0.05

def get_error_maps(sf, gt):
    gtmask = gt[:, :, 2] > 0
    flow_error = np.linalg.norm(sf[:, :, 0:2] * np.tile(np.expand_dims(gtmask, axis=2), [1, 1, 2]) - gt[:, :, 0:2] * np.tile(np.expand_dims(gtmask, axis=2), [1, 1, 2]), axis=2)
    disp0_error = np.abs(sf[:, :, 2] * gtmask - gt[:, :, 2] * gtmask)
    disp1_error = np.abs(sf[:, :, 3] * gtmask - gt[:, :, 3] * gtmask)
    gtFlowMag = np.linalg.norm(gt[:, :, 0:2] * np.tile(np.expand_dims(gtmask, axis=2), [1, 1, 2]), axis=2)
    D0_err = np.logical_and(disp0_error > ABS_THRESH, disp0_error > gt[:, :, 2] * REL_THRESH)
    D1_err = np.logical_and(disp1_error > ABS_THRESH, disp1_error > gt[:, :, 3] * REL_THRESH)
    Fl_err = np.logical_and(flow_error > ABS_THRESH, flow_error > gtFlowMag * REL_THRESH)
    flow_error_map = np.zeros([*gt.shape[0:2], 3])
    flow_error_map[Fl_err] = [1, 0, 1]
    flow_error_map[~Fl_err] = [0, 1, 0]
    flow_error_map = flow_error_map * np.expand_dims(gtmask, axis=2)
    disp0_error_map = np.zeros([*gt.shape[0:2], 3])
    disp0_error_map[D0_err] = [1, 0, 1]
    disp0_error_map[~D0_err] = [0, 1, 0]
    disp0_error_map = disp0_error_map * np.expand_dims(gtmask, axis=2)
    disp1_error_map = np.zeros([*gt.shape[0:2], 3])
    disp1_error_map[D1_err] = [1, 0, 1]
    disp1_error_map[~D1_err] = [0, 1, 0]
    disp1_error_map = disp1_error_map * np.expand_dims(gtmask, axis=2)
    return flow_error_map, disp0_error_map, disp1_error_map

def get_error_KITTI(sf, gt, noc=None):

    errSceneFlowKITTI = np.zeros((gt.shape[0], gt.shape[1], 3), dtype = np.uint8)
    LC1 = np.array([[0,0.0625,49,54,149],
    [0.0625,0.125,69,117,180],
    [0.125,0.25,116,173,209],
    [0.25,0.5,171,217,233],
    [0.5,1,224,243,248],
    [1,2,254,224,144],
    [2,4,253,174,97],
    [4,8,244,109,67],
    [8,16,215,48,39],
    [16,1000000000.0,165,0,38] ])

    gtmask = gt[:, :, 2] > 0
    flow_error = np.linalg.norm(sf[:, :, 0:2] * np.tile(np.expand_dims(gtmask, axis=2), [1, 1, 2]) - gt[:, :, 0:2] * np.tile(np.expand_dims(gtmask, axis=2), [1, 1, 2]), axis=2)
    disp0_error = np.abs(sf[:, :, 2] * gtmask - gt[:, :, 2] * gtmask)
    disp1_error = np.abs(sf[:, :, 3] * gtmask - gt[:, :, 3] * gtmask)
    gtFlowMag = np.linalg.norm(gt[:, :, 0:2] * np.tile(np.expand_dims(gtmask, axis=2), [1, 1, 2]), axis=2)
    n_err = np.zeros(disp0_error.shape, np.float32)
    n_err = np.maximum(n_err, np.minimum(flow_error / 3.0, 20.0 * flow_error / gtFlowMag))
    n_err = np.maximum(n_err, np.minimum(disp0_error / 3.0, 20.0 * disp0_error / gt[:, :, 2]))
    n_err = np.maximum(n_err, np.minimum(disp1_error / 3.0, 20.0 * disp1_error / gt[:, :, 3]))
    for i in range(10):
        y,x = np.where(np.logical_and( n_err >= LC1[i,0], n_err < LC1[i,1]))
        errSceneFlowKITTI[y, x, :] = LC1[i,-3:]
    if noc is not None:
        y, x = np.where(noc == 0)
        errSceneFlowKITTI[y, x, :] = (errSceneFlowKITTI[y, x, :]*0.5).astype(np.uint8)
    y, x = np.where(np.logical_and(errSceneFlowKITTI[:, :, 0] > 0, np.logical_and(errSceneFlowKITTI[:, :, 1]  > 0, errSceneFlowKITTI[:, :, 2] > 0)))
    h, w, _ = errSceneFlowKITTI.shape
    y1 = y-1
    y1[y1 < 0] = 0
    y2 = y+1
    y2[y2 > h] = h - 1
    x1 = x-1
    x1[x1 < 0] = 0
    x2 = x+1
    x2[x2 > w] = w - 1
    yy = [y, y1, y2]
    xx = [x, x1, x2]
    for i in range(3):
        for j in range(3):
            errSceneFlowKITTI[yy[i], xx[j], :] = errSceneFlowKITTI[y, x, :]
    return errSceneFlowKITTI

def plot_error_maps(sf, gt):
    flow_error_map, disp0_error_map, disp1_error_map = get_error_maps(sf, gt)
    plt.figure("Errors")
    ax = plt.subplot(131)
    plt.imshow(flow_error_map)
    ax.axis('off')
    ax = plt.subplot(132)
    plt.imshow(disp0_error_map)
    ax.axis('off')
    ax = plt.subplot(133)
    plt.imshow(disp1_error_map)
    ax.axis('off')
    return