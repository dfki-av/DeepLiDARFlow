from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

UNKNOWN_FLOW_THRESH = 1e7
def colored_flow(flow, maxflow=-1, mask=None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    maxrad = maxflow
    if maxrad < 0:
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    if mask is not None:
        img[mask != 1] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def colored_disparity(disp, maxdisp=-1, mask=None):
    maxd = maxdisp
    if maxd < 0:
        maxd = np.max(disp)
    vals = disp/maxd
    img = cm.jet(vals)
    img[vals > 1] = [1,0,0,1]
    if mask is not None:
        img[mask != 1] = [0,0,0,1]
    return np.uint8(img*255)

def colored_depth(depth, mindepth=-1, mask=None, overlay=None):
    invd = np.reciprocal(depth, where=mask)
    maxval = 1.0/mindepth
    if maxval < 0:
        maxval = np.max(invd)
    vals = invd/maxval
    img = cm.jet(vals)
    img[vals > 1] = [1,0,0,1]
    if mask is not None:
        img[mask != 1] = [0,0,0,1]
    img = img[:,:,0:3]
    if overlay is not None:
        img = (img * 2.0 + overlay) / 3.0
    return np.uint8(img*255)

def plot_flow(flow, maxflow=-1, mask=None):
    img = colored_flow(flow, maxflow, mask)
    plt.imshow(img)
    return


def plot_disparity(disp, maxdisp=-1, mask=None):
    if mask is None:
        mask = (disp > 0).astype(np.float)
    img = colored_disparity(disp, maxdisp, mask)
    plt.imshow(img)
    return

def plot_sceneflow(sf, name="SceneFlow"):
    mask = sf[:,:,2] > 0
    f = plt.figure(name)
    f.set_figheight(5)
    f.set_figwidth(20)
    plt.subplot(131)
    plt.title(name + ": Optical Flow")
    plot_flow(sf[:,:,0:2], mask=mask)
    plt.subplot(132)
    plt.title(name + ": Disparity 0")
    plot_disparity(sf[:,:,2], mask=mask)
    plt.subplot(133)
    plt.title(name + ": Disparity 1")
    plot_disparity(sf[:,:,3], mask=mask)
    return
