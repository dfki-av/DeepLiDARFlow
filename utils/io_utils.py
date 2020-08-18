from imageio import imread
import numpy as np
import os
import re

def load_kitti_image(BASEPATH, split, n, view, frame):
    str_view = "image_2"
    if view == 'right':
        str_view = "image_3"
    path_ = os.path.join(BASEPATH, split)
    path  = os.path.join(path_, str_view)
    impath  = os.path.join(path, "%s_%02d.png" % (n, frame+10))
    im = imread(impath, format="PNG-FI") / 255.0
    return im

def load_kitti_disparity(BASEPATH, split, n, frame, occ = 'occ'):
    path = os.path.join(BASEPATH, split)
    path = os.path.join(path, "disp_%s_%01d" % (occ, frame))
    disppath = os.path.join(path,"%s_%02d.png" % (n, 10))

    disp = imread(disppath, format="PNG-FI").astype(float)
    disp /= 256.0
    valid = disp > 0
    if occ == 'noc':
        h, w = valid.shape
        xs, ys = np.meshgrid(range(w), range(h))
        target_x = xs - disp
        valid = np.logical_and(np.logical_and(target_x >= 0, target_x < w-1), valid)
    return disp, valid

def load_kitti_disparity_t(BASEPATH, split, n, frame, occ = 'occ'):
    path = os.path.join(BASEPATH, split)
    path = os.path.join(path, "disp_%s_%01d_transformedOnSecFrame" % (occ, frame))
    if not os.path.exists(path):
        raise (ValueError("No de-warped disparities. \nPlease run download_preprocessed_kitti.sh and place them in your KITTI folder:\n" + BASEPATH + "/training"))
    disppath = os.path.join(path,"%s_%02d.png" % (n, 10))

    disp = imread(disppath, format="PNG-FI").astype(float)
    disp /= 256.0
    valid = disp > 0
    if occ == 'noc':
        h, w = valid.shape
        xs, ys = np.meshgrid(range(w), range(h))
        target_x = xs - disp
        valid = np.logical_and(np.logical_and(target_x >= 0, target_x < w-1), valid)
    return disp, valid

def load_kitti_sf(BASEPATH, split, n):
    path = os.path.join(BASEPATH, split)

    ofpath = os.path.join(path, "flow_occ/" + "%s_10.png" % n)
    d0path = os.path.join(path, "disp_occ_0/" + "%s_10.png" % n)
    d1path = os.path.join(path, "disp_occ_1/" + "%s_10.png" % n)

    of = imread(ofpath, format="PNG-FI").astype(float)
    u = (of[:,:,0] - 2**15) / 64.0
    v = (of[:,:,1] - 2**15) / 64.0
    d0 = imread(d0path) / 256.0
    d1 = imread(d1path) / 256.0

    sf = np.stack([u, v, d0, d1], axis=2)
    sf[d0 <= 0] = 0

    return sf

def read_ft3d(file):
    if file.endswith('.png'): return readImage(file)
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data

    return imread(name)



