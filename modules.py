import tensorflow as tf
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from tf_utils import convolution, upconvolution
from warping_utils import nearest_warp_1d, nearest_warp_2d, bilinear_warp_1d, bilinear_warp_2d

# Cost volume layer ----------------------------
class CostVolumeLayer(object):

    def __init__(self, search_range=4, name='cost_volume'):
        self.window = search_range
        self.name = name

    def __call__(self, x, warped, dim='2d'):
        assert dim in ['1d', '2d']

        total = []
        keys = []

        if dim == '1d':

            row_shifted = warped

            for i in range(2 * self.window + 1):
                if i != 0:
                    row_shifted = tf.pad(row_shifted, [[0, 0], [0, 0], [1, 0], [0, 0]])
                    row_shifted = tf.keras.layers.Cropping2D([[0, 0], [0, 1]])(row_shifted)

                total.append(tf.reduce_mean(row_shifted * x, axis=-1))

            stacked = tf.stack(total, axis=3)

            return stacked / (2 * self.window + 1)

        else:
            row_shifted = [warped]

            for i in range(self.window+1):
                if i != 0:
                    row_shifted = [tf.pad(row_shifted[0], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                                   tf.pad(row_shifted[1], [[0, 0], [1, 0], [0, 0], [0, 0]])]

                    row_shifted = [tf.keras.layers.Cropping2D([[1, 0], [0, 0]])(row_shifted[0]),
                                   tf.keras.layers.Cropping2D([[0, 1], [0, 0]])(row_shifted[1])]

                for side in range(len(row_shifted)):
                    total.append(tf.reduce_mean(row_shifted[side] * x, axis=-1))
                    keys.append([i * (-1) ** side, 0])
                    col_previous = [row_shifted[side], row_shifted[side]]

                    for j in range(1, self.window+1):
                        col_shifted = [tf.pad(col_previous[0], [[0, 0], [0, 0], [0, 1], [0, 0]]),
                           tf.pad(col_previous[1], [[0, 0], [0, 0], [1, 0], [0, 0]])]

                        col_shifted = [tf.keras.layers.Cropping2D([[0, 0], [1, 0]])(col_shifted[0]),
                                       tf.keras.layers.Cropping2D([[0, 0], [0, 1]])(col_shifted[1])]

                        for col_side in range(len(col_shifted)):
                            total.append(tf.reduce_mean(col_shifted[col_side] * x, axis=-1))
                            keys.append([i * (-1) ** side, j * (-1) ** col_side])

                        col_previous = col_shifted

                if i == 0:
                    row_shifted *= 2

            total = [t for t, _ in sorted(zip(total, keys), key=lambda pair: pair[1])]
            stacked = tf.stack(total, axis=3)

            return stacked / ((2*self.window+1)**2)

class WarpingLayer(object):

    def __init__(self, name='warping_layer'):
        self.name = name

    def __call__(self, x, displacement, type='bilinear', dim='2d'):

        assert type in ['nearest', 'bilinear']
        assert dim in ['1d', '2d']

        if type == 'nearest':
            if dim == '1d':
                return nearest_warp_1d(x, displacement)
            else:
                return nearest_warp_2d(x, displacement)

        else:
            if dim == '1d':
                return bilinear_warp_1d(x, displacement)
            else:
                return bilinear_warp_2d(x, displacement)

class OcclusionEstimator(object):

    def __init__(self, num, reg_constant, is_output=False):
        self.name = 'occlusion_estimator_network_' + num
        self.reg_constant = reg_constant
        self.is_output = is_output

    def __call__(self, inp):

        conv1 = convolution(inp,   128, '1', self.reg_constant)
        conv2 = convolution(conv1,  96, '2', self.reg_constant)
        conv3 = convolution(conv2,  64, '3', self.reg_constant)
        conv4 = convolution(conv3,  32, '4', self.reg_constant)
        features = convolution(conv4,  16, '_feat', self.reg_constant)
        occ_mask = convolution(features,   1, '_occ_mask', self.reg_constant, activation='sigmoid')

        if self.is_output:
            return occ_mask

        else:
            features_up = upconvolution(features, 1, '_up_feat', self.reg_constant, activation='sigmoid')
            occ_mask_up = upconvolution(occ_mask, 1, '_up_occ_mask', self.reg_constant, activation='sigmoid')
            return occ_mask, features_up, occ_mask_up

# Scene flow estimator network
class SceneFlowEstimator(object):

    def __init__(self, num, reg_constant, dense=False, is_output=False):
        self.name = 'scene_flow_estimator_' + num
        self.reg_constant = reg_constant
        self.dense = dense
        self.is_output = is_output

    def __call__(self, concat):

        if self.dense:
            activation = 'leaky_relu'
            for i, filters in zip(['1', '2', '3', '4', '_f', '_w'],
                                  [128, 128, 96, 64, 32, 4]):

                if i == '_w':
                    activation = None

                conv = convolution(concat, filters, i, self.reg_constant, activation=activation)

                if i != '_w':
                    concat = tf.concat([conv, concat], axis=-1)

            if self.is_output:
                return concat, conv

            else:
                flow_up = upconvolution(conv, 4, '_up_flow', self.reg_constant, activation=None)
                feature_up = upconvolution(concat, 4, '_up_feature', self.reg_constant, activation=None)
                return conv, flow_up, feature_up

        else:
            conv1 = convolution(concat, 128, '1',  self.reg_constant)
            conv2 = convolution(conv1,  128, '2',  self.reg_constant)
            conv3 = convolution(conv2,  96,  '3',  self.reg_constant)
            conv4 = convolution(conv3,  64,  '4',  self.reg_constant)

            f_lev = convolution(conv4,  32,  '_f', self.reg_constant)
            w_lev = convolution(f_lev,   4,  '_w', self.reg_constant, activation=None)

            if self.is_output:
                return f_lev, w_lev

            else:
                flow_up = upconvolution(w_lev, 4, '_up_flow', self.reg_constant, activation=None)
                feature_up = upconvolution(f_lev, 4, '_up_feature', self.reg_constant, activation=None)
                return w_lev, flow_up, feature_up

# Context network for scene flow refinement
class ContextNetwork(object):

    def __init__(self, reg_constant, name='context_network'):
        self.name = name
        self.reg_constant = reg_constant

    def __call__(self, inp):

        conv1 = convolution(inp,   128, '1', self.reg_constant, dilation=1)
        conv2 = convolution(conv1, 128, '2', self.reg_constant, dilation=2)
        conv3 = convolution(conv2, 128, '3', self.reg_constant, dilation=4)
        conv4 = convolution(conv3,  96, '4', self.reg_constant, dilation=8)
        conv5 = convolution(conv4,  64, '5', self.reg_constant, dilation=16)
        conv6 = convolution(conv5,  32, '6', self.reg_constant, dilation=1)
        conv7 = convolution(conv6,   4, '7', self.reg_constant, dilation=1, activation=None)

        return conv7
