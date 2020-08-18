import argparse
from matplotlib import pyplot as plt
import tensorflow as tf
import sys, os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from preprocessing_utils import *
from deeplidarflow import DeepLiDARFlowNet
from io_utils import readImage, readPFM
from vis_utils import plot_sceneflow, colored_flow, colored_disparity
from imageio import imwrite
from tensorflow.python.tools import freeze_graph

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data_path', default='./images', help='images directory [default: ./images]')
parser.add_argument('--ex', type = int, default=1, help='example [default: 1]')
parser.add_argument('--model_path', default='./model/DeepLiDARFlow',help='model path [default: ./model/DeepLiDARFlow]')

FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def compute_flow(images, _gt_shape, _interp_shape, MODEL_PATH):
    if not os.path.exists(MODEL_PATH + '.meta'):
        raise (ValueError(MODEL_PATH + " model is not available.\nPlease add the correct path of your model.\n"))
    with tf.device('/gpu:' + str(FLAGS.gpu)):
        with tf.Graph().as_default():
            img_ph = tf.placeholder(tf.float32, (None, None, None, None, 4), name='img_ph')
            gt_shape = tf.placeholder(tf.int32, (2,), name='gt_shape')
            if _interp_shape is not None:
                interp_shape = tf.placeholder(tf.int32, (2,), name='interp_shape')
                _,out_sf = DeepLiDARFlowNet(img_ph, gt_shape, interp_shape)
            else:
               _, out_sf = DeepLiDARFlowNet(img_ph, _gt_shape)

            saver = tf.train.Saver()
            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            saver.restore(sess, MODEL_PATH)

            if _interp_shape is not None:
                predicted_sf = sess.run(out_sf,feed_dict={img_ph: np.expand_dims(images, axis=0), gt_shape: _gt_shape, interp_shape: _interp_shape})[0, :, :, :]
            else:
                predicted_sf = sess.run(out_sf, feed_dict={img_ph: np.expand_dims(images, axis=0), gt_shape: _gt_shape})[0, :, :, :]
            return predicted_sf

def preprocess_images(data_type, image_10, image_11, disp_10, disp_11):
    image_10 -= get_rgb_mean(data_type)
    image_11 -= get_rgb_mean(data_type)
    mean_disp, std_disp = get_disp_mean_std(data_type)
    mask_10 = disp_10 > 0
    mask_11 = disp_11 > 0
    disp_10 = (disp_10 - mean_disp) / std_disp
    disp_11 = (disp_11 - mean_disp) / std_disp
    disp_10 *= mask_10
    disp_11 *= mask_11
    disp_10 = np.expand_dims(np.array(disp_10), axis=2)
    disp_11 = np.expand_dims(np.array(disp_11), axis=2)

    im_left_10 = np.concatenate((image_10, disp_10), axis=2)
    im_left_11 = np.concatenate((image_11, disp_11), axis=2)
    sf_2d = np.zeros(shape=im_left_10.shape).astype(np.float32)
    if data_type == 'KITTI':
        interp_shape = [384, 1216]
        images_list = np.array([im_left_10, im_left_11])
        final_shape = im_left_10.shape[0:2]
    else:
        interp_shape = None
        crop_shape = [512, 960]
        images_list, _, final_shape = get_cropped(crop_shape, im_left_10, im_left_11, sf_2d)

    return images_list, final_shape, interp_shape

if __name__ == '__main__':
    MODEL_PATH = FLAGS.model_path
    DATA_PATH = FLAGS.data_path
    EX = "%d" % (FLAGS.ex)
    image_10_ = readImage(DATA_PATH + '/ex_' + EX + '_im_10.png')[:,:,:3]
    image_10 = image_10_ / 255.
    image_11_ = readImage(DATA_PATH + '/ex_' + EX + '_im_11.png')[:,:,:3]
    image_11 = image_11_ / 255.

    disp_10 = readImage(DATA_PATH + '/ex_' + EX + '_disp_10.png') / 256.
    disp_11 = readImage(DATA_PATH + '/ex_' + EX + '_disp_11.png') / 256.

    if image_10.shape[1] > 1000:
        data_type = 'KITTI'
    else:
        data_type = 'FT3D'

    images, final_shape, interp_shape = preprocess_images(data_type, image_10, image_11, disp_10, disp_11)
    out_sf = compute_flow(images, final_shape, interp_shape, MODEL_PATH + '-' + data_type)

    # write output scene flow
    imwrite(DATA_PATH + '/out_' + EX + '_optical_flow.png', colored_flow(out_sf[:, :, :2]))
    imwrite(DATA_PATH + '/out_' + EX + '_disparity1.png', colored_disparity(out_sf[:, :, 2]))
    imwrite(DATA_PATH + '/out_' + EX + '_disparity2.png', colored_disparity(out_sf[:, :, 3]))

    # plot
    f, axarr = plt.subplots(2, 2)
    f.set_figheight(5)
    f.set_figwidth(20)
    f.suptitle('Input to network: Images')
    axarr[0, 0].imshow(image_10_)
    axarr[0, 1].imshow(image_11_)
    f.suptitle('Input to network: Sparse Disparities (5000 Samples)')
    axarr[1, 0].imshow(colored_disparity(disp_10, mask = disp_10 > 0))
    axarr[1, 1].imshow(colored_disparity(disp_11, mask = disp_11 > 0))
    plot_sceneflow(out_sf, name="Output Scene Flow")
    plt.show()
