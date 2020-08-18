import argparse
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import importlib
from tqdm import tqdm
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_loader'))
from data_generator import *
from deeplidarflow import DeepLiDARFlowNet, get_loss_KITTI, get_loss_FT3D, get_eval

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--samples', type = float, default=5000, help='Samples [default: 5000]')
parser.add_argument('--dataset', default='KITTI', help='Dataset type [default: KITTI]')
parser.add_argument('--data_path', default='./data_scene_flow', help='Dataset directory [default: ./data_scene_flow]')
parser.add_argument('--batch_size', type= int, default=1, help='batch size [default: 1]')
parser.add_argument('--model_path', default='./model/DeepLiDARFlow', help='model path [default: ./model/DeepLiDARFlow]')

FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def eval(dataset, BATCH_SIZE, ops, sess, handle_data):
    data_size = len(dataset)
    total_loss = 0.
    total_sf_keo = 0.
    total_d0_keo = 0.
    total_d1_keo = 0.
    total_fl_keo = 0.
    total_sf_epe = 0.
    count = 0.
    feed_dict = {ops['handle_pl']: handle_data}
    for _ in tqdm(range(data_size // BATCH_SIZE)):
        batch_loss, eval_kitti = sess.run([ops['loss_pl'], ops['eval_kitti_pl']], feed_dict=feed_dict)
        total_loss += batch_loss
        total_d0_keo += eval_kitti[0]
        total_d1_keo += eval_kitti[1]
        total_fl_keo += eval_kitti[2]
        total_sf_keo += eval_kitti[3]
        total_sf_epe += eval_kitti[4]
        count += 1.

    return total_loss / count, total_d0_keo / count, total_d1_keo / count, total_fl_keo / count, total_sf_keo / count, total_sf_epe / count

def evaluate(data_type, dataset, BATCH_SIZE, SAMPLES, PRETRAINED_MODEL):
    if not os.path.exists(PRETRAINED_MODEL + '.meta'):
        raise (ValueError(PRETRAINED_MODEL + " model is not available.\nPlease add the correct path of your model.\n"))
    with tf.device('/gpu:'+str(FLAGS.gpu)):
        with tf.Graph().as_default():
            handle, inputs, one_shot = init_input_pipeline(data_type, dataset, BATCH_SIZE, SAMPLES)
            images = inputs[0]
            gt_sf = inputs[1]
            gt_shape = inputs[2][0]
            if len(inputs) == 4:
                interp_shape = inputs[3][0]
            else:
                interp_shape = None
            out_sf_list, out_sf = DeepLiDARFlowNet(images, gt_shape, interp_shape)

            if FLAGS.dataset == 'KITTI':
                loss = get_loss_KITTI(out_sf_list, gt_sf, gt_shape)
            else:
                loss = get_loss_FT3D(out_sf_list, gt_sf)

            eval_kitti = get_eval(out_sf, gt_sf)
            saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            handle_data = sess.run(one_shot.string_handle())
            saver.restore(sess, PRETRAINED_MODEL)

            ops = {'handle_pl': handle,
                   'images_pl': images,
                   'gt_sf_pl': gt_sf,
                   'gt_shape_pl': gt_shape,
                   'out_sf_list': out_sf_list,
                   'out_sf_pl': out_sf,
                   'loss_pl': loss,
                   'eval_kitti_pl': eval_kitti}
            return eval(dataset, BATCH_SIZE, ops, sess, handle_data)

if __name__ == '__main__':
    DATAPATH = FLAGS.data_path
    SAMPLES = FLAGS.samples

    DATASET = importlib.import_module(FLAGS.dataset)
    dataset = DATASET.SceneFlow(
        root= DATAPATH,
        mode='TEST')

    # Training specifications
    PRETRAINED_MODEL = FLAGS.model_path + '-' + FLAGS.dataset

    # Hyperparameters
    BATCH_SIZE = FLAGS.batch_size
    DATA_TYPE = FLAGS.dataset
    SAMPLES = FLAGS.samples
    loss, d0_keo, d1_keo, fl_keo, sf_keo, sf_epe = evaluate(DATA_TYPE, dataset, BATCH_SIZE, SAMPLES, PRETRAINED_MODEL)
    print(f"loss: {loss}, do_keo: {d0_keo}, d1_keo: {d1_keo}, fl_keo: {fl_keo}, sf_keo: {sf_keo}, sf_epe: {sf_epe}")