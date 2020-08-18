import argparse
import tensorflow as tf
import numpy as np
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
from checkpointsaver import BestCheckpointSaver, get_best_checkpoint
from distutils import util
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--dataset', default='FT3D', help='Dataset type [default: FT3D]')
parser.add_argument('--data_path', default='./FlyingThings3D', help='Dataset directory [default: ./FlyingThings3D]')
parser.add_argument('--batch_size', type=int, default=2, help='batch size [default: 2]')
parser.add_argument('--max_epoch', type=int, default=650, help='max epoch [default: 650]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate [default: 0.0001]')
parser.add_argument('--best_checkpoint', type=util.strtobool, default=False, help='model best checkpoint [default: False]')
parser.add_argument('--pretrained_model', default=None, help='pretrained model to fine tune [default: None]')

FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
if FLAGS.dataset == 'KITTI':
    STEP_SIZE = 1
else:
    STEP_SIZE = 5

def eval_one_step(dataset, BATCH_SIZE, ops, sess, handle_data, writer, epoch, step):
    data_size = len(dataset)
    total_loss = 0.
    total_sf_keo = 0.
    total_d0_keo = 0.
    total_d1_keo = 0.
    total_fl_keo = 0.
    total_sf_epe = 0.
    count = 0.
    feed_dict = {ops['handle_pl']: handle_data}
    for _ in tqdm(range(data_size // BATCH_SIZE // STEP_SIZE)):
        summary, batch_loss, eval_kitti = sess.run([ops['summary_pl'], ops['loss_pl'], ops['eval_kitti_pl']], feed_dict=feed_dict)
        writer.add_summary(summary, step)
        writer.flush()
        total_loss += batch_loss
        total_d0_keo += eval_kitti[0]
        total_d1_keo += eval_kitti[1]
        total_fl_keo += eval_kitti[2]
        total_sf_keo += eval_kitti[3]
        total_sf_epe += eval_kitti[4]
        count += 1.

    mean_loss = total_loss / count
    mean_sf_keo = total_sf_keo / count
    mean_sf_epe = total_sf_epe / count
    tqdm.write('\n Valid: epoch/step: {:d}/{:d}, loss:{:.2f}, koe:{:.2f}, epe:{:.2f}'.format(epoch, step, mean_loss, mean_sf_keo, mean_sf_epe))
    return mean_sf_keo

def train_one_step(dataset, BATCH_SIZE, ops, sess, handle_data, writer, epoch, step):
    data_size = len(dataset)
    total_loss = 0.
    total_sf_keo = 0.
    total_d0_keo = 0.
    total_d1_keo = 0.
    total_fl_keo = 0.
    total_sf_epe = 0.
    count = 0.
    feed_dict = {ops['handle_pl']: handle_data}
    for _ in tqdm(range(data_size // BATCH_SIZE // STEP_SIZE)):
        _,summary, batch_loss, eval_kitti = sess.run([ops['train_op'], ops['summary_pl'], ops['loss_pl'], ops['eval_kitti_pl']], feed_dict=feed_dict)
        writer.add_summary(summary, step)
        writer.flush()
        total_loss += batch_loss
        total_d0_keo += eval_kitti[0]
        total_d1_keo += eval_kitti[1]
        total_fl_keo += eval_kitti[2]
        total_sf_keo += eval_kitti[3]
        total_sf_epe += eval_kitti[4]
        count += 1.
    mean_loss = total_loss / count
    mean_sf_keo = total_sf_keo / count
    mean_sf_epe = total_sf_epe / count
    tqdm.write('\n Train: epoch/step: {:d}/{:d}, loss:{:.2f}, koe:{:.2f}, epe:{:.2f}'.format(epoch, step, mean_loss, mean_sf_keo, mean_sf_epe))
    return mean_sf_keo

def load_model(sess, weights_dirctory = False):
    saver = tf.train.Saver(name='saver', max_to_keep=100)
    if weights_dirctory:
        print('load best model...')
        # make sure to include it in the checpoints/best_checkpoints JSON file!!
        if not os.path.exists('best_checkpoints'):
            raise(ValueError("No best check points available \n"))
        saver.restore(sess, get_best_checkpoint('model', select_maximum_value=False))
    elif FLAGS.pretrained_model is not None:
        if not os.path.exists(FLAGS.pretrained_model + '.meta'):
            raise(ValueError("No pretrained model available \n"))
        print('load pretrained model...')
        saver.restore(sess, FLAGS.pretrained_model)
    else:
        print('Initializing new model...')
        sess.run(tf.global_variables_initializer())
    return True

def train(data_type, dataset_train, dataset_val, BATCH_SIZE, MAX_EPOCH, SAMPLES, LEARNING_RATE):
    with tf.device('/gpu:'+str(FLAGS.gpu)):
        with tf.Graph().as_default():
            handle, inputs, one_shot_train = init_input_pipeline(data_type, dataset_train, BATCH_SIZE, SAMPLES, True)
            _, _, one_shot_val = init_input_pipeline(data_type, dataset_val, BATCH_SIZE, SAMPLES)

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

            # preparing the training params
            with tf.variable_scope('train'):
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                loss_with_l2 = tf.add_n([loss] + reg_losses, name='loss_with_L2')
                global_step = tf.Variable(-1, dtype=tf.int32, trainable=False, name='global_step')
                increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')
                optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='Adam_optimizer')
                train_op = optimizer.minimize(loss_with_l2, name='train_op')

            # preparing the summary objects
            with tf.variable_scope('summaries'):
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('outliers', eval_kitti[3])
                tf.summary.scalar('epe', eval_kitti[4])
                summary = tf.summary.merge_all()

            saver = tf.train.Saver(name='saver',max_to_keep=100)

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            handle_train = sess.run(one_shot_train.string_handle())
            handle_val = sess.run(one_shot_val.string_handle())

            # summary writers
            writer_train = tf.summary.FileWriter(os.path.join('summary', 'train'), sess.graph)
            writer_valid = tf.summary.FileWriter(os.path.join('summary', 'valid'), sess.graph)
            load_model(sess, weights_dirctory=FLAGS.best_checkpoint)
            best_ckpt_saver = BestCheckpointSaver(save_dir='model', num_to_keep=5, maximize=False, saver=saver)

            ops = {'handle_pl': handle,
                   'images_pl': images,
                   'gt_sf_pl': gt_sf,
                   'gt_shape_pl': gt_shape,
                   'train_op':train_op,
                   'out_sf_list': out_sf_list,
                   'out_sf_pl': out_sf,
                   'loss_pl': loss,
                   'eval_kitti_pl': eval_kitti,
                   'summary_pl': summary}

            for epoch in range(MAX_EPOCH):
                count = 0
                name = 'DeepLiDARFlow-' + FLAGS.dataset + '-epoch-' + str(epoch) + '-step-'
                while count < STEP_SIZE:
                    step = sess.run(increment_global_step)
                    train_one_step(dataset_train, BATCH_SIZE, ops, sess, handle_train, writer_train, epoch, step)
                    sf_keo = eval_one_step(dataset_val, BATCH_SIZE, ops, sess, handle_val, writer_valid, epoch, step)

                    # Save the variables to disk.
                    best_ckpt_saver.handle(sf_keo, sess, step, name)
                    count += 1

if __name__ == '__main__':
    DATAPATH = FLAGS.data_path
    DATASET = importlib.import_module(FLAGS.dataset)
    SAMPLES = np.array([100, 500, 1000, 5000, 10000])
    dataset_train = DATASET.SceneFlow(
        root= DATAPATH,
        mode='TRAIN')
    dataset_val = DATASET.SceneFlow(
        root= DATAPATH,
        mode='VAL')

    # Hyperparameters
    BATCH_SIZE = FLAGS.batch_size
    MAX_EPOCH = FLAGS.max_epoch
    DATA_TYPE = FLAGS.dataset
    LEARNING_RATE = FLAGS.learning_rate
    train(DATA_TYPE, dataset_train, dataset_val, BATCH_SIZE, MAX_EPOCH, SAMPLES, LEARNING_RATE)
