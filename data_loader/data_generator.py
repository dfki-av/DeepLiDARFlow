import tensorflow as tf
import sys, os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from preprocessing_utils import *

def init_input_pipeline(data_type, dataset, batch_size, num_samples, augment = False):
    if data_type == 'KITTI':
        interp_shape = [384, 1216]
        crop_shape = [370, 1220]
    else:
        interp_shape = None
        crop_shape = [512, 960]

    gen_func, gen_types, gen_shapes = get_batch_gen(data_type, dataset, num_samples, crop_shape, batch_size)

    gen_data = tf.data.Dataset.from_generator(gen_func, gen_types, gen_shapes)

    batch_gen_data = gen_data.batch(batch_size)

    map_func = get_tf_mapping(augment, interp_shape = interp_shape)
    batch_gen_data = batch_gen_data.map(map_func=map_func)

    batch_gen_data = batch_gen_data.prefetch(batch_size)

    handle = tf.placeholder(tf.string, shape=(), name='initializer')
    iter = tf.data.Iterator.from_string_handle(handle, batch_gen_data.output_types, batch_gen_data.output_shapes)
    flat_inputs = iter.get_next()
    one_shot = batch_gen_data.make_one_shot_iterator()
    return handle, flat_inputs, one_shot

def get_batch_gen(data_type, dataset, _num_samples, crop_shape, batch_size):
    def generator():
        # Generator loop
        while True:
            for idx in range(0, len(dataset)):
                image_10, image_11, disp_10, disp_11, valid_d_10, valid_d_11, sf_2d = dataset[idx]
                image_10 -= get_rgb_mean(data_type)
                image_11 -= get_rgb_mean(data_type)

                if np.isscalar(_num_samples):
                    num_samples = [_num_samples]
                else:
                    num_samples = _num_samples
                idx = 0
                if len(num_samples) > 1:
                    idx = np.random.randint(0, len(num_samples))

                mask_10 = dense_to_sparse(valid_d_10, num_samples[idx])
                mask_11 = dense_to_sparse(valid_d_11, num_samples[idx])
                mean_disp, std_disp = get_disp_mean_std(data_type)
                disp_10 = (disp_10 - mean_disp) / std_disp
                disp_11 = (disp_11 - mean_disp) / std_disp

                disp_10 *= mask_10
                disp_11 *= mask_11

                disp_10 = np.expand_dims(np.array(disp_10), axis=2)
                disp_11 = np.expand_dims(np.array(disp_11), axis=2)

                im_left_10 = np.concatenate((image_10, disp_10), axis=2)
                im_left_11 = np.concatenate((image_11, disp_11), axis=2)
                if data_type == 'FT3D' or batch_size == 2:
                    yield get_cropped(crop_shape, im_left_10, im_left_11, sf_2d)
                else:
                    yield np.array([im_left_10, im_left_11]), sf_2d, sf_2d.shape[0:2]


    gen_func = generator
    gen_types = (tf.float32, tf.float32, tf.int32)
    gen_shapes = ([None, None, None, 4], [None, None, 4], [2])
    return gen_func, gen_types, gen_shapes

def get_tf_mapping(augment, interp_shape):
    def tf_map(_batch_images, batch_sf, batch_orig_shape):
        if augment:
            batch_images = augment_images(_batch_images)
        else:
            batch_images = _batch_images

        input_data = [batch_images]
        input_data += [batch_sf]
        input_data += [batch_orig_shape]
        if interp_shape is not None:
            input_data += [tf.convert_to_tensor([interp_shape])]
        return input_data
    return tf_map


def augment_images(images):
    # Gaussian noise has a sigma uniformly sampled from [0, 0.04]
    image_rgb = images[:, :, :, :, :3]
    disparity = images[:, :, :, :, 3]

    noise = tf.random_normal(shape=tf.shape(image_rgb), mean=0.0, stddev=tf.random_uniform((), 0., 0.04),
                             dtype=tf.float32)
    image_rgb = image_rgb + noise

    # Contrast is sampled within [0.2, 1.4]
    image_rgb = tf.image.adjust_contrast(image_rgb, tf.random_uniform((), 0.2, 1.4))

    # Multiplicative colour changes to the RGB channels per image from [0.5, 2]
    mult = tf.random_uniform((3,), 0.5, 2.)
    image_rgb *= mult
    image_rgb = tf.clip_by_value(image_rgb, 0., 1.)

    # Gamma values from [0.7, 1.5]
    gamma = tf.random_uniform((), 0.7, 1.5)
    image_rgb = tf.image.adjust_gamma(image_rgb, gamma=gamma)

    # Additive brightness changes using Gaussian with a sigma of 0.2
    image_rgb = tf.image.adjust_brightness(image_rgb, tf.truncated_normal((), mean=0., stddev=0.2))
    image_rgb = tf.clip_by_value(image_rgb, 0., 1.)

    disparity = tf.expand_dims(disparity, axis=-1)
    augmented_images = tf.concat([image_rgb, disparity], axis=-1)

    return augmented_images
