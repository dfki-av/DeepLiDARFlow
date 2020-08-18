import tensorflow as tf
from networks import final_network
from modules import ContextNetwork, CostVolumeLayer, WarpingLayer, SceneFlowEstimator

reg_constant = 0.0001

def get_loss_KITTI(out_sf_list, gt_sf, gt_shape):
    loss = tf.constant(0.)
    ϵ = 0.01
    mask = gt_sf[:, :, :, 2] > 0.
    gt_rel = tf.boolean_mask(gt_sf / 20., mask)
    weights = [1., 1., 1., 2., 4.]

    for weight, flow in zip(weights, out_sf_list):
        resized_pred = tf.image.resize_bilinear(flow, gt_shape, name='resized_prediction')
        prediction_rel = tf.boolean_mask(resized_pred, mask)
        difference = tf.abs(prediction_rel - gt_rel)
        error = tf.reduce_sum(difference, axis=-1)
        error = tf.pow(error + ϵ, 0.4)
        error = tf.reduce_mean(error)
        loss += weight * error
    return loss

def get_loss_FT3D(flows,sf):
    loss = tf.constant(0.)
    weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    for weight, flow in zip(weights, flows):
        h = tf.shape(flow)[1]
        w = tf.shape(flow)[2]
        resized_sf = tf.image.resize_bilinear(sf / 20., [h, w], name='resized_sf')
        l2_norm = tf.norm(flow - resized_sf, ord='euclidean', axis=-1, name='l2_norm')
        error = tf.reduce_mean(tf.reduce_sum(l2_norm, axis=(1, 2), name='error_per_image'))
        loss += weight * error
    return loss

def get_eval(out_sf, gt_sf):

    gt_mask = gt_sf[:, :, :, 2] > 0.
    gt_masked_sf = tf.boolean_mask(gt_sf, gt_mask)

    out_masked_sf = tf.boolean_mask(out_sf, gt_mask)
    error = tf.abs(out_masked_sf - gt_masked_sf)
    sf_epe = tf.reduce_mean(tf.norm(error, axis=-1))

    flow_mag = tf.norm(gt_masked_sf[:, :2], axis=-1, name='flow_magnitude')

    flow_epe = tf.norm(error[:, :2], axis=-1, name='flow_epe')
    d0_error = error[:, 2]
    d1_error = error[:, 3]

    d0_pre_keo = tf.logical_and(d0_error > 3., (d0_error / gt_masked_sf[:, 2]) > 0.05, name='d0_thresh')
    d1_pre_keo = tf.logical_and(d1_error > 3., (d1_error / gt_masked_sf[:, 3])> 0.05, name='d1_thresh')
    fl_pre_keo = tf.logical_and(flow_epe > 3., (flow_epe / flow_mag) > 0.05 , name='flow_thresh')

    sf_keo = tf.logical_or(fl_pre_keo, tf.logical_or(d0_pre_keo, d1_pre_keo))
    sf_keo = tf.cast(sf_keo, dtype=tf.float32)  # convert boolean to float
    sf_keo = tf.reduce_mean(sf_keo) * 100
    d0_keo = tf.reduce_mean(tf.cast(d0_pre_keo, dtype=tf.float32)) * 100
    d1_keo = tf.reduce_mean(tf.cast(d1_pre_keo, dtype=tf.float32)) * 100
    fl_keo = tf.reduce_mean(tf.cast(fl_pre_keo, dtype=tf.float32)) * 100
    return d0_keo, d1_keo, fl_keo, sf_keo, sf_epe

def DeepLiDARFlowNet(batch_images, gt_shape, interp_shape= None):
    with tf.variable_scope('inputs'):
        if interp_shape is not None:
            final_images = [tf.image.resize(batch_images[:, 0, :, :, :], interp_shape, method='nearest'),
                            tf.image.resize(batch_images[:, 1, :, :, :], interp_shape, method='nearest')]
        else:
            final_images = [batch_images[:, 0, :, :, :], batch_images[:, 1, :, :, :]]
        image_list = [final_images[0][:, :, :, :3], final_images[1][:, :, :, :3]]
        disp_list = [tf.expand_dims(final_images[0][:, :, :, 3], axis=3),
                     tf.expand_dims(final_images[1][:, :, :, 3], axis=3)]
        confidence_list = [tf.cast(tf.cast(disp_list[0], dtype=tf.bool), dtype=tf.float32),
                           tf.cast(tf.cast(disp_list[1], dtype=tf.bool), dtype=tf.float32)]

    # Forward pass of the network
    with tf.variable_scope('model'):
        with tf.variable_scope('guidance_network') as scope:
            guide_pyramid = final_network(reg_constant)
            featuresl0, conf_encoder, conf_decoder = guide_pyramid(image_list[0], disp_list[0], confidence_list[0])
            scope.reuse_variables()
            featuresl1, conf_encoder1, conf_decoder1 = guide_pyramid(image_list[1], disp_list[1], confidence_list[1])

        corr_layer = CostVolumeLayer(search_range=4)
        warper = WarpingLayer()

        up_flow, up_feature = None, None
        f, flow = None, None
        flows = []
        for i, [cl1, cl2] in enumerate(zip(featuresl0, featuresl1)):
            is_output = (i == 4)
            with tf.variable_scope('warping_layer_' + str(6 - i)):

                if i == 0:
                    cw1 = cl2
                else:
                    cw1 = warper(cl2, up_flow[:, :, :, :2] * 20. / (2 ** (6 - i)))

            with tf.variable_scope('cost_volume_layer_' + str(6 - i)):

                cv1 = corr_layer(cl1, cw1)
                cv1 = tf.nn.leaky_relu(cv1, 0.1, name='cv1')

            with tf.variable_scope('scene_flow_estimator_' + str(6 - i)):

                concat = [cv1, cl1, cw1]

                if up_flow is not None:
                    concat.append(up_flow)

                if up_feature is not None:
                    concat.append(up_feature)

                concat = tf.concat(concat, axis=-1)

                if i != 4:
                    network = SceneFlowEstimator(str(6 - i), reg_constant, is_output=False)
                    flow, up_flow, up_feature = network(concat)
                    flows.append(flow)

                else:
                    network = SceneFlowEstimator('2', reg_constant, is_output=True)
                    f, flow = network(concat)

        with tf.variable_scope('context_network'):

            context_net = ContextNetwork(reg_constant)
            context_input = tf.concat([f, flow], axis=-1)
            refined_flow = flow + context_net(context_input)

            flows.append(refined_flow)

    out_sf = tf.multiply(20., tf.image.resize(refined_flow, gt_shape, method='bilinear', name='resized_out_sf'), name='out_sf')
    return flows, out_sf
