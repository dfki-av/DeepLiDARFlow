import tensorflow as tf
import numpy as np
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from tf_utils import convolution, upconvolution, max_pool_normalized, upsample, get_shape

def get_features_encoder_rgb(rgb,reg_constant):
        rgb_encoder = []
        filter_by_level = [16,32,64,96,128,196]
        for i,num_filters in zip(range(1,7),filter_by_level):
            if i==1:
                conva = convolution(rgb,num_filters, str(i) + 'a', reg_constant, stride=2)
                convb = convolution(conva,num_filters, str(i) + 'b', reg_constant)
                rgb_encoder.append([conva,convb])
            else:
                conva = convolution(rgb_encoder[-1][1],num_filters, str(i) + 'a', reg_constant, stride=2)
                convb = convolution(conva,num_filters, str(i) + 'b', reg_constant)
                rgb_encoder.append([conva,convb])
        
        return rgb_encoder
    
def get_decoded_features_rgb(images,rgb_encoder,confidences,reg_constant):
    pyr_top = tf.concat([rgb_encoder[-1][1],confidences[0]],axis=-1)
    
    pyr_top = convolution(pyr_top,196,'_pyr_top',reg_constant,ksize = 1,stride=1)
    pyramid = [pyr_top]
    for i in range(4,0,-1):
            feature = rgb_encoder[i][1]
            channels = feature.shape[-1]
            
            #upsampling_lower_level
            upsample = upconvolution(pyramid[-1],channels,'_upsample'+str(i+1) + 'to' + str(i),reg_constant)
         
            #refinement
            pyramid_feature = upsample + feature
            pyramid_feature = tf.concat([pyramid_feature,confidences[5-i]],axis=-1)
            refine = convolution(pyramid_feature,channels,'_refine' + str(i),reg_constant)
            pyramid.append(refine)
    return pyramid

def get_features_encoder_depth(depth,masks,reg_constant):
    depth_encoder = []
    filters_by_level = [16,32,64,96,128,196]
    for i,num_filters in (zip(range(0,6),filters_by_level)):
        if i==0:
            data,conf = max_pool_normalized(depth,masks)
            norm_inst = normal_convolution(num_filters=num_filters,num= str(i) + 'b',input_shape =get_shape(data),reg_constant=reg_constant)
            data,conf = norm_inst(data,conf)
            depth_encoder.append([data,conf])
        else:
            data,conf = depth_encoder[-1]
            data,conf = max_pool_normalized(data,conf)
            norm_inst = normal_convolution(num_filters=num_filters,num= str(i) + 'b',input_shape =get_shape(data),reg_constant=reg_constant)
            data,conf = norm_inst(data,conf)
            depth_encoder.append([data,conf])
    return depth_encoder

def get_decoded_features_depth(raw_depth_and_masks,encoded_depth,reg_constant):
    
    pyr_top_conv = normal_convolution(num_filters=196,reg_constant=reg_constant,input_shape=get_shape(encoded_depth[-1][-1]),num='_pyr_top')
    norm_top,conf_top = pyr_top_conv(encoded_depth[-1][0],encoded_depth[-1][1])
    decoder = [[norm_top,conf_top]]    
    for i in range(4,0,-1):
        norm,conf = encoded_depth[i]
        channels = norm.get_shape().as_list()[-1]
        #upsampling_lower_level
        with tf.variable_scope("upsampling" + str(i+1) + 'to' + str(i)):
            upsample_conv = normal_convolution(num_filters = channels,reg_constant = reg_constant,input_shape = get_shape(decoder[-1][1]),num = str(i+2) + 'to' + str(i+1) )
            upsample_norm,upsample_conf = upsample_conv(decoder[-1][0],decoder[-1][1])
            upsample_norm = upsample(upsample_norm)
            upsample_conf = upsample(upsample_conf)
            
        #refinement
        with tf.variable_scope("refine" + str(i)):
            pyramid_feature_n = upsample_norm + norm
            pyramid_feature_c = upsample_conf + conf
            refine_layer = normal_convolution(num_filters=channels,reg_constant = reg_constant,input_shape = get_shape(pyramid_feature_n),num='refinement')
            refined_n,refined_c = refine_layer(pyramid_feature_n,pyramid_feature_c)
        decoder.append([refined_n,refined_c])
    return decoder

# confidence convolution
class normal_convolution():
    def __init__(self, num_filters, reg_constant, input_shape, num, reuse=True, ksize=3, strides=1, padding='SAME',
                 eps=1e-5, activation='leaky_relu'):
        self.padding = padding
        self.eps = eps
        self.num_filters = num_filters
        self.ksize = ksize
        self.stride = strides
        self.reg_constant = reg_constant
        self.input_shape = input_shape
        self.num = num
        self.kernels = tf.get_variable(name="kernels_" + num, trainable=True,
                                       regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant),
                                       shape=[self.ksize, self.ksize, self.input_shape[3], self.num_filters],
                                       initializer=tf.contrib.layers.xavier_initializer())

        self.activation = activation
        self.biases = tf.Variable(name='biases' + self.num, initial_value=tf.zeros(shape=[1, 1, 1, num_filters]),
                                  trainable=True)

    def __call__(self, data, confidence):
        with tf.variable_scope("computations"):
            filters = tf.nn.softplus(self.kernels)

            denominator = tf.nn.conv2d(confidence, filters, strides=[1, self.stride, self.stride, 1],
                                       padding=self.padding, name='denominator')
            for_normalization = tf.cast(confidence > 0., dtype=tf.float32)
            for_normalization = tf.nn.conv2d(confidence, filters, strides=[1, self.stride, self.stride, 1],
                                             padding=self.padding, name='normalization')
            for_normalization += self.eps

            numerator = tf.nn.conv2d(data * confidence, filters, strides=[1, self.stride, self.stride, 1],
                                     padding=self.padding, name='numerator')
            nconv = numerator / (denominator + self.eps)
            nconv += self.biases
            if self.activation != 'leaky_relu':
                nconv = tf.keras.layers.Activation(self.activation)(nconv)
            else:
                nconv = tf.nn.leaky_relu(nconv)
            cout = denominator / tf.reshape(tf.reduce_sum(filters, axis=(0, 1, 2)), shape=[1, 1, 1, self.num_filters])
        return nconv, cout

class final_network():
    def __init__(self,reg_constant):
        self.reg_constant = reg_constant
    
    def __call__(self,images,depth,masks):
        with tf.variable_scope("rgb_features_encoder_decoder"):
            rgb_encoder = get_features_encoder_rgb(images,self.reg_constant)
        
        with tf.variable_scope("rgb_guidance_connections"):
            depth_encoder = get_features_encoder_depth(depth,masks,self.reg_constant)
            
        with tf.variable_scope("final_scene_flow_tensors"):
            for_scene_flow = []
            confidences = []
            decoded_features_depth = get_decoded_features_depth([depth,masks],depth_encoder,self.reg_constant)
            for [n,c] in (decoded_features_depth):
                confidences.append(c)

            decoded_features_rgb = get_decoded_features_rgb(images,rgb_encoder,confidences,self.reg_constant)
            norm_data = []


            #conf_encoder and conf_decoder are used for confidence visualizations, if not required, they can be left
            conf_encoder = []
            conf_decoder = []
            for norm,conf in depth_encoder:
                conf_encoder.append(conf)

            #the fusion module
            #this is for depth preprocessing
            for i,[norm,conf] in enumerate(decoded_features_depth):
                conv1 = convolution(norm,16,'depth_refine_a' + str(i),self.reg_constant)
                conv2 = convolution(conv1,16,'depth_refine_b' + str(i),self.reg_constant)
                conv3 = convolution(conv2,16,'depth_refine_c' + str(i),self.reg_constant)
                conv4 = convolution(conv3,16,'depth_refine_d' + str(i),self.reg_constant)

                norm_data.append(conv4)
                conf_decoder.append(conf)
            
            for i,f1,f2 in zip(range(0,len(decoded_features_rgb)),decoded_features_rgb,norm_data):
                final_features = tf.concat([f1,f2],axis=3)
                conv1 = convolution(final_features,64,'concat_refine_a' + str(i),self.reg_constant)
                conv2 = convolution(conv1,64,'concat_refine_b' + str(i),self.reg_constant)
                conv3 = convolution(conv2,64,'concat_refine_c' + str(i),self.reg_constant)
                conv4 = convolution(conv3,64,'concat_refine_d' + str(i),self.reg_constant)

                for_scene_flow.append(conv4)
                
        return for_scene_flow,conf_decoder,conf_encoder