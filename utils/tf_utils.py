import tensorflow as tf



def convolution(inp, filters, num, reg_constant, ksize=3, stride=1, pad='SAME', dilation=1, activation='leaky_relu'):

    if activation is not None:
        activation = lambda x: tf.nn.leaky_relu(x, 0.1, name='ReLU' + num)

    return tf.layers.conv2d(inp, filters=filters, kernel_size=(ksize, ksize), strides=(stride, stride), padding=pad,
                            activation=activation, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_constant),
                            dilation_rate=(dilation, dilation),
                            name='conv' + num, reuse=tf.AUTO_REUSE)


# Utility upconvolution function to call in other modules
def upconvolution(inp, filters, num, reg_constant, ksize=4, stride=2, pad='SAME', activation='leaky_relu'):

    if activation is not None:
        activation = lambda x: tf.nn.leaky_relu(x, 0.1, name='ReLU' + num)

    return tf.layers.conv2d_transpose(inp, filters=filters, kernel_size=(ksize, ksize), strides=(stride, stride), padding=pad,
                                      activation=activation, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_constant),
                                      name='deconv' + num, reuse=tf.AUTO_REUSE)


def down_sample(inp,size):
    inp = tf.layers.max_pooling2d(inp,pool_size = (size,size),strides = (size,size))
    return inp

def get_shape(tensor):
    return tensor.get_shape().as_list()

#pool according to the confidence values
def max_pool_normalized(data,confidence,ksize=2,stride=2):
    pooled_conf,conf_max_indices = tf.nn.max_pool_with_argmax(confidence,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding='VALID',include_batch_in_index=True)
    batch = tf.shape(data)[0]
    height = tf.shape(data)[1]
    width = tf.shape(data)[2]
    channels = tf.shape(data)[-1]
    data = tf.reshape(data,shape = [batch*height*width*channels])
    pooled_data = tf.gather(data,conf_max_indices)
    return pooled_data,pooled_conf

#nearest neighbour interpolation
def upsample(data):
    h = tf.shape(data)[1]
    w = tf.shape(data)[2]
    data = tf.image.resize_nearest_neighbor(data,(2*h,2*w))
    return data


