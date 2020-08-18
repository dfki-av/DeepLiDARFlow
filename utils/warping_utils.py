import tensorflow as tf

# Warping layer ----------------------------
def get_grid(x):

    batch_size, height, width, filters = tf.unstack(tf.shape(x))

    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width), indexing='ij')

    return Bg, Yg, Xg


def nearest_warp_1d(x, disp):

    grid_b, grid_y, grid_x = get_grid(x)
    disp = tf.cast(disp, tf.int32)

    warped_gx = tf.add(grid_x, disp)

    w = tf.unstack(tf.shape(x))[2]
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)

    warped_indices = tf.stack([grid_b, grid_y, warped_gx], axis=3)

    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x


def nearest_warp_2d(x, flow):

    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:, :, :, 1])
    warped_gx = tf.add(grid_x, flow[:, :, :, 0])

    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)

    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis=3)

    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x


def bilinear_warp_1d(x, disp):

    w = tf.unstack(tf.shape(x))[2]
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    d0 = tf.floor(disp)
    d1 = d0 + 1

    # warping indices
    w_lim = tf.cast(w-1, tf.float32)
    gx_0 = tf.clip_by_value(grid_x + d0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + d1, 0., w_lim)

    g0 = tf.cast(tf.stack([grid_b, grid_y, gx_0], axis=3), tf.int32)
    g1 = tf.cast(tf.stack([grid_b, grid_y, gx_1], axis=3), tf.int32)

    # gather contents
    x0 = tf.gather_nd(x, g0)
    x1 = tf.gather_nd(x, g1)

    # coefficients
    c0 = tf.expand_dims((d1 - disp), axis=3)
    c1 = tf.expand_dims((disp - d0), axis=3)

    return c0*x0 + c1*x1


def bilinear_warp_2d(x, flow):

    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis=-1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0 + 1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0 + 1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)

    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis=3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis=3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis=3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis=3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis=3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis=3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis=3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis=3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11