import tensorflow as tf


def power2_radom_uniform(shape, maxval, minval):
    rands = tf.random.uniform(shape=shape, maxval=maxval, minval=minval)
    power2rands = tf.pow(tf.constant(2., dtype=tf.float32), rands)

    return power2rands


def random_scale_shift_bboxes(bboxes, scale, shift):
    cxs = 0.5 * tf.math.reduce_sum(bboxes[:, ::2], axis=1, keepdims=True)
    cys = 0.5 * tf.math.reduce_sum(bboxes[:, 1::2], axis=1, keepdims=True)
    ws = bboxes[:, 2:3] - bboxes[:, 0:1]
    hs = bboxes[:, 3:4] - bboxes[:, 1:2]
    sizes = tf.concat([ws, hs], axis=1)
    scales = power2_radom_uniform(shape=sizes.shape, maxval=scale, minval=-scale)
    sizes = sizes * scales

    centers = tf.concat([cxs, cys], axis=1)
    shifts = tf.random.uniform(shape=sizes.shape, maxval=shift, minval=-shift)
    centers += shifts

    half_sizes = 0.5 * sizes

    ltxs = centers[:, 0:1] - half_sizes[:, 0:1]
    ltys = centers[:, 1:] - half_sizes[:, 1:]
    rbxs = centers[:, 0:1] + half_sizes[:, 0:1]
    rbys = centers[:, 1:] + half_sizes[:, 1:]

    dst = tf.concat([ltxs, ltys, rbxs, rbys], axis=1)

    return dst


def random_crop_and_resize(image, bbox, scale, shift, out_size):
    bboxes = tf.expand_dims(bbox, axis=0)
    bboxes = random_scale_shift_bboxes(bboxes, scale, shift)
    ltxs = bboxes[:, 0:1]
    ltys = bboxes[:, 1:2]
    rbxs = bboxes[:, 2:3]
    rbys = bboxes[:, 3:4]
    imshape = tf.dtypes.cast(tf.shape(image), tf.float32)
    imw = imshape[2] - 1.
    imh = imshape[1] - 1.
    nltxs = ltxs / imw
    nltys = ltys / imh
    nrbxs = rbxs / imw
    nrbys = rbys / imh
    bbs = tf.concat([nltys, nltxs, nrbys, nrbxs], axis=1)
    print(bbs)
    cropped_img = tf.image.crop_and_resize(image=image,
                                           boxes=bbs,
                                           box_indices=tf.constant([0]),
                                           crop_size=out_size[::-1])

    return cropped_img, bboxes


