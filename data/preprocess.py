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

    ltxs = cxs - sizes[:, 0] * 0.5
    ltys = cys - sizes[:, 1] * 0.5
    rbxs = cxs + sizes[:, 0] * 0.5
    rbys = cys + sizes[:, 1] * 0.5

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


def test_random_scale_shift_bboxes():
    bboxes = tf.constant([[10, 20, 30, 50]], dtype=tf.float32)
    scale = 1.
    shift = 0.

    bb2 = random_scale_shift_bboxes(bboxes, scale, shift)

    print(bb2)


def test_random_crop_resize():
    import cv2
    import numpy as np
    from util.io import load_image
    from util.bb import draw_bb

    path_image = "../dataset/v47/Cam_A/Images/in/1.01/1.bmp"
    img = load_image(path_image)
    img = tf.expand_dims(img, axis=0)

    bboxes = tf.constant([240, 112, 320, 376], dtype=tf.float32)
    resized_img, crop_bboxes = random_crop_and_resize(img, bboxes, 1., 0., [192, 512])

    print(crop_bboxes)

    cvimg = img.numpy()[0, :, :, ::-1].copy()
    cv_resized_img = resized_img.numpy()[0, :, :, ::-1].copy()

    cvimg = draw_bb(cvimg, crop_bboxes[0, :], color=(0, 255, 0))
    cv2.imshow("img", cvimg)
    cv2.imshow("rimg", cv_resized_img.astype(np.uint8))

    cv2.waitKey()





if __name__ == "__main__":
    # test_random_scale_shift_bboxes()
    test_random_crop_resize()