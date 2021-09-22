from tensorflow.keras import backend as K
import tensorflow as tf


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def MSE(y_true, y_pred):
    MSE_loss = tf.reduce_mean(K.mean(K.square(y_pred - y_true), axis=-1))
    return MSE_loss


def ssim(y_true, y_pred):
    SSIM = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return SSIM


def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 1.0)
    dessim = 1 - ssim
    return dessim


def contrast(y_true, y_pred):
    std_pred = tf.math.reduce_std(y_pred)
    std_true = tf.math.reduce_std(y_true)
    contrast2 = abs(std_true - std_pred)
    return contrast2


def total_loss(y_true, y_pred):
    contrast_loss = contrast(y_true, y_pred)
    MSE_loss = MSE(y_true, y_pred)
    SSIM_loss = ssim_loss(y_true, y_pred)
    add_loss = MSE_loss + SSIM_loss + contrast_loss
    return add_loss
