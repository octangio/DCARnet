import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Conv2DTranspose, concatenate


def conv2d_bn(input_tensor, filters, kernel_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), use_bias=True,
              kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001), activation=None, name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:

        conv_name = name + '_conv2d'
        conv2d_name = name + '_conv2d'
        act_name = name + '_atv'
    else:
        conv_name = 'conv2d'
        act_name = None
        conv2d_name = None
    with tf.name_scope(name=conv_name):
        xi = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,
            name=conv2d_name)(input_tensor)
        if activation is not None:
            xi = layers.Activation(activation, name=act_name)(xi)
    return xi


def block20(x0):
    for i in range(20):
        x = conv2d_bn(x0, 64, (3, 3), padding='same', strides=1, activation='relu')
        x0 = x
    return x0


def block15(x0):
    for i in range(15):
        x = conv2d_bn(x0, 64, (3, 3), padding='same', strides=1, activation='relu')
        x0 = x
    return x0


def block10(x0):
    for i in range(10):
        x = conv2d_bn(x0, 64, (3, 3), padding='same', strides=1, activation='relu')
        x0 = x
    return x0


def dcarnet(x):
    x1 = conv2d_bn(x, 128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
    bypass1_b1 = block20(x1)

    bypass2_x1 = conv2d_bn(x, 128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')
    bypass2_b1 = block15(bypass2_x1)
    bypass2_decov = Conv2DTranspose(128, 2, strides=2, use_bias=True)(bypass2_b1)
    bypass2_decov = tf.keras.layers.Activation('relu')(bypass2_decov)

    bypass3_x1 = conv2d_bn(x, 128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')
    bypass3_x2 = conv2d_bn(bypass3_x1, 128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')
    bypass3_x3 = block10(bypass3_x2)
    bypass3_decov1 = Conv2DTranspose(128, 2, strides=2)(bypass3_x3)
    bypass3_decov1 = tf.keras.layers.Activation('relu')(bypass3_decov1)
    bypass3_decov2 = Conv2DTranspose(128, 2, strides=2)(bypass3_decov1)
    bypass3_decov2 = tf.keras.layers.Activation('relu')(bypass3_decov2)

    feature_fusion_all = concatenate([bypass1_b1, bypass2_decov, bypass3_decov2])

    out = conv2d_bn(feature_fusion_all, 1, kernel_size=(3, 3), strides=1, padding='same')
    return out
