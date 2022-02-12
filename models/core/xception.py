# -*- coding: utf-8 -*-
'''Xception V1 model for tensorflow.keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790.
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.

# Reference:

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

'''
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K
from .utils import _obtain_input_shape

TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Xception(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000,
             debug=False,
             summary=False):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    """ 
        Entry flow
    """
    # Block 1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_batchnorm1')(x)
    x = Activation('relu', name='block1_activation1')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_batchnorm2')(x)
    x = Activation('relu', name='block1_activation2')(x)
    # print('Shape block 1: ', x.shape)

    residual = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='entry_conv_residual1')(x)
    residual = BatchNormalization(name='entry_batchnorm_residual1')(residual)

    # Block2
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False, name='block2_separable1')(x)
    x = BatchNormalization(name='block2_batchnorm1')(x)

    x = Activation('relu', name='block2_activation1')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False, name='block2_separable2')(x)
    x = BatchNormalization(name='block2_batchnorm2')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block2_maxpooling')(x)

    x = add([x, residual], name='merge_1')
    # print('Shape block 2: ', x.shape)

    # Block3
    residual = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='entry_conv_residual2')(x)
    residual = BatchNormalization(name='entry_batchnorm_residual2')(residual)

    x = Activation('relu', name='block3_activation1')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False, name='block3_separable1')(x)
    x = BatchNormalization(name='block3_batchnorm1')(x)

    x = Activation('relu', name='block3_activation2')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False, name='block3_separable2')(x)
    x = BatchNormalization(name='block3_batchnorm2')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block3_maxpooling')(x)

    x = add([x, residual], name='merge_2')
    # print('Shape block 3: ', x.shape)

    # Block4
    residual = Conv2D(filters=728, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='entry_conv_residual3')(x)
    residual = BatchNormalization(name='entry_batchnorm_residual3')(residual)

    x = Activation('relu', name='block4_activation1')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block4_separable1')(x)
    x = BatchNormalization(name='block4_batchnorm1')(x)

    x = Activation('relu', name='block4_activation2')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block4_separable2')(x)
    x = BatchNormalization(name='block4_batchnorm2')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block4_maxpooling')(x)

    x = add([x, residual], name='merge_3')
    # print('Shape block 4: ', x.shape)

    """ 
        Middle flow
    """ 
    # Block 5 - 12
    for i in range(8):
        prefix = 'block' + str(i + 5) + '_'
        residual = x

        x = Activation('relu', name=prefix + 'activation1')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + 'separable1')(x)
        x = BatchNormalization(name=prefix + 'batchnorm1')(x)

        x = Activation('relu', name=prefix + 'activation2')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + 'separable2')(x)
        x = BatchNormalization(name=prefix + 'batchnorm2')(x)

        x = Activation('relu', name=prefix + 'activation3')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + 'separable3')(x)
        x = BatchNormalization(name=prefix + 'batchnorm3')(x)

        x = add([x, residual], name='merge_' + str(i + 5))
        # print('Shape block {}: {}'.format(i+5, x.shape))

    """ 
        Exit flow
    """ 
    # Block 13
    residual = Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='exit_conv_residual')(x)
    residual = BatchNormalization(name='exit_batchnorm_residual')(residual)

    x = Activation('relu', name='block13_activation1')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block13_separable1')(x)
    x = BatchNormalization(name='block13_batchnorm1')(x)

    x = Activation('relu', name='block13_activation2')(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False, name='block13_separable2')(x)
    x = BatchNormalization(name='block13_batchnorm2')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block13_maxpooling')(x)

    x = add([x, residual], name='merge_13')
    # print('Shape block 13: ', x.shape)

        # Block 14
    x = SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same', use_bias=False, name='block14_separable1')(x)
    x = BatchNormalization(name='block14_batchnorm1')(x)
    x = Activation('relu', name='block14_activation1')(x)

    x = SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same', use_bias=False, name='block14_separable2')(x)
    x = BatchNormalization(name='block14_batchnorm2')(x)
    x = Activation('relu', name='block14_activation2')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='Xception')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    if summary:
      model.summary()

    return model


if __name__ == '__main__':
    model = Xception()
