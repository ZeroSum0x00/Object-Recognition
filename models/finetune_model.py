import os
import json
import PIL
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout,GlobalAveragePooling2D,Input
from tensorflow.keras.layers import Reshape, Conv2D, Activation
#from tensorflow.python.keras.applications import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

from .core.xception import Xception
from .core.vgg16 import VGG16
from .core.vgg19 import VGG19
from .core.mobilenet import MobileNet

def finetune_vgg16(transfer_layer, x_trainable='all', fc_layer=None, num_classes=1000, new_weights=None):
    base_model = VGG16(include_top=False, weights = 'imagenet', input_shape=(224,224,3))
    get_transfer = base_model.get_layer(index=transfer_layer)

    # number of freezed layers: 0 (by a default)
    freezed_layers = 0
    if x_trainable != "all":
        if x_trainable == 0:
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers = freezed_layers + 1
        else:
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers = freezed_layers + 1  
    all_layers = len(base_model.layers)
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freezed_layers))
    # adding custom layers to the classification part of a model

    x = get_transfer.output
    if fc_layer is not None:
        for fc in fc_layer:
            x = fc(x)
    else:
        x = Flatten()(x)
        for fc in fc_layers:
            # New FC layer, random init
            x = Dense(fc, activation='relu')(x) 
            x = Dropout(0.4)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)  
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights is not None:
        finetune_model.load_weights(new_weights)
    return finetune_model

def finetune_vgg19(transfer_layer, x_trainable='all', fc_layer=None, num_classes=1000, new_weights=None):
    base_model = VGG19(include_top=False, weights = 'imagenet', input_shape=(224,224,3))
    get_transfer = base_model.get_layer(index=transfer_layer)

    # number of freezed layers: 0 (by a default)
    freezed_layers = 0
    if x_trainable != "all":
        if x_trainable == 0:
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers = freezed_layers + 1
        else:
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers = freezed_layers + 1  
    all_layers = len(base_model.layers)
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freezed_layers))
    # adding custom layers to the classification part of a model

    x = get_transfer.output
    if fc_layer is not None:
        for fc in fc_layer:
            x = fc(x)
    else:
        x = Flatten()(x)
        for fc in fc_layers:
            # New FC layer, random init
            x = Dense(fc, activation='relu')(x) 
            x = Dropout(0.4)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)  
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights is not None:
        finetune_model.load_weights(new_weights)
    return finetune_model

def finetune_mobilenet(transfer_layer, x_trainable='all', fc_layer=None, num_classes=1000, new_weights=None):
    base_model = MobileNet(include_top=False, weights = 'imagenet', input_shape=(224,224,3))
    get_transfer = base_model.get_layer(index=transfer_layer)

    # number of freezed layers: 0 (by a default)
    freezed_layers = 0
    if x_trainable != "all":
        if x_trainable == 0:
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers = freezed_layers + 1
        else:
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers = freezed_layers + 1    
    all_layers = len(base_model.layers)
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freezed_layers))
    # adding custom layers to the classification part of a model

    x = get_transfer.output
    if fc_layer is not None:
        for fc in fc_layer:
            x = fc(x)
    else:
        shape = (1, 1, int(1024 * 1.0))
        x = GlobalAveragePooling2D(name='average_pooling')(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(0.3, name='dropout')(x)
        x = Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        

    predictions = Reshape((num_classes,), name='reshape_2')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights is not None:
        finetune_model.load_weights(new_weights)
    return finetune_model

def finetune_xception(transfer_layer, x_trainable='all', fc_layer=None, num_classes=1000, new_weights=None):
    base_model = Xception(include_top=False, weights = 'imagenet', input_shape=(299,299,3))
    get_transfer = base_model.get_layer(index=transfer_layer)

    # number of freezed layers: 0 (by a default)
    freezed_layers = 0
    if x_trainable != "all":
        if x_trainable == 0:
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers = freezed_layers + 1
        else:
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers = freezed_layers + 1    
    all_layers = len(base_model.layers)
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freezed_layers))
    # adding custom layers to the classification part of a model

    x = get_transfer.output
    if fc_layer is not None:
        for fc in fc_layer:
            x = fc(x)
    else:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.4)(x)

    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights is not None:
        finetune_model.load_weights(new_weights)
    return finetune_model

if __name__ == "__main__":
    pass
