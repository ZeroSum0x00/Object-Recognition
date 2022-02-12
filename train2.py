import os
import json
import shutil
import pickle
import argparse
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

#tf.test.gpu_device_name()
from models.core.vgg16 import VGG16
from models.core.vgg19 import VGG19
from models.core.xception import Xception
from models.core.mobilenet import MobileNet

from models.finetune_model import finetune_vgg16
from models.finetune_model import finetune_vgg19
from models.finetune_model import finetune_mobilenet
from models.finetune_model import finetune_xception
# from data_utils.data_flow import get_train_test_data
from data_utils.create_datagenerators import create_data_generators

from visualizer.grap_plot import plot_training, plot_training_history

TRAIN_DIR = '/content/sample_data/small_cars/train'
TEST_DIR = '/content/sample_data/small_cars/validation'

DATA_ZIP = "/content/drive/MyDrive/Object Recognition/car_recognition/datasets/animals.zip"
DATA_DST = "/content/sample_data/"
SAVE_RESULRS_DIR = '/content/sample_data/saved_weights/'
EPOCHS = 50
BATCHSIZE = 8
NEW_WEIGHTS = ""
LEARN_RATE = 0.0001


# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('-data_type', '--data_type', type=str, default=None, help='Training dataset.')
parser.add_argument('-data_dir', '--data_dir', type=str, default='/hdd/nyu_data.zip', help='Data directory')
parser.add_argument('-dst_dir', '--dst_dir', type=str, default=None, help='Data saved directory')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default=None, help='Start training from an existing model.')
parser.add_argument('--saved_weights', type=str, default='./save_weights/', help='Saved weights model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
args = parser.parse_args()


def create_folder_with_results(saved_dir):
    now = datetime.datetime.now()
    training_time = now.strftime("%Y%m%d_%H%M")
    # name of dir due to today date
    TRAINING_TIME_PATH = saved_dir + training_time
    access_rights = 0o755
    try:  
        os.makedirs(TRAINING_TIME_PATH, access_rights)
        print ("Successfully created the directory %s" % TRAINING_TIME_PATH)
        return TRAINING_TIME_PATH
    except: 
        print ("Creation of the directory %s failed" % TRAINING_TIME_PATH)


def balance_class_weights(data, class_name, mode='balanced', verbose=False):
    # compute_weight = compute_class_weight(class_weight=mode,
    #                                       classes=np.unique(data.labels),
    #                                       y=data.labels)
    compute_weight = compute_class_weight(class_weight=mode,
                                          classes=np.unique(data.classes),
                                          y=data.classes)

    class_weight = {}
    show_class_weight = {}
    for idx, weights in enumerate(compute_weight):
        class_weight[idx] = weights
        show_class_weight[idx] = [class_name[idx], weights]

    print('Training used %s class weights!' %mode)
    if verbose:
        print('{}'.format('-'*85))
        print ("{:<6}|{:<57}|{:<30}".format('  ID ','   Class Name','   Balance Percent'))
        print('{:<6}+{:<57}+{:<25}'.format('-'*6, '-'*57, '-'*20))
        for key, value in show_class_weight.items():
            name, percent = value
            print (" {:<4} | {:<56}|   {:<5.2f} %".format(key, name, percent*100))
        print('{}'.format('-'*85))
    return class_weight


def train(args):
    TRAINING_TIME_PATH = create_folder_with_results(SAVE_RESULRS_DIR)
    input_shape = (224, 224, 3)

    # train_generator, test_generator = get_train_test_data(DATA_ZIP, DATA_DST, (input_shape[0], input_shape[1], input_shape[2]), BATCHSIZE)
    train_generator, test_generator = create_data_generators((input_shape[0], input_shape[1]), BATCHSIZE, 
                            TRAIN_DIR, TEST_DIR, 
                            save_augumented=None, plot_imgs=False)

    # class_names = list(train_generator.classes.keys())
    class_names = list(train_generator.class_indices.keys())

    with open(TRAINING_TIME_PATH + '/class_names.txt', 'w') as filehandle:
        for listitem in class_names:
            filehandle.write('%s\n' % listitem)
    num_class = len(class_names)

    model = finetune_mobilenet(transfer_layer=-1, fc_layer=args, num_classes=num_class)
    # model_shape = model.layers[0].output_shape[0][1:3]
    # transfer_layer = model.get_layer(index=-1)

    optimizer = Adam(learning_rate=LEARN_RATE)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

    class_weight = balance_class_weights(data=train_generator, 
                                        class_name=class_names, 
                                        mode='balanced', 
                                        verbose=True)

    # # CHECKPOINTS
    SAVED_PATH = TRAINING_TIME_PATH + '/weights.best.hdf5'
    checkpoint = ModelCheckpoint(SAVED_PATH, 
                                 monitor='val_acc',
                                 verbose=1, 
                                #  save_weights_only=True, 
                                 save_best_only=True, 
                                 mode='max')
    
    early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)

    callbacks_list = [checkpoint, early_stop]

    # Fit the model - train
    #save model architecture
    NEW_MODEL_PATH_STRUCTURE = TRAINING_TIME_PATH+'/model.json'
    # # serialize model to JSON
    model_json = model.to_json()
    with open(NEW_MODEL_PATH_STRUCTURE, "w") as json_file:
        json_file.write(model_json)

    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_generator.n // BATCHSIZE,
                        class_weight=class_weight,
                        validation_data=test_generator,
                        validation_steps=test_generator.n // BATCHSIZE,
                        callbacks=callbacks_list)

    model.save(SAVED_PATH)

    with open(TRAINING_TIME_PATH +'/history.txt', 'w') as f:  
        f.write(str(history.history))

    with open(TRAINING_TIME_PATH +'/model_summary.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    plot_training(history, 
        TRAINING_TIME_PATH + '/acc_vs_epochs.png', 
        TRAINING_TIME_PATH + '/loss_vs_epochs.png')

    plot_training_history(history,
        TRAINING_TIME_PATH + '/train_test_accuracy.png')
        
if __name__ == "__main__":
    # args = [
		#     GlobalAveragePooling2D(name='avg_pool'),
    #     Dropout(0.2),
    #     Dense(units=1024, name='dense1'),
    #     Dropout(0.4)
    # ]
    
    # args = [
    #     Flatten(name='flatten'),
    #     Dense(4096, activation='relu', name='fc1'),
    #     Dropout(0.2),
    #     Dense(4096, activation='relu', name='fc2'),
    #     Dropout(0.4)
    # ]

    args = None
    train(args)
