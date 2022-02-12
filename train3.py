import os
import json
import argparse
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

#tf.test.gpu_device_name()
from models.core.vgg16 import VGG16

from data_utils.data_flow import get_train_test_data
from visualizer.grap_plot import plot_training, plot_training_history

tf.config.run_functions_eagerly(True)
def get_args(args=None):
    def tuple_type(x): return tuple(list(map(int, x.split(','))))
    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('-data_path', '--data_path', type=str, default='/hdd/animals.zip', help='Data directory')
    parser.add_argument('-dst_path', '--destination_path', type=str, default=None, help='Data saved directory')
    parser.add_argument('-s', '--input_shape', type=tuple_type, default=None, help='Size of the input images (height, width, channel), input will be resized to given size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('-sfreq', '--save_freq', type=int, default=5, help='Number of save frequency')
    parser.add_argument('-ifreq', '--info_freq', type=int, default=5, help='Number of information frequency')
    parser.add_argument('-saved', '--saved_weights', type=str, default='./saved_weights/', help='Saved weights model.')
    parser.add_argument('-fc', '--fully_connected', type=list, default=None, help='Fully Connected layers')
    args = parser.parse_args(args)
    return args

def create_folder_weights(saved_dir):
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
    compute_weight = compute_class_weight(class_weight=mode,
                                          classes=np.unique(data.labels),
                                          y=data.labels)

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
    TRAINING_TIME_PATH = create_folder_weights(args.saved_weights)

    train_generator, test_generator = get_train_test_data(args.data_path,
                                                          args.destination_path, 
                                                          args.input_shape, 
                                                          args.batch_size, 
                                                          augmentor=None, 
                                                          normalizer='divide')

    class_names = list(train_generator.classes.keys())

    with open(TRAINING_TIME_PATH + '/class_names.txt', 'w') as filehandle:
        for listitem in class_names:
            filehandle.write('%s\n' % listitem)
    num_class = len(class_names)

    model = VGG16(include_top=True, weights=None, input_shape=(224, 224, 3), classes=num_class)
    model.summary()

    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = Adam(learning_rate=args.learning_rate)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            y_pred = model(images, training=True)
            loss_value = loss(labels, y_pred)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss_value)
        train_accuracy(labels, y_pred)

    @tf.function
    def test_step(images, labels):
        y_pred = model(images, training=False)
        t_loss = loss(labels, y_pred)

        test_loss(t_loss)
        test_accuracy(labels, y_pred)

    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for step, (images, labels) in enumerate(train_generator):
            train_step(images, labels)
            
            if step % args.info_freq == 0:
                print(
                    f'\tstep: {step}/{len(train_generator)}, '
                    f'Train Loss: {train_loss.result():.4f}, '
                    f'Train Accuracy: {train_accuracy.result():.4f}, '
                )

        for test_images, test_labels in test_generator:
            test_step(test_images, test_labels)

        if epoch % args.save_freq == 0:
            model.save_weights(filepath=TRAINING_TIME_PATH + "/epoch-{}".format(epoch), save_format="tf")

        print(
            f'Epoch: {epoch+1}/{args.epochs}, '
            f'Total Train Loss: {train_loss.result():.4f}, '
            f'Total Train Accuracy: {train_accuracy.result():.4f}, '
            f'Total Test Loss: {test_loss.result():.4f}, '
            f'Total Test Accuracy: {test_accuracy.result():.4f}'
        )
        print('-'*50)

    model.save_weights(filepath=TRAINING_TIME_PATH + "saved_model", save_format="tf")

        
if __name__ == "__main__":
    args = get_args()
    train(args)