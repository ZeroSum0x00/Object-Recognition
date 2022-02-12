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
from models.core.xception import Xception
# from models.finetune_model import finetune_mobilenet
# from models.finetune_model import finetune_xception

from losses.focal_loss import Focal_Loss

from data_utils.data_flow import get_train_test_data
from visualizer.grap_plot import plot_training, plot_training_history


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
                                                          augmentor='mixing_ver2', 
                                                          normalizer='divide')

    class_names = list(train_generator.classes.keys())

    with open(TRAINING_TIME_PATH + '/class_names.txt', 'w') as filehandle:
        for listitem in class_names:
            filehandle.write('%s\n' % listitem)
    num_class = len(class_names)

    # model = finetune_mobilenet(transfer_layer=-1, fc_layer=args.fully_connected, num_classes=num_class)
    model = Xception(include_top=True, weights=None, input_shape=(224, 224, 3), classes=num_class)
    model.summary()
    # model_shape = model.layers[0].output_shape[0][1:3]
    # transfer_layer = model.get_layer(index=-1)

    optimizer = Adam(learning_rate=args.learning_rate)
    # loss = Focal_Loss(alpha=1)
    loss = 'categorical_crossentropy'
    # categorical_accuracy
    model.compile(optimizer, loss=loss, metrics=['accuracy', 'categorical_accuracy'])

    class_weight = balance_class_weights(data=train_generator, 
                                        class_name=class_names, 
                                        mode='balanced', 
                                        verbose=True)

    # # CHECKPOINTS
    SAVED_PATH = TRAINING_TIME_PATH + '/weights.best.hdf5'
    # checkpoint = ModelCheckpoint(SAVED_PATH, 
    #                              monitor='val_acc',
    #                              verbose=1, 
    #                             #  save_weights_only=True, 
    #                              save_best_only=True, 
    #                              mode='max')
    checkpoint = ModelCheckpoint(SAVED_PATH, 
                                 monitor='val_acc',
                                 verbose=1, 
                                 save_weights_only=False, 
                                 save_best_only=False, 
                                 mode='max',
                                 save_freq="epoch")
    
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
                        epochs=args.epochs,
                        steps_per_epoch=len(train_generator),
                        class_weight=class_weight,
                        validation_data=test_generator,
                        validation_steps=len(test_generator),
                        shuffle=True,
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

    args = get_args()
    train(args)
