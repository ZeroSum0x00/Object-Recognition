from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from matplotlib import pyplot as plt
import numpy as np

'''
cls_train = generator_train.classes
cls_test = generator_test.classes
class_names = list(generator_train.class_indices.keys())
num_classes = generator_train.num_classes
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

steps_per_epoch = generator_train.n / batch_size 
steps_test = generator_test.n / batch_size
'''

def create_data_generators(input_shape, batch_size, 
                            train_dir, test_dir, 
                            save_augumented=False, plot_imgs = False):
    datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20.,
      width_shift_range=0.1,
      height_shift_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

    # datagen_train = ImageDataGenerator(rescale=1./255)

    datagen_test = ImageDataGenerator(rescale=1./255)

    if save_augumented:
        save_to_dir='augmented_images/'
    else:
        save_to_dir = None

    generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)   
                                        
    generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)


    # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
    # for i in range(4):
    #     # convert to unsigned integers for plotting
    #     image = next(generator_train)[0].astype('uint8')
    #     print(image.shape)
    #     # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
    #     image = np.squeeze(image)
    #     image = image[i]
    #     # plot raw pixel data
    #     ax[i].imshow(image)
    #     ax[i].axis('off')


    return generator_train, generator_test


if __name__ == "__main__":
    # data
    #local
    TRAIN_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/train'
    TEST_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/test'
    create_data_generators((224,224,3), 20, 
                            TRAIN_DIR, TEST_DIR, 
                            save_augumented=None, plot_imgs = True)
    