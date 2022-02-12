import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from data_utils.data_processing import get_data, Augmentor, Normalizer


def get_train_test_data(data_zipfile, dst_dir, target_size, batch_size, augmentor=None, normalizer='divide'):
    data_train, data_test, class_names = get_data(data_zipfile, dst_dir)
    train_generator = Train_Data_Sequence(data_train, 
                                          target_size=target_size, 
                                          batch_size=batch_size, 
                                          classes=class_names, 
                                          augmentor=augmentor,
                                          normalizer=normalizer)
    test_generator = Test_Data_Sequence(data_test, 
                                        target_size=target_size, 
                                        batch_size=batch_size, 
                                        classes=class_names,
                                        normalizer=normalizer)

    return train_generator, test_generator


class Train_Data_Sequence(Sequence):
    def __init__(self, dataset, target_size, batch_size, classes, augmentor=None, normalizer=None):
        # data
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = (batch_size, target_size[0], target_size[1], target_size[2])
        
        # label
        self.classes = classes
        self.num_class = len(self.classes)
        
        # shuffle
        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)
        self.labels = [label for _, label in self.dataset]

        # augmentor
        if isinstance(augmentor, str):
            self.augmentor = Augmentor(aug_mode=augmentor)
        else:
            self.augmentor = augmentor

        # nomalizer image
        self.normalizer = Normalizer(normalizer)

        # N iter
        self.N = self.n = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = np.zeros(self.target_size)
        batch_y = np.zeros((self.batch_size, self.num_class))

        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = cv2.imread(sample[0], 1)

            if self.augmentor is not None:
                x = self.augmentor(x)

            x = self.normalizer(x, 
                                target_size=self.target_size[1:], 
                                interpolation=cv2.INTER_NEAREST)
            batch_x[i] = x

            y = sample[1]
            batch_y[i][y] = 1

        return batch_x, batch_y


class Test_Data_Sequence(Sequence):
    def __init__(self, dataset, target_size, batch_size, classes, normalizer):
        # data
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = (batch_size, target_size[0], target_size[1], target_size[2])
        
        # label
        self.classes = classes
        self.num_class = len(self.classes)
        self.labels = [label for _, label in self.dataset]
        
        # nomalizer image
        self.normalizer = Normalizer(normalizer)
        
        # N iter
        self.N = self.n = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = np.zeros(self.target_size)
        batch_y = np.zeros((self.batch_size, self.num_class))

        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = cv2.imread(sample[0], 1)

            x = self.normalizer(x, 
                                target_size=self.target_size[1:], 
                                interpolation=cv2.INTER_NEAREST)
            batch_x[i] = x

            y = sample[1]
            batch_y[i][y] = 1

        return batch_x, batch_y