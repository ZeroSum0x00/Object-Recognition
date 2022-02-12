import os
import cv2
import numpy as np
from glob import glob
from utils.files import extract_zip, verify_folder
from data_utils.augmentation import Available_Augmentation


def get_data(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar']

    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = verify_folder(data_destination) + folder_name 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
    train_path = verify_folder(data_destination) + 'train/'
    val_path = verify_folder(data_destination) + 'validation/'
        
    class_names = {path.split("/")[-1]:idx for idx, path in enumerate(glob(train_path + '*'))}

    train_list_file = []
    val_list_file = []
    for idx, name in enumerate(class_names.keys()):
        files_train = os.listdir('%s/%s' %(train_path, name))
        files_train = [[train_path + name + '/' + file_name, idx] for file_name in files_train]
        train_list_file.extend(files_train)
        
        files_val = os.listdir('%s/%s' %(val_path, name))
        files_val = [[val_path + name + '/' + file_name, idx] for file_name in files_val]
        val_list_file.extend(files_val)

    return train_list_file, val_list_file, class_names


class Normalizer():
    def __init__(self, mode="divide"):
        self.mode = mode
    
    def _sub_divide(self, img, target_size, interpolation=None):
        img = cv2.resize(img, (target_size[0], target_size[1]), interpolation=interpolation)
        img = img.astype(np.float32)
        img = img / 127.5 - 1
        img = np.clip(img, 0, 1)
        return img

    def _divide(self, img, target_size, interpolation=None):
        img = cv2.resize(img, (target_size[0], target_size[1]), interpolation=interpolation)
        img = img.astype(np.float32)
        img = img / 255.0
        img = np.clip(img, 0, 1)
        return img

    def _basic(self, img, target_size, interpolation=None):
        img = cv2.resize(img, (target_size[0], target_size[1]), interpolation=interpolation)
        return img

    def __call__(self, input, *args, **kargs):
        if self.mode == "divide":
            return self._divide(input, *args, **kargs)
        elif self.mode == "sub_divide":
            return self._sub_divide(input, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)


class Augmentor():
    def __init__(self, aug_mode="mixing_ver2"):
        self.augmentor = self._load_augmentation(aug_mode)
      
    def _load_augmentation(self, aug_mode):
        augmentation = Available_Augmentation()
        augmentation_functions = {
            "geometric": augmentation.aug_geometric,
            "non_geometric": augmentation.aug_non_geometric,
            "mixing_ver1": augmentation.mixing_ver1,
            "mixing_ver2": augmentation.mixing_ver2,
        }
        if aug_mode not in augmentation_functions:
            raise ValueError("Augmentation name not supported")

        return augmentation_functions[aug_mode]()

    def __call__(self, input):
        img_aug = self.augmentor.to_deterministic()
        img_aug = self.augmentor.augment_image(input)
        return img_aug