import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from glob import glob

from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_images(
    labeled_images=None, 
    labeled_fine=None, 
    labeled_coarse=None, 
    unlabeled_images=None, 
    unlabeled_coarse=None
):
    label_mapping = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0
    }
    if labeled_images:
        for f in tqdm(glob(str(Path(labeled_images, 'original', '*', '*_leftImg8bit.png'))), 'l_img'):
            f_img = load_img(f, target_size=(256, 512))
            f_img.save(Path(labeled_images) / 'processed' / Path(f).name)
    
    if unlabeled_images:
        for f in tqdm(glob(str(Path(unlabeled_images, 'original', '*', '*_leftImg8bit.png'))), 'ul_img'):
            f_img = load_img(f, target_size=(256, 512))
            f_img.save(Path(unlabeled_images) / 'processed' / Path(f).name)
        
    if labeled_fine:
        for f in tqdm(glob(str(Path(labeled_fine, 'original', '*', '*_gtFine_labelIds.png'))), 'l_fine'):
            f_img = load_img(f, target_size=(256, 512), color_mode='grayscale')
            f_array = img_to_array(f_img)
            label_mask = np.zeros_like(f_array)
            for k in label_mapping:
                label_mask[f_array == k] = label_mapping[k]
            np.save(Path(labeled_fine) / 'processed' / Path(f).stem, label_mask)
    
    if labeled_coarse:
        for f in tqdm(glob(str(Path(labeled_coarse, 'original', '*', '*_gtCoarse_labelIds.png'))), 'l_coarse'):
            f_img = load_img(f, target_size=(256, 512), color_mode='grayscale')
            f_array = img_to_array(f_img)
            label_mask = np.zeros_like(f_array)
            for k in label_mapping:
                label_mask[f_array == k] = label_mapping[k]
            np.save(Path(labeled_coarse) / 'processed' / Path(f).stem, label_mask)
    
    if unlabeled_coarse:
        for f in tqdm(glob(str(Path(unlabeled_coarse, 'original', '*', '*_gtCoarse_labelIds.png'))), 'ul_coarse'):
            f_img = load_img(f, target_size=(256, 512), color_mode='grayscale')
            f_array = img_to_array(f_img)
            label_mask = np.zeros_like(f_array)
            for k in label_mapping:
                label_mask[f_array == k] = label_mapping[k]
            np.save(Path(unlabeled_coarse) / 'processed' / Path(f).stem, label_mask)

def load_fine(folder_path):
    images = glob(str(Path(folder_path, '*_leftImg8bit.png')))
    data_fine = pd.DataFrame({
        'image_path': images,
        'mask_fine': [image.replace('/images/', '/masks-fine/').replace('_leftImg8bit.png', '_gtFine_labelIds.npy') for image in images],
        'mask_coarse': [image.replace('/images/', '/masks-coarse/').replace('_leftImg8bit.png', '_gtCoarse_labelIds.npy') for image in images]
    })
    return data_fine

def load_coarse(folder_path, n=19984, seed=0):
    images = glob(str(Path(folder_path, '*_leftImg8bit.png')))
    data_coarse = pd.DataFrame({
        'image_path': images,
        'mask_coarse': [image.replace('/images/', '/masks-coarse/').replace('_leftImg8bit.png', '_gtCoarse_labelIds.npy') for image in images]
    })
    data_coarse = data_coarse.sample(frac=1, random_state=seed).reset_index(drop=True)
    return data_coarse.iloc[:n]

def split_data(data, n_train, n_val, n_test, seed=0):
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    data_train = data.iloc[:n_train]
    data_val = data.iloc[n_train:(n_train+n_val)]
    data_test = data.iloc[(n_train+n_val):(n_train+n_val+n_test)]

    return data_train, data_val, data_test

def count_classes(mask_folder):
    mask_files = glob(str(Path(mask_folder, '*.npy')))
    
    n_classes = 0
    for mask_file in mask_files:
        mask = np.load(mask_file)
        max_class = np.max(mask)
        n_classes = max(n_classes, max_class)
    
    return int(n_classes) + 1

if __name__ == '__main__':
    preprocess_images(labeled_coarse = 'input/data/cityscapes-fine/masks-coarse')