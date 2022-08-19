import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.utils import Sequence, to_categorical, plot_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.ops.image_ops import flip_left_right

from omni.utils import transform_color, transform_space, img_to_array
from omni.metrics import dice, dice_loss, iou, tversky, tversky_loss
from omni.model import unet

class DataGenStreet(Sequence):
    def __init__(
        self, 
        data,
        batch_size,
        n_classes,
        data_omni=None,
        omni_samples=0,
        t_color=None,
        shuffle=True
    ):
        self.data = data.copy()
        self.n = len(self.data)
        self.n_classes = n_classes

        self.t_color = t_color

        self.shuffle = shuffle

        if omni_samples != 0:
            self.data_omni = data_omni.copy()
            self.batch_size_omni = omni_samples
            self.batch_size_labeled = batch_size - omni_samples
        else:
            self.data_omni = None
            self.batch_size_omni = 0
            self.batch_size_labeled = batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, t_c, t_f):
        img = load_img(path)
        if t_c is not None:
            x_array = transform_color(img, t_c, self.t_color)
        else:
            x_array = img_to_array(img)
        if t_f > 0.5:
            x_array = flip_left_right(x_array)
        return x_array/255.

    def __get_output(self, path, t_f):
        y_array = np.load(path)
        if t_f > 0.5:
            y_array = flip_left_right(y_array)
        return to_categorical(y_array, num_classes=self.n_classes)

    def __sample_omni(self):
        return self.data_omni.sample(n=self.batch_size_omni)

    def __get_data(self, batches):
        x_paths = list(batches['image_path']) 
        y_paths = list(batches['mask_fine'])

        if self.batch_size_omni > 0:
            omni_batches = self.__sample_omni() 
            x_paths += list(omni_batches['image_path'])
            y_paths += list(omni_batches['mask_fine'])

        t_c = np.random.random(4) if self.t_color is not None else None
        t_f = np.random.random()
        X_batch = np.asarray([self.__get_input(x, t_c, t_f) for x in x_paths])
        y_batch = np.asarray([self.__get_output(y, t_f) for y in y_paths])

        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.data[index * self.batch_size_labeled:(index + 1) * self.batch_size_labeled]
        x, y = self.__get_data(batches)        
        return x, y
    
    def __len__(self):
        return self.n // self.batch_size_labeled

class DataGenStreetCoarse(Sequence):
    def __init__(
        self, 
        data,
        batch_size,
        n_classes,
        data_omni=None,
        omni_samples=0,
        t_color=None,
        shuffle=True
    ):
        self.data = data.copy()
        self.n = len(self.data)
        self.n_classes = n_classes

        self.t_color = t_color

        self.shuffle = shuffle

        if omni_samples != 0:
            self.data_omni = data_omni.copy()
            self.batch_size_omni = omni_samples
            self.batch_size_labeled = batch_size - omni_samples
        else:
            self.data_omni = None
            self.batch_size_omni = 0
            self.batch_size_labeled = batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, t_c, t_f):
        img = load_img(path)
        if t_c is not None:
            x_array = transform_color(img, t_c, self.t_color)
        else:
            x_array = img_to_array(img)
        if t_f > 0.5:
            x_array = flip_left_right(x_array)
        return x_array/255.

    def __get_coarse(self, path, t_f):
        coarse = np.load(path)
        if t_f > 0.5:
            coarse = flip_left_right(coarse)
        return to_categorical(coarse, num_classes=self.n_classes)

    def __get_output(self, path, t_f):
        y_array = np.load(path)
        if t_f > 0.5:
            y_array = flip_left_right(y_array)
        return to_categorical(y_array, num_classes=self.n_classes)

    def __sample_omni(self):
        return self.data_omni.sample(n=self.batch_size_omni)

    def __get_data(self, batches):
        x_paths = list(batches['image_path']) 
        y_paths = list(batches['mask_fine'])
        coarse_paths = list(batches['mask_coarse'])
        if self.batch_size_omni > 0:
            omni_batches = self.__sample_omni() 
            x_paths += list(omni_batches['image_path'])
            y_paths += list(omni_batches['mask_fine'])
            coarse_paths += list(omni_batches['mask_coarse'])

        t_c = np.random.random(4) if self.t_color is not None else None
        t_f = np.random.random()
        X_batch = np.asarray([self.__get_input(x, t_c, t_f) for x in x_paths])
        c_coarse = np.asarray([self.__get_coarse(c, t_f) for c in coarse_paths])
        X_batch = np.concatenate((X_batch, c_coarse), axis=-1)
        y_batch = np.asarray([self.__get_output(y, t_f) for y in y_paths])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.data[index * self.batch_size_labeled:(index + 1) * self.batch_size_labeled]
        x, y = self.__get_data(batches)        
        return x, y
    
    def __len__(self):
        return self.n // self.batch_size_labeled

def train(
    model_name,
    data_train,
    data_val,

    img_size, 
    filter_size, 
    n_classes,

    epochs,
    tversky_alpha,
    tversky_beta,
    batch_size,
    t_color=None,

    data_omni=None,
    omni_samples=0,

    coarse=False
):
    
    if not coarse:
        model = unet(img_size, filter_size, n_classes)
        if omni_samples == 0 or data_omni is None:
            train_gen = DataGenStreet(
                data=data_train,
                batch_size=batch_size,
                n_classes=n_classes,
                t_color=t_color
            )
        else:
            train_gen = DataGenStreet(
                data=data_train, 
                batch_size=batch_size,
                n_classes=n_classes,
                t_color=t_color,
                data_omni=data_omni,
                omni_samples=omni_samples,
            )
            
        val_gen = DataGenStreet(
            data=data_val, 
            batch_size=batch_size,
            n_classes=n_classes,
            t_color=t_color
        )
    else:
        model = unet_label(img_size, filter_size, n_classes)
        if omni_samples == 0 or data_omni is None:
            train_gen = DataGenStreetCoarse(
                data=data_train,
                batch_size=batch_size,
                n_classes=n_classes,
                t_color=t_color
            )
        else:
            train_gen = DataGenStreetCoarse(
                data=data_train, 
                batch_size=batch_size,
                n_classes=n_classes,
                t_color=t_color,
                data_omni=data_omni,
                omni_samples=omni_samples,
            )
            
        val_gen = DataGenStreetCoarse(
            data=data_val, 
            batch_size=batch_size,
            n_classes=n_classes,
            t_color=t_color
        )
        
    checkpoint = ModelCheckpoint(
        Path('output', 'models', model_name, 'weights.hdf5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss=tversky_loss(tversky_alpha, tversky_beta),
        #metrics=[dice, iou]
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    pd.DataFrame(history.history).to_excel(Path('output', 'models', model_name, 'training_table.xlsx'))
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(Path('output', 'models', model_name, 'training_plot.png'))
