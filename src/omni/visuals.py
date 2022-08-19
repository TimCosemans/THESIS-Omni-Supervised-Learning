import numpy as np
import random
from matplotlib import pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

from omni.metrics import metrics_single

def visualize_pred(x_img, y_img, pred, n_classes):
    pred_argmax = np.argmax(pred, -1)
    pred_img = array_to_img(
        np.expand_dims(pred_argmax,-1), 
        scale=False,
        dtype="uint8"
    )

    fig, ax = plt.subplots(1, 3)

    ax[0].set_title('Input')
    ax[1].set_title('Truth')
    ax[2].set_title('Predict')

    ax[0].imshow(x_img)
    ax[1].imshow(y_img, vmin=0, vmax=n_classes-1)
    ax[2].imshow(pred_img, vmin=0, vmax=n_classes-1)
    ax[2].text(300, 100, 
        metrics_single(
            to_categorical(img_to_array(y_img), n_classes), 
            pred,
            'text'
        ),
    fontsize=16, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show(block=True)

def visualize_multiple(x_paths, y_paths, preds, img_size, indices=None):
    if indices is not None:
        x_paths = np.take(x_paths, indices,axis=0)
        y_paths = np.take(y_paths, indices,axis=0)
        preds = np.take(preds, indices,axis=0)
    
    fig, axs = plt.subplots(preds.shape[0], 3)
    axs[0, 0].set_title('Input', fontsize=32)
    axs[0, 1].set_title('Truth', fontsize=32)
    axs[0, 2].set_title('Predict', fontsize=32)

    for i, (x_path, y_path, pred) in enumerate(zip(x_paths, y_paths, preds)):
        pred = np.argmax(pred,axis=-1)

        axs[i, 0].imshow(load_img(x_path, target_size=img_size))
        axs[i, 1].imshow(load_img(y_path, target_size=img_size, color_mode="grayscale"))
        axs[i, 2].imshow(array_to_img(np.expand_dims(pred, 2), scale=False))
        axs[i, 2].text(300, 100, metrics_single(y_path, pred, 'text', img_size), fontsize=16, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.show(block=True)

def visualize_mti(x_img, pred, n_classes, tform_name):
    pred_img = array_to_img(
        np.expand_dims(pred,-1), 
        scale=False,
        dtype="uint8"
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title(tform_name)
    ax[1].set_title('Predict')

    ax[0].imshow(x_img)
    ax[1].imshow(pred_img, vmin=0, vmax=n_classes-1)

    plt.show()
