import numpy as np

from PIL import ImageEnhance
from tensorflow.keras.preprocessing.image import img_to_array, apply_affine_transform
from tensorflow.python.ops.image_ops import flip_left_right

def transform_flip(array):
    return flip_left_right(array)

def transform_color(img, t, t_i):
    t_min = t_i[0]
    t_id = t_i[1]-t_i[0]
    img = ImageEnhance.Color(img).enhance(t_min + t_id * t[0])
    img = ImageEnhance.Contrast(img).enhance(t_min + t_id * t[1])
    img = ImageEnhance.Brightness(img).enhance(t_min + t_id * t[2])
    img = ImageEnhance.Sharpness(img).enhance(t_min + t_id * t[3])
    return img_to_array(img)

def transform_space(array, t, t_i=[-10,10]):
    t_id = t_i[1]-t_i[0]
    array = apply_affine_transform(
        array,
        theta=-10+t_id*t[0],
        tx=-10+t_id*t[1], ty=-10+t_id*t[2],
        zx=1-0.1+0.2*t[3], zy=1-0.1+0.2*t[3],
        row_axis=0, col_axis=1,channel_axis=2,
        #fill_mode='constant', cval=-1
    )
    return array

def transform_color_multi(img, ts, img_size):
    array_ts = np.zeros((ts.shape[0],) + img_size + (3,))
    for i, t in enumerate(ts):
        img_t = transform_color(img, t)
        array_ts[i] = img_to_array(img_t)
    return array_ts

def transform_space_multi(img, ts, img_size):
    array = img_to_array(img)
    array_ts = np.zeros((ts.shape[0],) + img_size + (3,))
    for i, t in enumerate(ts):
        array_t = transform_space(array, t)
        array_ts[i] = array_t
    return array_ts

def reverse_space(preds, t_spaces):
    t_spaces *= -1
    return np.asarray([transform_space(pred, t_space) for pred, t_space in zip(preds, t_spaces)])

def reverse_flip(preds, t_flips):
    return np.asarray([transform_flip(pred) if t_flip == 1 else pred for pred, t_flip in zip(preds, t_flips)])

def ensemble_predictions(preds, method='mean', hard_labels=False):
    if method == 'mean':
        ensemble = np.mean(preds, axis=0)
    elif method == 'max':
        ensemble = np.max(preds, axis=0)
    else:
        raise ValueError('Unknown ensembling method')
    
    if hard_labels:
        return np.argmax(ensemble, axis=-1)
    else:
        return ensemble
