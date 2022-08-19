import numpy as np
import pandas as pd

from tensorflow.keras.backend import sum, flatten

def metrics_single(y, pred):
    return {
        'dice': dice(y.astype(np.float32), pred.astype(np.float32)),
        'iou': iou(y.astype(np.float32), pred.astype(np.float32))
    }

def metrics_text(y, pred):
    metrics = metrics_single(y, pred)
    return 'Dice: {}, Iou: {}'.format(metrics['dice'], metrics['iou'])

def metrics_batch(ys, preds):
    return pd.DataFrame({
        'dice': [dice(y.astype(np.float32), pred.astype(np.float32)) for y, pred in zip(ys, preds)],
        'iou': [iou(y.astype(np.float32), pred.astype(np.float32)) for y, pred in zip(ys, preds)]
    })

def dice(y, p, smooth=1e-6):
    y_true = flatten(y)
    y_pred = flatten(p)
    intersection = sum(y_pred * y_true)
    dice = (2.0*intersection + smooth) / (sum(y_pred + y_true, axis=-1) + smooth)
    return dice

def tversky(y, p, alpha, beta, smooth=1e-6):
    y_true = flatten(y)
    y_pred = flatten(p)
    tp = sum(y_true * y_pred)
    fn = sum(y_true * (1-y_pred))
    fp = sum((1-y_true) * y_pred)
    tversky = (tp + smooth) / (tp + alpha*fn + beta*fp + smooth)
    return tversky

def tversky_loss(alpha, beta):
    def loss_function(y, p):
        return 1 - tversky(y, p, alpha, beta)
    return loss_function

def dice_loss(y, p):
    return 1 - dice(y, p)

def iou(targets, preds, smooth=0.001):
    intersection = sum(targets * preds)
    union = sum(targets + preds) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def iou_loss(y, pred):
    return 1 - iou(y, pred)

def uncertainty(preds):
    return np.sum(np.var(preds, axis=0))
