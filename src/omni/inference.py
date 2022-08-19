import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from omni.utils import transform_color, transform_flip, ensemble_predictions
from omni.metrics import metrics_batch, uncertainty, metrics_text

def inference(model, data, t_color):
    images = list(data['image_path'])
    if t_color is not None:
        preds = []
        for pred, _, _ in data_distillation_batch(images, model, 16, t_color):
            preds.append(pred)
        preds = np.asarray(preds)
    elif isinstance(model, list):
        preds = model_distillation(np.asarray([img_to_array(load_img(x_path))/255. for x_path in  data['image_path']]), model)
    else:
        preds = model.predict(np.asarray([img_to_array(load_img(x_path))/255. for x_path in  data['image_path']]))

    ys = np.asarray([to_categorical(np.load(y_path), num_classes=preds.shape[-1]) for y_path in data['mask_fine']])
    metrics = metrics_batch(ys, preds)

    print('Truth shape:', ys.shape)
    print('Preditions output:', preds.shape)
    print(
        ('Dice: {} ({})').format(
            round(metrics['dice'].mean(), 4),
            round(metrics['dice'].std(), 4) 
        )
    )
    print(
        ('IoU: {} ({})').format(
            round(metrics['iou'].mean(), 4),
            round(metrics['iou'].std(), 4) 
        )
    )
    return preds, metrics

def data_distillation(x_path, model, n_transform, t_color):
    t_colors = np.random.random((n_transform,4))
    t_flips = np.random.random(n_transform)
    x_array = np.asarray([transform_color(load_img(x_path), t_c, t_color) for t_c in t_colors])
    x_array[t_flips > 0.5] = transform_flip(x_array[t_flips > 0.5])
    if isinstance(model, list):
        preds = model_distillation(np.asarray(x_array)/255., model)
    else:
        preds = model.predict(x_array/255., batch_size=16, verbose=0)
    preds_orig = preds.copy()
    preds[t_flips > 0.5] = transform_flip(preds[t_flips > 0.5])
    return ensemble_predictions(preds, 'mean'), uncertainty(preds), Path(x_path).stem, x_array, preds_orig

def data_distillation_batch(x_paths, model, n_transform, t_color):
    for x_path in tqdm(x_paths):
        t_colors = np.random.random((n_transform,4))
        t_flips = np.random.random(n_transform)
        x_array = np.asarray([transform_color(load_img(x_path), t_c, t_color) for t_c in t_colors])
        x_array[t_flips > 0.5] = transform_flip(x_array[t_flips > 0.5])
        if isinstance(model, list):
            preds = model_distillation(np.asarray(x_array)/255., model)
        else:
            preds = model.predict(x_array/255., batch_size=16, verbose=0)
        preds[t_flips > 0.5] = transform_flip(preds[t_flips > 0.5])
        yield ensemble_predictions(preds, 'mean'), uncertainty(preds), Path(x_path).stem

def model_distillation(x, models):
    preds = np.asarray([model.predict(x, batch_size=16, verbose=0) for model in models])
    return ensemble_predictions(preds, 'mean')

def multi_transform_inference(
    model,
    data,
    labels_folder, 
    t_color,
    n_transform,
    n_output,
    n_input=None
):
    if n_input is not None:
        data = data.sample(n_input)
    if n_output is None:
        n_output = len(data)

    pred_paths = []
    uncertainty_scores = []

    for preds, uncertainty_score, filename in data_distillation_batch(list(data['image_path']), model, n_transform, t_color):
        pred_path = Path(labels_folder, f'{filename}.npy')
        np.save(
            pred_path, 
            np.argmax(preds, axis=-1, keepdims=True)
        )
        pred_paths.append(pred_path)
        uncertainty_scores.append(uncertainty_score)

    data_omni = pd.DataFrame({
        'image_path': data['image_path'],
        'mask_fine': pred_paths,
        'mask_coarse': data['mask_coarse'],
        'uncertainty': uncertainty_scores
    })
    
    data_omni.to_csv(Path(labels_folder, '.info.csv'))

def multi_transform_inference_visual(
    model,
    data,
    visuals_folder, 
    t_color,
    n_transform,
): 
    x_paths = list(data['image_path'])
    y_paths = list(data['mask_fine'])
    for x_path, y_path in zip(x_paths, y_paths):
        y = np.load(y_path)
        x_base = img_to_array(load_img(x_path))
        pred_base = model.predict(np.asarray([x_base/255.]))[0]
        ensemble, uncertainty_score, filename, xs, preds = data_distillation(x_path, model, n_transform, t_color)

        fig, axs = plt.subplots(len(preds)+2, 3)
        axs[0, 0].set_title('Input')
        axs[0, 1].set_title('Truth')
        axs[0, 2].set_title('Predict')

        axs[0, 0].imshow(x_base/255.)
        axs[0, 1].imshow(y)
        axs[0, 2].imshow(np.argmax(pred_base, axis=-1))
        axs[0, 2].text(300, 100, metrics_text(to_categorical(y, num_classes=20), pred_base), bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        for j, (x, pred) in enumerate(zip(xs, preds)):

            axs[j+1, 0].imshow(x/255.)
            axs[j+1, 1].imshow(y)
            axs[j+1, 2].imshow(np.argmax(pred, axis=-1))
            axs[j+1, 2].text(300, 100, metrics_text(to_categorical(y, num_classes=20), pred), bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        axs[-1, 0].imshow(x_base/255.)
        axs[-1, 1].imshow(y)
        axs[-1, 2].imshow(np.argmax(ensemble, axis=-1))
        axs[-1, 2].text(300, 100, metrics_text(to_categorical(y, num_classes=20), ensemble), bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        fig.savefig(Path(visuals_folder, f'{filename}.png'))
        