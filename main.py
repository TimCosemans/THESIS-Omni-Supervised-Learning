import yaml, argparse
import pandas as pd
from pathlib import Path

from omni.data import load_fine, load_coarse, split_data
from omni.model import unet
from omni.train import train
from src.omni.inference import inference, multi_transform_inference, multi_transform_inference_visual



class Pipeline():
    def __init__(self, params_data_path, params_model_path):
        params_data = self.load_params(params_data_path)
        params_model = self.load_params(params_model_path)

        self.n_train = params_data['n_train']
        self.n_val = params_data['n_val']
        self.n_test = params_data['n_test']
        self.n_unlabeled = params_data['n_unlabeled']

        self.img_size = (params_model['img_height'], params_model['img_width'])
        self.filter_size = params_model['filter_size']
        self.n_classes = params_model['n_classes']
        
        self.load_data(
            params_data['folder_labeled'],
            params_data['folder_unlabeled']
        )
    @staticmethod
    def load_params(yaml_file):
        with open(yaml_file, 'r') as f:
            parsed_yaml = yaml.full_load(f)
        return parsed_yaml

    def load_data(self, folder_labeled, folder_unlabeled):
        data_labeled = load_fine(folder_labeled)
        self.data_train, self.data_val, self.data_test = split_data(data_labeled, self.n_train, self.n_val, self.n_test)
        self.data_unlabeled = load_coarse(folder_unlabeled, self.n_unlabeled)

        print("Total labeled samples:", len(data_labeled))
        print("Train/val/test split:", '/'.join(str(v) for v in [len(self.data_train),len(self.data_val), len(self.data_test)]))
        print("Total unlabeled samples:", len(self.data_unlabeled))
    
    def load_model(self, model_name=None):
        if isinstance(model_name, list):
            model = []
            for m in model_name:
                model.append(unet(self.img_size, self.filter_size, self.n_classes, Path('output','models', m, 'weights.hdf5')))
        else:
            model = unet(self.img_size, self.filter_size, self.n_classes, Path('output','models', model_name, 'weights.hdf5'))
        return model
    
    def train_model_sup(self, model_name, epochs, batch_size, tversky_alpha, tversky_beta, t_color=None, coarse=False):
        train(
            model_name=model_name,
            data_train=self.data_train, 
            data_val=self.data_val, 

            img_size=self.img_size,
            filter_size=self.filter_size,
            n_classes=self.n_classes,

            epochs=epochs,
            batch_size=batch_size,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            t_color=t_color,
            coarse=coarse
        )

    def test_model(self, model_name, t_color=None):
        model = self.load_model(model_name)
        inference(
            model=model, 
            data=self.data_test, 
            t_color=t_color
        )

    def test_model_visual(self, model_name, visuals_folder, n_input, n_transform, t_color):
        model = self.load_model(model_name)
        multi_transform_inference_visual(
            model=model, 
            data=self.data_test.sample(n_input), 
            visuals_folder=visuals_folder, 
            t_color=t_color, 
            n_transform=n_transform
        )

    def auto_label(self, model_name, omni_folder, t_color, n_transform, n_output=None, n_input=None):
        model = self.load_model(model_name)
        
        multi_transform_inference(
            model=model,
            data=self.data_unlabeled,
            labels_folder=omni_folder,    
            n_input=n_input,
            n_transform=n_transform,
            t_color=t_color,
            n_output=n_output
        )

    def train_model_omni(self, model_name, epochs, batch_size, tversky_alpha, tversky_beta, omni_folder, omni_samples, t_color):
        data_omni = pd.read_csv(Path(omni_folder, '.info.csv'))
        train(
            model_name=model_name,

            data_train=self.data_train, 
            data_val=self.data_val, 

            img_size=self.img_size,
            filter_size=self.filter_size,
            n_classes=self.n_classes,

            epochs=epochs,
            batch_size=batch_size,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            t_color=t_color,

            data_omni=data_omni,
            omni_samples=omni_samples,
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='config')
    args = parser.parse_args()

    experiment = Pipeline(
        params_data_path='input/configs/data.yaml', 
        params_model_path='input/configs/model.yaml'
    )
    
    params = experiment.load_params(f'input/configs/{args.config}.yaml')
    step = params.pop('step')

    if step == 'train_sup':
        experiment.train_model_sup(**params)
    elif step == 'test':
        experiment.test_model(**params)
    elif step == 'test_visual':
        experiment.test_model_visual(**params)
    elif step == 'mti':
        experiment.auto_label(**params)
    elif step == 'train_omni':
        experiment.train_model_omni(**params)
    else:
        raise ValueError('unknown step')