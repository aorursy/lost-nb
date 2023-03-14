#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored.
        
    probability: np.array
    threshold: int
    min_size: int
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)

    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1

    return predictions




get_ipython().system('mkdir -p /tmp/pip/cache/')
get_ipython().system('cp /kaggle/input/segmentation-models-zip-003/efficientnet_pytorch-0.4.0.xyz /tmp/pip/cache/efficientnet_pytorch-0.4.0.tar.gz')
get_ipython().system('cp /kaggle/input/segmentation-models-zip-003/pretrainedmodels-0.7.4.xyz /tmp/pip/cache/pretrainedmodels-0.7.4.tar.gz')
get_ipython().system('cp /kaggle/input/segmentation-models-zip-003/segmentation_models_pytorch-0.0.3.xyz /tmp/pip/cache/segmentation_models_pytorch-0.0.3.tar.gz')




get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/ efficientnet-pytorch')
get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/ segmentation-models-pytorch')
get_ipython().system('pip install /kaggle/input/tta-pytorch/ttach-0.0.1-py3-none-any.whl')




import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt

import albumentations as albu
from tqdm import tqdm_notebook
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.jit import load

import ttach as tta
from severstal_utils import rle2mask, mask2rle




unet_se_resnext50_32x4d =     load('/kaggle/input/severstalmodels/unet_se_resnext50_32x4d.pth').cuda()
unet_mobilenet2 = load('/kaggle/input/severstalmodels/unet_mobilenet2.pth').cuda()
unet_resnet34 = load('/kaggle/input/severstalmodels/unet_resnet34.pth').cuda()




from typing import Callable, Dict
from albumentations import ImageOnlyTransform

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class ChannelTranspose(ImageOnlyTransform):
    def get_transform_init_args_names(self):
        return ()

    def get_params_dependent_on_targets(self, params):
        pass

    def __init__(self, axes=(2, 0, 1)):
        super().__init__(always_apply=True)
        self.axes = axes

    def apply(self, img, **params):
        return np.transpose(img, self.axes)


class ImageDataset(Dataset):
    def __init__(
            self,
            *,
            img_folder: str,
            fold_csv: str = None,
            fold_number: int = None,
            is_test: bool = False,
            gray_scale: bool = False,
            num_classes=2,
            max_count=None,
            meta_cols=(),
            transforms=None,
            postprocess_func: Callable[[Dict], Dict] = None,
            include_image_orig=False
    ):
        self.img_folder = img_folder

        if fold_csv:
            df = pd.read_csv(fold_csv)
            if fold_number is not None:
                if is_test:
                    self.data = df[df['fold'] == fold_number]
                else:
                    self.data = df[df['fold'] != fold_number]
            else:
                self.data = df
        else:
            self.data = pd.DataFrame(
                {'image': os.listdir(img_folder)}).sort_values(by='image')

        self.data = self.data.to_dict(orient='row')
        if max_count is not None:
            self.apply_max_count(max_count)

        for row in self.data:
            self.preprocess_row(row)

        self.transforms = transforms
        self.gray_scale = gray_scale
        self.num_classes = num_classes
        self.meta_cols = meta_cols
        self.postprocess_func = postprocess_func
        self.include_image_orig = include_image_orig

    def apply_max_count(self, max_count):
        if isinstance(max_count, Number):
            self.data = self.data[:max_count]
        else:
            data = defaultdict(list)
            for row in self.data:
                data[row['label']].append(row)
            min_index = np.argmin(max_count)
            min_count = len(data[min_index])
            for k, v in data.items():
                count = int(min_count * (max_count[k] / max_count[min_index]))
                data[k] = data[k][:count]

            self.data = [v for i in range(len(data)) for v in data[i]]

    def preprocess_row(self, row: dict):
        row['image'] = os.path.join(self.img_folder, row['image'])

    def __len__(self):
        return len(self.data)

    def _get_item_before_transform(self, row: dict, item: dict):
        pass

    def _get_item_after_transform(self, row: dict,
                                  transformed: dict,
                                  res: dict):
        if 'label' in row:
            res['targets'] = ast.literal_eval(str(row['label']))
            if isinstance(res['targets'], list):
                res['targets'] = np.array(res['targets'], dtype=np.float32)

    def __getitem__(self, index):
        row = self.data[index]
        image = self.read_image_file(row['image'], self.gray_scale)
        item = {'image': image}

        self._get_item_before_transform(row, item)

        if self.transforms:
            item = self.transforms(**item)
        if self.gray_scale:
            item['image'] = np.expand_dims(item['image'], axis=0)
        res = {
            'features': item['image'].astype(np.float32),
            'image_file': row['image']
        }
        if self.include_image_orig:
            res['image'] = image

        for c in self.meta_cols:
            res[c] = row[c]

        self._get_item_after_transform(row, item, res)
        if self.postprocess_func:
            res = self.postprocess_func(res)
        return res

    @staticmethod
    def read_image_file(path: str, gray_scale=False):
        if path.endswith('.tiff') and not gray_scale:
            return tifffile.imread(path)
        elif path.endswith('.npy'):
            return np.load(path)
        else:
            if gray_scale:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                assert img is not None,                     f'Image at path {path} does not exist'
                return img.astype(np.uint8)
            else:
                img = cv2.imread(path)
                assert img is not None,                     f'Image at path {path} does not exist'
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
class TtaWrap(Dataset):
    def __init__(self, dataset: Dataset, tfms=()):
        self.dataset = dataset
        self.tfms = tfms

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def inverse(self, a: np.array):
        last_dim = len(a.shape) - 1
        for t in self.tfms:
            if isinstance(t, albu.HorizontalFlip):
                a = flip(a, last_dim)
            elif isinstance(t, albu.VerticalFlip):
                a = flip(a, last_dim - 1)
            elif isinstance(t, albu.Transpose):
                axis = (0, 1, 3, 2) if len(a.shape) == 4 else (0, 2, 1)
                a = a.permute(*axis)

        return a




class Model:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)

model = Model([unet_se_resnext50_32x4d, unet_mobilenet2, unet_resnet34])




def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        albu.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = albu.Compose(res)
    return res

img_folder = '/kaggle/input/severstal-steel-defect-detection/test_images'
batch_size = 2
num_workers = 0

# Different transforms for TTA wrapper
transforms = [
    [],
    [albu.HorizontalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]




thresholds = [0.50, 0.50, 0.50, 0.50]
min_area = [600, 600, 1000, 2000]

res = []
# Iterate over all TTA loaders
total = len(datasets[0])//batch_size
for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
    preds = []
    image_file = []
    for i, batch in enumerate(loaders_batch):
        features = batch['features'].cuda()
        p = torch.sigmoid(model(features))
        # inverse operations for TTA
        p = datasets[i].inverse(p)
        preds.append(p)
        image_file = batch['image_file']
    
    # TTA mean
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()
    
    # Batch post processing
    for p, file in zip(preds, image_file):
        file = os.path.basename(file)
        # Image postprocessing
        for i in range(4):
            p_channel = p[i]
            imageid_classid = file+'_'+str(i+1)
            
            # HERE we change original post processing (commented) 
            p_channel = post_process(p_channel, thresholds[i], min_area[i])
            #p_channel = (p_channel>thresholds[i]).astype(np.uint8)
            #if p_channel.sum() < min_area[i]:
            #    p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)

            res.append({
                'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })
        
df = pd.DataFrame(res)
df.to_csv('submission.csv', index=False)	




df = pd.DataFrame(res)
df = df.fillna('')
df.to_csv('submission.csv', index=False)




df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])
df['empty'] = df['EncodedPixels'].map(lambda x: not x)
df[df['empty'] == False]['Class'].value_counts()

