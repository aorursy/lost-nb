#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install pandas pillow matplotlib sklearn torch torchvision albumentations -q')
get_ipython().system('pip install git+https://github.com/qubvel/segmentation_models.pytorch -q')
get_ipython().system('pip install catalyst -q')

import os

import cv2

import albumentations as albu
import albumentations.pytorch as AT

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback

import numpy as np
import pandas as pd

import segmentation_models_pytorch as smp

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader, Dataset

from sklearn.model_selection import train_test_split

import seaborn as sns

import torch.nn as nn
from segmentation_models_pytorch.utils.base import Activation

import torch

get_ipython().system('pip install --upgrade pip -q')

get_ipython().run_line_magic('matplotlib', 'inline')




get_ipython().system('ls ../input/understanding_cloud_organization')




home = '../input/understanding_cloud_organization'

def load_image_data():
    data = pd.read_csv('{}/train.csv'.format(home))
    data.loc[:, 'id'] = data.loc[:, 'Image_Label'].apply(lambda x: x.split('_')[0])
    data.loc[:, 'label'] = data.loc[:, 'Image_Label'].apply(lambda x: x.split('_')[1])
    data.loc[data.loc[:, 'EncodedPixels'].isnull(), 'label'] = np.nan
    data.set_index('id', drop=True, inplace=True)
    return data

raw = load_image_data()

data = raw.copy()

num_epochs = 2
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
ACTIVATION = None

logdir = "./logs/segmentation"

bs = 32
num_workers = 1

data = data.iloc[:128, :]

print(data.shape)
data.head()




#  number of labels in each image
data.groupby('id').count().sort_values('label')




#  number of images that have a number of labels
data.groupby('id').count().sort_values('label').groupby('label').count()




from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


def make_single_mask_from_rle(mask_rle, shape=(1400, 2100)):
    " https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools "
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def plot_images(images):
    nrows = 4
    sample = np.random.choice(images, nrows)
    print(sample)

    shape = (1400, 2100)

    f, axes = plt.subplots(nrows, 4, figsize=(25, 16))

    for idx, img_id in enumerate(sample):
        sub = data.loc[data.index == img_id]
        img = Image.open('{}/train_images/{}'.format(home, img_id))

        for col, ax in enumerate(axes[:, idx]):
            ax.imshow(img)
            ax.set_title(sub.loc[:, 'label'].iloc[col])

            try:
                mask = make_single_mask_from_rle(sub.loc[:, 'EncodedPixels'].iloc[col], shape)

            except AttributeError:
                mask = np.zeros(shape)

            ax.imshow(mask, alpha=0.5 , cmap='gray')
            
images = list(set(data.index))
plot_images(images)




plot_images(images)




def get_img(data, idx=None, folder='train_images'):
    if idx is None:
        idx = np.random.randint(0, data.shape[0])
        
    img_data = data.iloc[idx, :]
    img_id = img_data.name
    img = cv2.imread(os.path.join(home, folder, img_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = data.EncodedPixels
    return img, img_id, mask


def get_masks(data, img_id):
    return data.loc[data.index == img_id, :]


def plot_with_aug(aug, img):
    f, axes = plt.subplots(ncols=2, figsize=(25, 16))
    
    axes[0].imshow(img)
    axes[0].set_title('original')
    
    axes[1].imshow(aug(image=img)['image'])
    axes[1].set_title(repr(aug)) 
    return f




f = plot_with_aug(albu.HorizontalFlip(p=0.5), get_img(data)[0])




f = plot_with_aug(albu.ShiftScaleRotate(
    scale_limit=1, #  smaller or bigger
    rotate_limit=10,
    shift_limit=1, 
    p=1.0, 
    border_mode=0
), get_img(data)[0])




f = plot_with_aug(albu.GridDistortion(
    p=1.0,
), get_img(data)[0])




f = plot_with_aug(albu.OpticalDistortion(
    p=1.0, distort_limit=2, shift_limit=0.5
), get_img(data)[0])




def make_all_masks(data, img_id, shape=(1400, 2100)):
    label_order = ['Fish', 'Flower', 'Gravel', 'Sugar']
    
    mask_data = get_masks(data, img_id)
    masks = np.zeros((*shape, len(label_order)))

    for idx, expected_label in enumerate(label_order):
        data = mask_data.iloc[idx]
        lbl = data.loc['label']

        if lbl is not np.nan:
            assert lbl == expected_label
            masks[:, :, idx] = make_single_mask_from_rle(data.loc['EncodedPixels'], shape)
            
    return masks

masks = make_all_masks(data, data.index[0])




class CloudDataset(Dataset):
    
    def __init__(
        self,
        data,
        dataset='train',
        transform=albu.Compose([albu.Resize(320, 640)]),
        preprocessing=None
    ):
        super().__init__()
        self.data = data
        self.folder = os.path.join(home, '{}_images'.format(dataset))
        self.transform = transform
        self.preprocessing = preprocessing
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img_data = self.data.iloc[idx, :]
        img_id = img_data.name
                
        img = cv2.imread(os.path.join(self.folder, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = make_all_masks(self.data, img_id)

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
            
        return img, mask

    
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        #albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        #albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(320, 640)
    ]
    return albu.Compose(train_transform)





id_grp = data.groupby(data.index).count()
sub = id_grp.iloc[:, :]
train_id, valid_id = train_test_split(sub.index, test_size=0.2, stratify=sub.loc[:, 'label'])
print(train_id.shape, valid_id.shape)

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=4,
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = CloudDataset(
    data.loc[train_id, :], 
    dataset='train', 
    transform=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn)
)

valid_dataset = CloudDataset(
    data.loc[valid_id, :], 
    dataset='train', 
    transform=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn)
)

loaders = {
    'train': DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers),
    'valid': DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
}




len(DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=num_workers))




# model, criterion, optimizer
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2}, 
    {'params': model.encoder.parameters(), 'lr': 1e-3},  
])


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps)             / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)

class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce

scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion = BCEDiceLoss(eps=1.)  #  training on CE, reporting on DICE
runner = SupervisedRunner()




runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)




from catalyst.dl import utils

utils.plot_metrics(
    logdir=logdir, 
    metrics=["loss", "dice", 'lr', '_base/lr']
)




from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
import tqdm


runner.infer(
    model=model,
    loaders={"valid": loaders['valid']},
    callbacks=[
        CheckpointCallback(
            resume=f"{logdir}/checkpoints/best.pth"),
        InferCallback()
    ],
)

valid_masks = []
probabilities = np.zeros((2220, 350, 525))
for i, (batch, output) in enumerate(tqdm.tqdm(zip(
        valid_dataset, runner.callbacks[0].predictions["logits"]))):
    image, mask = batch
    for m in mask:
        if m.shape != (350, 525):
            m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        valid_masks.append(m)

    for j, probability in enumerate(output):
        if probability.shape != (350, 525):
            probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        probabilities[i * 4 + j, :, :] = probability




def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)
    intersection = np.logical_and(img1, img2)
    return 2. * intersection.sum() / (img1.sum() + img2.sum())

class_params = {}
for class_id in range(4):
    print(class_id)
    attempts = []
    for t in range(0, 100, 5):
        t /= 100
        for ms in [0, 100, 1200, 5000, 10000]:
            masks = []
            for i in range(class_id, len(probabilities), 4):
                probability = probabilities[i]
                predict, num_predict = post_process(sigmoid(probability), t, ms)
                masks.append(predict)

            d = []
            for i, j in zip(masks, valid_masks[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))

            attempts.append((t, ms, np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]
    
    class_params[class_id] = (best_threshold, best_size)




print(class_params)
sns.lineplot(x='threshold', y='dice', hue='size', data=attempts_df);
plt.title('Threshold and min size vs dice for one of the classes');

