#!/usr/bin/env python
# coding: utf-8



class CFG:
    debug=True
    #height=256
    #width=256
    lr=1e-4
    batch_size=16
    epochs=1 # you can train more epochs
    seed=777
    target_size=1
    n_fold=4
    warmup=1
    device=1




import time
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import random
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

    
def init_logger(log_file='train.log'):
    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler
    
    log_format = '%(asctime)s %(levelname)s %(message)s'
    
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
    
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
    
    logger = getLogger('RSNA2020')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger
import datetime
dt_now = datetime.datetime.now()
print("実験開始",dt_now)
LOG_FILE = 'train{}.log'.format(dt_now)
LOGGER = init_logger(LOG_FILE)




def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)




#データの居場所を確認
import os
os.listdir("../input/siim-acr-pneumothorax-segmentation")




import pandas as pd
train = pd.read_csv("../input/siim-acr-pneumothorax-segmentation/stage_2_train.csv")
train.head()




image_id = train['ImageId'].values[0]
image_path = os.path.join("../input/siim-png-images/train_png", image_id+".png")
print(image_path)
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image = cv2.imread(image_path)
plt.imshow(image)
plt.show()
plt.imshow(image[100:1024-100,100:1024-100])
plt.show()




image.shape




import numpy as np
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)




#mask
annotations = train["EncodedPixels"].values[0]
    
mask = rle2mask(annotations,1024,1024)




import pydicom as dcm
dcm_path = "../input/siim-acr-pneumothorax-segmentation/stage_2_images/ID_0011fe81e.dcm"
dcm_data = dcm.read_file(dcm_path)
import cv2
dcm_image = dcm_image[50:1024-50,50:1024-50]
plt.imshow(dcm_image,cmap="bone")
plt.show()




plt.imshow(image[100:1024-100,100:1024-100])
plt.show()
plt.imshow(mask.T[100:1024-100,100:1024-100])
plt.show()




import torch
mask = cv2.resize(mask,(256, 256))
image = cv2.resize(image,(256, 256))

mask = np.expand_dims(mask, axis=0)
pos = np.where(np.array(mask)[0, :, :])
xmin = np.min(pos[1])
xmax = np.max(pos[1])
ymin = np.min(pos[0])
ymax = np.max(pos[0])
boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)#物体検出でとくためのbbox
labels = torch.ones((1,), dtype=torch.int64)
masks = torch.as_tensor(mask, dtype=torch.uint8)




exist = []
for i in train["ImageId"].values:
    if os.path.exists("../input/siim-png-images/train_png/"+i + '.png')==False:
        exist.append(0)
    else:
        exist.append(1)
train["exist"]=exist
train=train[train["exist"]==1]
train = train.reset_index(drop=True)
train["has_mask"]=1
negative_idx= train[train["EncodedPixels"]=="-1"].index.tolist()
train["has_mask"].values[negative_idx]=0
train.head()




import torchvision
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class SIIMDataset(torch.utils.data.Dataset):
    
    def __init__(self, df,):
        self.df = df
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        image_id = self.df['ImageId'].values[idx]
        image_path = os.path.join("../input/siim-png-images/train_png", image_id+".png")
        
        image = cv2.imread(image_path)[100:1024-100,100:1024-100]
        
        h,w =image.shape[0],image.shape[1]
        annotations = train["EncodedPixels"].values[idx]
        if annotations=="-1":
            mask = np.zeros((h,w))[100:1024-100,100:1024-100]
            label = torch.zeros((1,), dtype=torch.int64)
        else:
            mask = rle2mask(annotations,h,w)[100:1024-100,100:1024-100]
            label = torch.zeros((1,), dtype=torch.int64)
        mask = cv2.resize(mask,(256, 256))
        image = cv2.resize(image,(256, 256))
        mask = np.expand_dims(mask, axis=0)
        
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        image = torch.as_tensor(image.transpose((2,0,1)), dtype=torch.uint8)
        
        return image,mask,label
        
        
        
        




import torch.nn as nn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
class mask_r_cnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features_mask = self.model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        num_classes=2
        self.model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    def forward(self,x,target):
        x = self.model_ft(x,target)
        return x
    
#targetもいれないといけない...mask bboxなどを含むもの、不明　




get_ipython().system('pip install git+https://github.com/qubvel/segmentation_models.pytorch > /dev/null 2>&1 # Install segmentations_models.pytorch, with no bash output.')
import segmentation_models_pytorch as smp
model = smp.Unet('efficientnet-b0', encoder_weights='imagenet', classes=2)
print(model)




if CFG.debug:
    #20人分だけ..OK!
    folds = train.sample(50).reset_index(drop=True).copy()
else:
    folds = train.copy()




folds.head()




from sklearn.model_selection import StratifiedKFold,GroupKFold
kf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, folds["has_mask"])):
    folds.loc[val_index, 'fold'] = int(fold)

folds['fold'] = folds['fold'].astype(int)




folds.head()




from torch.nn import functional as F
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()




def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))




def dice_score(prob, truth, threshold=0.5):
    num = prob.size(0)
    prob= torch.sigmoid(prob)

    prob = prob > threshold
    truth = truth > 0.5

    prob = prob.view(num, -1)
    truth = truth.view(num, -1)
    intersection = (prob * truth)

    score = 2. * (intersection.sum(1) + 1.).float() / (prob.sum(1) + truth.sum(1) + 2.).float()
    score[score >= 1] = 1
    score = score.sum() / num
    return score




#SIIM2019の3rdで紹介されていた　https://github.com/bestfitting/kaggle/blob/master/siim_acr/src/layers/loss_funcs/loss.py

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

class SymmetricLovaszLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SymmetricLovaszLoss, self).__init__()
    def forward(self, logits, targets):
        return ((lovasz_hinge(logits, targets, per_image=True))                 + (lovasz_hinge(-logits, 1-targets, per_image=True))) / 2




def weighted_bce_loss(logit_pixel, truth_pixel):
    logit = logit_pixel.view(-1)#ここがfloatらしい...RuntimeError: result type Float can't be cast to the desired output type Byte
    truth = truth_pixel.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')#ここでエラー発生
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

    return loss

class Weighted_Bce_Loss(nn.Module):
    def __init__(self):
        super(Weighted_Bce_Loss, self).__init__()
    def forward(self, logits, targets):
        return weighted_bce_loss(logits, targets)
    
    
def soft_dice_loss(logit_pixel, truth_pixel):
    batch_size = len(logit_pixel)
    logit = logit_pixel.view(batch_size,-1)
    truth = truth_pixel.view(batch_size,-1)
    assert(logit.shape==truth.shape)

    loss = soft_dice_criterion(logit, truth)
    del logit, truth

    loss = loss.mean()
    return loss

def soft_dice_criterion(logit, truth, weight=[0.2,0.8]):

    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)
    w = truth.detach()
    w = w*(weight[1]-weight[0])+weight[0]

    p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice
    return loss

class Soft_Dice_Loss(nn.Module):
    def __init__(self):
        super(Soft_Dice_Loss, self).__init__()
    def forward(self, logits, targets):
        return soft_dice_loss(logits, targets)









def train_fn(fold):
    print(f"### fold: {fold} ###")
    
    
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    
    #タイルごとに拡張を入れる
    train_dataset = SIIMDataset(folds.loc[trn_idx].reset_index(drop=True))
    valid_dataset = SIIMDataset(folds.loc[val_idx].reset_index(drop=True))
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=8,pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=8,pin_memory=True)
    
    #model = smp.Unet('efficientnet-b0', encoder_weights='imagenet', classes=1)
    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=1)
    model.to(device)
    
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.lr, amsgrad=False)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
    
    #criterion = FocalLoss()
    criterion = SymmetricLovaszLoss()
    #criterion =Soft_Dice_Loss()
    #criterion = Weighted_Bce_Loss()#エラー発生
    best_score = 0
    best_loss = np.inf
    best_preds = None
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()


        model.train()
        avg_loss = 0.

        optimizer.zero_grad()
        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images,masks,labels) in tk0:

            images = images.to(device)
            masks = masks.to(device)
            #labels = labels.to(device)#0or1
            
            y_preds = model(images.float())
            
            
        
            loss = criterion(y_preds,masks)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        avg_val_loss = 0.
        preds = []
        valid_masks = []
        
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images,masks,labels) in tk1:
            
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                y_preds = model(images.float())
                
            
            valid_masks.append(masks.to('cpu'))
            loss = criterion(y_preds, masks)
            #print("valid_preds",y_preds.size())
            preds.append(y_preds.to('cpu'))
            
            avg_val_loss += loss.item() / len(valid_loader)
        
        #scheduler.step(avg_val_loss)
            
        #preds = np.concatenate(preds)
        #valid_masks = np.concatenate(valid_masks)
        
        preds = torch.cat(preds)
        #print(preds.size())
        valid_masks = torch.cat(valid_masks)
        
        score = dice_score(preds,valid_masks)

        elapsed = time.time() - start_time#loggerのためのもの
        
        LOGGER.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.debug(f'  Epoch {epoch+1} - dice_loss: {score}')
        
        if score>best_score:#QWKのスコアが良かったら予測値を更新...best_epochをきめるため
            best_score = score
            best_preds = preds
            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Score: {best_score:.4f}')
            torch.save(model.state_dict(), f'fold{fold}_efnet.pth')
    #各epochのモデルを保存。。。best_epoch終了時のモデルを推論に使用する？
    del model,preds
    return best_preds, valid_masks




len(folds)




preds = []
valid_masks = []
for fold in range(CFG.n_fold):
    _preds, _valid_masks = train_fn(fold)
    preds.append(_preds)
    valid_masks.append(_valid_masks)




del _preds, _valid_masksdel 




preds = torch.cat(preds)
valid_masks = torch.cat(valid_masks)
score = dice_score(preds,valid_masks)
print("DICE:",score)






