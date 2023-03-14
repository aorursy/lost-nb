#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorwatch')
get_ipython().system('pip install graphviz')
get_ipython().system('pip install pretrainedmodels')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os
import numpy as np
import math
import time
import random
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Parameter

from torchvision import transforms
import torchvision.models as models

import pretrainedmodels

import warnings
warnings.filterwarnings(action='ignore')
print(torch.__version__)


# In[3]:


print('Train set:')
for cls in os.listdir('../input/train/train'):
    print('{}:{}'.format(cls, len(os.listdir(os.path.join('../input/train/train', cls)))))
im = Image.open('../input/train/train/cgm/train-cgm-738.jpg')
print(im.size)


# In[4]:


Model = 'resnet18' # se_resnext50_32x4d || resnet18
Checkpoint = 'resnet18'
Loss = 'LabelSmoothSoftmaxCE' # FocalLoss || CrossEntropy || LabelSmoothSoftmaxCE

Freeze = True
Resume = False

Num_classes = 5
Size = 224 # image size
Batch_size = 256
Num_epochs = 300
Init_lr = 0.0001 
Step_size = 20 # StepLr decay rate

# warmup lr schedule
Multiplier = 80
Total_epoch = 20

# Focal Loss
Alpha = 0.25 # 0.3
Gamma = 1.5 # 2


# In[5]:


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# In[6]:


def RandomErasing(im, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[128, 128, 128]):
    '''
    performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    img: PIL img
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    if random.uniform(0, 1) > probability:
        return im

    else:
        img = np.array(im)
        area = img.shape[0] * img.shape[1]
       
        while True:
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if img.shape[0] > w and img.shape[1] > h:
                break

        if w < img.shape[0] and h < img.shape[1]:
            x1 = random.randint(0, img.shape[0] - w)
            y1 = random.randint(0, img.shape[1] - h)
            # if img.size()[0] == 3:
            if im.mode == 'RGB':
                img[x1:x1+h, y1:y1+w, 0] = mean[0]
                img[x1:x1+h, y1:y1+w, 1] = mean[1]
                img[x1:x1+h, y1:y1+w, 2] = mean[2]
            elif im.mode == 'L':
                img[x1:x1+h, y1:y1+w] = mean[0]
        img = Image.fromarray(np.uint8(img))

        return img


# In[7]:


# train
cls2label = {}
for label, cls in enumerate(os.listdir('../input/train/train')):
    cls2label[cls] = label
print(cls2label)

ims2labels = {}
ims2labels_train = {}
ims2labels_val = {}
for cls in os.listdir('../input/train/train'):
    im_num = len(os.listdir(os.path.join('../input/train/train', cls)))
    # total ims
    for im in os.listdir(os.path.join('../input/train/train', cls)):
        impath = os.path.join('../input/train/train', cls, im)
        ims2labels[impath] = cls2label[cls]
    val_ims = random.sample(os.listdir(os.path.join('../input/train/train', cls)), int(im_num*0.1))
    for im in val_ims:
        impath = os.path.join('../input/train/train', cls, im)
        ims2labels_val[impath] = cls2label[cls]
    for im in os.listdir(os.path.join('../input/train/train', cls)):
        if im not in val_ims:
            impath = os.path.join('../input/train/train', cls, im)
            ims2labels_train[impath] = cls2label[cls]
        
print('total:', list(ims2labels.items())[:5], len(list(ims2labels.items())))
print('train:', list(ims2labels_train.items())[:5], len(list(ims2labels_train.items())))
print('validation:', list(ims2labels_val.items())[:5], len(list(ims2labels_val.items())))

# test
df_test = pd.read_csv('../input/sample_submission_file.csv')
test_data = df_test['Id']

class CDCDataset(Dataset):
    def __init__(self, dataset, transform=None, mode='train', tta=False, idx=0):
        self.tta=tta
        self.idx = idx
        self.mode = mode
        if self.mode == 'train' or self.mode == 'val':
            self.ims, self.labels = [], []
            for item in dataset.items():
                self.ims.append(item[0])
                self.labels.append(item[1])
            # print(self.ims, self.labels)
        elif self.mode == 'test':
            self.im_names = dataset
            self.ims = [os.path.join('../input/test/test/0', im) for im in dataset]
        self.transform = transform

    def __getitem__(self, index):
        im_path = self.ims[index]
        if self.mode == 'train' or self.mode == 'val':
            label = self.labels[index]
        elif self.mode == 'test':
            im_name = self.im_names[index]
        im = Image.open(im_path)
        if self.mode == 'train' or self.mode == 'val':
            if self.transform is not None:
                im = RandomErasing(im, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[128, 128, 128])
                im = self.transform(im)
            return im, label
        elif self.mode == 'test':
            if self.tta:
                w, h = im.size
                if self.idx == 0:
                    im = im.crop((0, 0, int(w*0.9), int(h*0.9))) # top left
                elif self.idx == 1:
                    im = im.crop((int(w*0.1), 0, w, int(h*0.9))) # top right
                elif self.idx == 2:
                    im = im.crop((int(w*0.05), int(h*0.05), w-int(w*0.05), h-int(h*0.05))) # center
                elif self.idx == 3:
                    im = im.crop((0, int(h*0.1), w-int(w*0.1), h)) # bottom left
                elif self.idx == 4:
                    im = im.crop((int(w*0.1), int(h*0.1), w, h)) # bottom right
                elif self.idx == 5:
                    im = im.crop((0, 0, int(w*0.9), int(h*0.9))) 
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # top left and HFlip
                elif self.idx == 6:
                    im = im.crop((int(w*0.1), 0, w, int(h*0.9)))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # top right and HFlip
                elif self.idx == 7:
                    im = im.crop((int(w*0.05), int(h*0.05), w-int(w*0.05), h-int(h*0.05)))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # center and HFlip
                elif self.idx == 8:
                    im = im.crop((0, int(h*0.1), w-int(w*0.1), h))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # bottom left and HFlip
                elif self.idx == 9:
                    im = im.crop((int(w*0.1), int(h*0.1), w, h))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # bottom right and HFlip
            if self.transform is not None:
                im = self.transform(im)
            return im, im_name

    def __len__(self):
        return len(self.ims)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = CDCDataset(ims2labels, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=1, num_workers=0)
    #for im, loc, cls in dataloader_train:
    for data in dataloader_train:
        print(data)
        break


# In[8]:


def whitening(im):
    batch_size, channel, h, w = im.shape
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = torch.cat([(im[:,[0]]-mean[0])/std[0],
                    (im[:,[1]]-mean[1])/std[1],
                    (im[:,[2]]-mean[2])/std[2]], 1)
    return im

def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model

        #self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(128, num_classes)
        self.fc = nn.Linear(512, num_classes)
        
        self.conv_last = nn.Conv2d(512, num_classes, 1)
        

    def forward(self, x):
        # x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # print(x.size())
        x = self.backbone.avgpool(x)
        
        '''
        # FC
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)        
        x = self.fc(x)
        x = l2_norm(x)
        '''
        
        # Full conv
        #x = self.conv_last(x)
        x = x.view(x.size(0), -1)
        x = l2_norm(x)

        return x
    
#if __name__ == '__main__':
#    backbone = models.resnet18(pretrained=True)
#    models = ResNet18(backbone, 5)
#    data = torch.randn(1, 3, 224, 224)
#    x = models(data)
#    #print(x)
#    print(x.size())


# In[9]:


class se_resnext50_32x4d(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(se_resnext50_32x4d, self).__init__()
        self.backbone = model

        #self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(128, num_classes)
        self.fc = nn.Linear(2048, 1024)
        
        self.conv_last = nn.Conv2d(2048, num_classes, 3)

    def forward(self, x):
        # x = whitening(x)
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avg_pool(x)

        # FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
#         # Full conv
#         x = self.conv_last(x)
#         x = x.view(x.size(0), -1)
        
        x = l2_norm(x)

        return x
# if __name__ == '__main__':
#     backbone = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
#     models = se_resnext50_32x4d(backbone, 5)
#     data = torch.randn(1, 3, 224, 224)
#     x = models(data)
#     #print(x)
#     print(x.size())


# In[10]:


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = torch.ones(class_num, 1)*alpha
            else:
                # self.alpha = Variable(alpha).cuda()
                self.alpha = Variable(torch.ones(class_num, 1)*alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# In[11]:


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance:
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          s: norm of input feature
          m: margin
          cos(theta + m)
      """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


# In[12]:


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[13]:


# validation
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1


# In[14]:


# inference
def inference(model):    
    # test data
    test_transform = transforms.Compose([transforms.Resize((int(Size), int(Size))),
                        #transforms.TenCrop(Size),
                        #Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])])
    dst_test = CDCDataset(test_data, transform=test_transform, mode='test')
    dataloader_test = DataLoader(dst_test, shuffle=False, batch_size=Batch_size//2, num_workers=8)

    
    model.eval()
    results = []
    print('Inferencing ...')
    for ims, im_names in dataloader_test:
        input = Variable(ims).cuda()
        output = model(input)
        _, preds = output.topk(1, 1, True, True)
        preds = preds.cpu().detach().numpy()
        for pred, im_name in zip(preds, im_names):
            top1_name = [list(cls2label.keys())[list(cls2label.values()).index(p)] for p in pred]
            results.append({'Id':im_name, 'Category':''.join(top1_name)})
    df = pd.DataFrame(results, columns=['Category', 'Id'])
    df.to_csv('sub.csv', index=False)
def inference_TTA(model):   
    # print(model)
    # test data
    test_transform = transforms.Compose([transforms.Resize((int(Size), int(Size))),
                        #transforms.TenCrop(Size),
                        #Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])])
    # 10 TTA
    TTA10 = []
    results = []
    print('Inferencing with TTA ...')
    for idx in range(10):
        dst_test = CDCDataset(test_data, transform=test_transform, mode='test', tta=True, idx=idx)
        dataloader_test = DataLoader(dst_test, shuffle=False, batch_size=Batch_size//2, num_workers=0)


        model.eval().cuda()
        names2probs = {}
        for ims, im_names in dataloader_test:
            input = Variable(ims).cuda()
            output = model(input)
            probs = F.softmax(output)
            probs = probs.cpu().detach().numpy()
            for prob, im_name in zip(probs, im_names):
                names2probs[im_name] = prob
        TTA10.append(names2probs)
    for im_name in TTA10[0].keys():
        prob = (TTA10[0][im_name]+TTA10[1][im_name]+TTA10[2][im_name]+TTA10[3][im_name]+TTA10[4][im_name]               +TTA10[5][im_name]+TTA10[6][im_name]+TTA10[7][im_name]+TTA10[8][im_name]+TTA10[9][im_name])/10
        top1_idx = prob.argsort()[-1]
        top1_name = list(cls2label.keys())[list(cls2label.values()).index(top1_idx)]
        results.append({'Id':im_name, 'Category':''.join(top1_name)})
    df = pd.DataFrame(results, columns=['Category', 'Id'])
    df.to_csv('sub_tta10.csv', index=False)


# In[15]:


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# In[16]:


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)
    
    inten = torch.randn(10, 5).cuda()
    lbs = torch.randint(5, (10,)).cuda()
    print('inten:', inten)
    print('lbs:', lbs)

    import torch.nn.functional as F

    loss = criteria(inten, lbs)
    print('loss:', loss)


# In[17]:


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# In[18]:


def train():
    begin_time = time.time()
    # model
    if Model == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=Num_classes)
        metric_fc = ArcMarginProduct(512, Num_classes, s=30, m=0.5, easy_margin=False)
    elif Model == 'se_resnext50_32x4d':
        backbone = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
        model = se_resnext50_32x4d(backbone, 5)
        metric_fc = ArcMarginProduct(1024, Num_classes, s=30, m=0.5, easy_margin=False)
    # print(model)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    metric_fc.cuda()
    
    # freeze layers
    if Freeze:
        if Model == 'se_resnext50_32x4d':
            for p in model.backbone.layer0.parameters(): p.requires_grad = False
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False
        # for p in model.backbone.layer4.parameters(): p.requires_grad = False

    # train data
    train_transform = transforms.Compose([transforms.Scale(256),
                                    transforms.RandomSizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(30),
                                    transforms.Resize((Size, Size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_train = CDCDataset(ims2labels, transform=train_transform) # ims2labels_train
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=Batch_size, num_workers=8)

#     # validation data
#     test_transform = transforms.Compose([transforms.Resize((Size, Size)),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                                               std=[0.229, 0.224, 0.225])])
#     dst_valid = CDCDataset(ims2labels_val, transform=test_transform)
#     dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=Batch_size//2, num_workers=8)

    # load checkpoint
    if Resume:
        model = torch.load(os.path.join('./checkpoints', Checkpoint))

    # train
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    
    # loss
    if Loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss().cuda()
    elif Loss == 'FocalLoss':
        criterion = FocalLoss(Num_classes, alpha=Alpha, gamma=Gamma, size_average=True)
    elif Loss == 'LabelSmoothSoftmaxCE':
        criterion = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=0.05) # lb_neg=5e-3
    criterion.cuda()
    
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters())}, {'params': metric_fc.parameters()}], 
                                 lr=Init_lr, betas=(0.9, 0.999), weight_decay=0.0002)
    # lr schedule
    # scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Step_size, gamma=0.1, last_epoch=-1)
    # warmup schedule
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Num_epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=Multiplier, total_epoch=Total_epoch, after_scheduler=scheduler_cosine)
    
    
    print('Start training ...')
    train_loss_list, val_loss_list = [], []
    train_top1_list, val_top1_list = [], []
    lr_list = []
    for epoch in range(Num_epochs):
        # scheduler_steplr.step()
        scheduler_warmup.step()
#         print('Inference testset...')
#         inference(model)
        ep_start = time.time()
        # val_loss, val_top1 = eval(model, dataloader_valid, criterion)
        model.train()
        top1_sum = 0
        for i, (ims, labels) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(labels).cuda().long()

            feature = model(input)
            output = metric_fc(feature, target)
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1 = accuracy(output.data, target.data, topk=(1,))
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += top1[0]
            sum += 1
            top1_sum += top1[0]
        
        lr = optimizer.state_dict()['param_groups'][0]['lr']
#         print('Epoch [%d/%d]  |  lr: %f  |  train@loss: %.4f  train@top1: %.4f  |  val@loss:%.4f  val@top1:%.4f  |  time:%.4f s'
#                %(epoch+1, Num_epochs, lr, train_loss_sum/sum, train_top1_sum/sum, val_loss, val_top1, time.time()-ep_start))
        print('Epoch [%d/%d]  |  lr: %f  |  train@loss: %.4f  train@top1: %.4f  |  time:%.4f s'
               %(epoch+1, Num_epochs, lr, train_loss_sum/sum, train_top1_sum/sum, time.time()-ep_start))
        train_loss_list.append(train_loss_sum/sum)
        # val_loss_list.append(val_loss)
        train_top1_list.append(train_top1_sum/sum)
        # val_top1_list.append(val_top1)
        lr_list.append(lr)
        
        sum = 0
        train_loss_sum = 0
        train_top1_sum = 0
        
        if (epoch+1) % 50 == 0 and epoch < Num_epochs or (epoch+1) == Num_epochs:
            print('Taking snapshot...')
            torch.save(model, '{}.pth'.format(Checkpoint))
            
        if (time.time()-begin_time)/60/60 > 8:
            break
    
    inference(model)
    inference_TTA(model)
    # draw curve
    figs = plt.figure()
    fig1 = figs.add_subplot(3, 1, 1)
    fig2 = figs.add_subplot(3, 1, 2)
    fig3 = figs.add_subplot(3, 1, 3)
    x = [i for i in range(len(train_loss_list))]
    fig1.plot(x, train_loss_list, label='train loss')
    #     fig1.plot(x, val_loss_list, label='valid loss')
    fig1.legend(loc='upper right')

    fig2.plot(x, train_top1_list, label='train loss')
    #     fig2.plot(x, val_top1_list, label='valid loss')
    fig2.legend(loc='bottom right')

    fig3.plot(x, lr_list, label='lr')
    fig3.legend(loc='upper right')

    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    plt.show()


# In[19]:


if __name__ == '__main__':
    setup_seed(88)
    train()


# In[20]:


get_ipython().system('nvidia-smi')

