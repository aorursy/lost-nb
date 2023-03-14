#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import sys

import librosa
#import audioread
import soundfile
import torch
import random
from torch.utils.data import Dataset, DataLoader

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold

import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback, ReduceLROnPlateauCallback
from torch.utils.data import Dataset, DataLoader
from radam import *
from mish_activation import *
import glob
#from torchlibrosa.augmentation import SpecAugmentation


# In[2]:


PATH = ['../input/birdsong-resampled-train-audio-00/','../input/birdsong-resampled-train-audio-01/',
         '../input/birdsong-resampled-train-audio-02/','../input/birdsong-resampled-train-audio-03/',
         '../input/birdsong-resampled-train-audio-04/']
LABELS = '../input/birdsong-recognition/train.csv'
NUM_WORKERS = 12
nfolds = 4
SEED = 2020
OUT = 'model0'
bs = 64#48

class config:
    sampling_rate = 32000
    duration = 5#20.03
    samples = int(sampling_rate*duration)
    top_db = 60 # Noise filtering, default = 60
    
    # Frequencies kept in spectrograms
    fmin = 50
    fmax =  14000

    # Spectrogram parameters
    n_mels = 128 # = spec_height
    n_fft = 1024
    hop_length = 313
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

os.makedirs(OUT, exist_ok=True)
seed_everything(SEED)


# In[3]:


files = []
for p in PATH:
    files += glob.glob(p + '/*/*.wav')
files = {os.path.basename(f):f for f in files}


# In[4]:


df = pd.read_csv(LABELS)
label_map = {p:i for i,p in enumerate(sorted(df.ebird_code.unique()))}
df['label'] = df.ebird_code.map(label_map)

splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
df['split'] = 0
for i,s in enumerate(list(splits.split(df,df.label))):
    df.loc[s[1],'split'] = i
df['filename_w'] = [f[:-3] + 'wav' for f in df.filename]
df = df.loc[df.filename_w.isin(files.keys())].reset_index()
df.head()


# In[5]:


mean,std = -40.0,12.0 #quick estimation

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class BirdDataset(Dataset):
    def __init__(self, df, fold=0, train=True, tfms=None):
        self.df = df.copy()
        self.df = self.df.loc[self.df.split != fold] if train else self.df.loc[self.df.split == fold]
        self.df = self.df.reset_index(drop=True)
        self.train = train
        self.tfms = tfms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label,fname = self.df.iloc[idx][['label','filename_w']]
        #sample according to length
        #if self.train:
        #    tmp_df = self.df.loc[self.df.label == label, 'duration']
        #    duration = np.sqrt(tmp_df.values)
        #    idx = np.random.choice(tmp_df.index, 1, p=duration/duration.sum())
        #    fname = self.df.iloc[idx]['filename_w'].item()
        #fname = os.path.join(PATH,fname)
        fname = files[fname]
        
        l = soundfile.info(fname).frames
        while l == 0: #there are corrupted files
            idx = np.random.randint(len(self.df))
            label,fname = self.df.iloc[idx][['label','filename_w']]
            fname = os.path.join(PATH,fname)
            l = soundfile.info(fname).frames   
                
        effective_length = config.samples
        if l <= effective_length:
            wave, sr = soundfile.read(fname)
            new_wave = np.zeros(effective_length, dtype=wave.dtype)
            start = np.random.randint(effective_length - len(wave)) if effective_length > len(wave) else 0
            new_wave[start:start + l] = wave
            wave = new_wave.astype(np.float32)
        else:
            start = np.random.randint(l - effective_length) if l > effective_length else 0
            wave, sr = soundfile.read(fname,start=start,stop=start+effective_length)
        wave= wave.astype(np.float32)

        #norm wave and random rescale
        #wave = wave*(0.025/max(wave.std(),0.01))
        #if self.train: wave = wave*max(np.random.normal(1.0, 0.2),0.3)
            
        mel = librosa.feature.melspectrogram(wave, 
                    sr=config.sampling_rate,
                    n_mels=config.n_mels,
                    hop_length=config.hop_length,
                    n_fft=config.n_fft,
                    fmin=config.fmin,
                    fmax=config.fmax)
        logmel = librosa.power_to_db(mel).astype(np.float32)
        return (img2tensor(logmel) - mean)/std, label


# In[6]:


class AttnBlock(nn.Module):
    def __init__(self, n=512, nheads=8, dim_feedforward=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(n,nheads)
        self.norm = nn.LayerNorm(n)
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0],shape[1],-1).permute(2,0,1)
        x = self.norm(self.drop(self.attn(x,x,x)[0]) + x)
        x = x.permute(1,2,0).reshape(shape)
        return x    

class Model(nn.Module):
    def __init__(self, n=len(label_map), arch='resnext50_32x4d_ssl', 
                 path='facebookresearch/semi-supervised-ImageNet1K-models', ps=0.5):
        super().__init__()
        m = torch.hub.load(path, arch)
        nc = list(m.children())[-1].in_features
        self.enc = nn.Sequential(*list(m.children())[:-2])
        
        shape = self.enc[0].weight.shape
        w = self.enc[0].weight.sum(1).unsqueeze(1)
        self.enc[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.enc[0].weight = nn.Parameter(w)

        nh = 768
        self.head = nn.Sequential(nn.Conv2d(nc,nh,(config.n_mels//32,1)),AttnBlock(nh),AttnBlock(nh),
                                  nn.Conv2d(nh,n,1))
        
    def forward(self, x):
        x = self.head(self.enc(x))
        #bs,n,1,len//32
        return torch.logsumexp(x,-1).squeeze() - torch.Tensor([x.shape[-1]]).to(x.device).log()


# In[7]:


class OneHot(Callback):
    def __init__(self, nunique=len(label_map)):
        super().__init__()
        self.nunique = nunique
        
    def on_batch_begin(self, last_target, **kwargs):
        last_target = F.one_hot(last_target.long(), self.nunique).float()
        return {'last_target': last_target}
    
#correct implementation of focal loss for soft labels
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target, reduction='mean'):
        n = input.shape[-1]
        input = input.view(-1).float()
        target = target.view(-1).float()
        loss = -target*F.logsigmoid(input)*torch.exp(self.gamma*F.logsigmoid(-input)) -           (1.0 - target)*F.logsigmoid(-input)*torch.exp(self.gamma*F.logsigmoid(input))
        
        return n*loss.mean() if reduction=='mean' else loss
    
def acc_m(x,y):
    return (x.argmax(-1) == y.argmax(-1)).float().mean()

class FBetaMax(Callback):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.preds = []
        self.targets = []
        
    def on_epoch_begin(self, **kwargs):
        self.preds = []
        self.targets = []
    
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        self.preds.append(last_output.cpu())
        self.targets.append(last_target.cpu())
    
    def on_epoch_end(self, last_metrics, **kwargs):
        p = torch.cat(self.preds,0)
        t = torch.cat(self.targets,0)
        th1,th2 = p.min(), p.max()
        nth = 1000
        metric = torch.stack([fbeta(p,t,thresh=th1+(th2-th1)*i/(nth-1),
                                    beta=self.beta,sigmoid=False) for i in range(nth)]).max()

        return add_metrics(last_metrics, metric)
    
class AccMax(Callback):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.preds = []
        self.targets = []
        
    def on_epoch_begin(self, **kwargs):
        self.preds = []
        self.targets = []
    
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        self.preds.append(last_output.cpu())
        self.targets.append(last_target.cpu())
        
    def metric(self,p,t,th):
        return ((p > th).long() == t.long()).min(-1)[0].float().mean()
    
    def on_epoch_end(self, last_metrics, **kwargs):
        p = torch.cat(self.preds,0)
        t = torch.cat(self.targets,0)
        th1,th2 = p.min(), p.max()
        nth = 1000
        metric = torch.stack([self.metric(p,t,th1+(th2-th1)*i/(nth-1)) for i in range(nth)]).max()

        return add_metrics(last_metrics, metric)   


# In[8]:


def mixup(x, y, alpha=0.4):
    gamma = np.random.beta(alpha, alpha)
    gamma = max(1-gamma, gamma)
    shuffle = torch.randperm(x.shape[0]).to(x.device)
    x = gamma*x + (1-gamma)*x[shuffle]
    y = gamma*y + (1-gamma)*y[shuffle]
    return x, y

def cutmix(x, ys, alpha=0.4):
    gamma = np.random.beta(alpha, alpha)
    gamma = max(1-gamma, gamma)
    shuffle = torch.randperm(x.shape[0]).to(x.device)
    ys_shuffle = [y[shuffle] for y in ys]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), gamma)
    x[..., bbx1:bbx2, bby1:bby2] = x[shuffle][..., bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y = lam*y + (1-lam)*y[shuffle]
    return x, y
  
class Mixup(LearnerCallback):
    def __init__(self, learn, alpha=0.4):
        super().__init__(learn)
        self.alpha = alpha
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #    freq_drop_width=8, freq_stripes_num=0)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return
        bs = last_input.shape[0]
        #last_input = self.spec_augmenter(last_input)
        last_input, last_target = mixup(last_input, last_target, self.alpha)
        return {'last_input': last_input, 'last_target': last_target}
    
class Cutmix(LearnerCallback):
    def __init__(self, learn, alpha=1.0):
        super().__init__(learn)
        self.alpha = alpha
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return
        bs = last_input.shape[0]
        last_input, last_target = cutmix(last_input, last_target, self.alpha)
        return {'last_input': last_input, 'last_target': last_target}


# In[9]:


#5s
fname = 'model0'
for fold in range(1):
    ds_t = BirdDataset(df, fold=fold, train=True)
    ds_v = BirdDataset(df, fold=fold, train=False)
    dl_t = DataLoader(ds_t,bs,num_workers=NUM_WORKERS,shuffle=True)
    dl_v = DataLoader(ds_v,bs,num_workers=NUM_WORKERS)
    data = DataBunch(dl_t,dl_v)
    model = Model()
    model = nn.DataParallel(model)
    learn = Learner(data, model, loss_func=FocalLoss(), opt_func=partial(Over9000,eps=1e-4),
            metrics=[FBetaMax(),acc_m,AccMax()],).to_fp16(clip=1.0,max_noskip=100)
    learn.callbacks.append(OneHot())
    learn.callbacks.append(Mixup(learn))
    learn.split([model.module.head])
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, max_lr=1e-2, div_factor=5, pct_start=0.0)
    learn.unfreeze()
    #learn.callbacks.append(ReduceLROnPlateauCallback(learn=learn, monitor='f_beta_max', mode='max',
    #                patience=2, factor=0.85, min_lr=1e-5))
    #learn.fit(64, lr=0.75e-3, wd=1e-3, callbacks = [SaveModelCallback(learn,name=f'model',monitor='f_beta_max')])
    learn.fit_one_cycle(16, max_lr=(1e-3,1e-2), div_factor=50, pct_start=0.0, 
          callbacks = [SaveModelCallback(learn,name=f'model',monitor='f_beta_max')])
    torch.save(learn.model.module.state_dict(),os.path.join(OUT,f'{fname}_{fold}.pth'))


# In[ ]:




