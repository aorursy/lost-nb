#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

*()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
*/

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# In[4]:


path = Path('../input/')


# In[5]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)


# In[6]:


print(torch.cuda.is_available())


# In[7]:


df_train = pd.read_csv(path/'train.csv')
df_train.head()


# In[8]:


# Image porcessed with size (224,224) as previously built models have performed well
import PIL
import cv2
IMG_SIZE = 224


# In[9]:


from PIL import Image

def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


# In[10]:


from fastai.vision import Image

def _load_format(path,convert_mode, after_open) -> Image :
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0), 10) ,-4 ,128)
    img_fastai = Image((pil2tensor(image, np.float32)).div_(255)) #fastai Image format
    
    return (img_fastai)

vision.data.open_image = _load_format


# In[11]:


src = (ImageList.from_csv(path, 'train.csv', folder='train_images', suffix='.png')
       .split_by_rand_pct(0.2)
       .label_from_df(cols='diagnosis',label_cls=FloatList))
src


# In[12]:


tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      max_rotate=360)

tfms


# In[13]:


data = (src.transform(tfms,size=224,resize_method=ResizeMethod.SQUISH,padding_mode='reflection')
        .databunch().normalize(imagenet_stats))
data


# In[14]:


data.show_batch(rows=2, figsize=(12,9))


# In[15]:


# Definition of Quadratic Kappa
from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')


# In[16]:


learn = cnn_learner(data, models.densenet201, metrics=[quadratic_kappa],model_dir='/kaggle')


# In[17]:


# Find a good learning rate
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[18]:


lr = 1.20E-02
lr


# In[19]:


learn.fit_one_cycle(4,lr)


# In[20]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[21]:


# Save the model
learn.save("stage-1", return_path=True)


# In[22]:


# Unfreeze and finding best LR
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[23]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[24]:


# Save the model
learn.save("stage-2", return_path=True)


# In[25]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[26]:


import scipy as sp
from functools import partial
from sklearn import metrics


# In[27]:


valid_preds = learn.get_preds(ds_type=DatasetType.Valid)


# In[28]:


# ref: https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa , thanks Abhishek Thakur

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[29]:


optR = OptimizedRounder()
optR.fit(valid_preds[0],valid_preds[1])


# In[30]:


coefficients = optR.coefficients()
print(coefficients)


# In[31]:


# test_df = pd.read_csv(path/'test.csv')
# test_df.head()
sample_df = pd.read_csv(path/'sample_submission.csv')
sample_df.head()


# In[32]:


learn.data.add_test(ImageList.from_df(sample_df,path,folder='test_images',suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
test_predictions = optR.predict(preds, coefficients)


# In[33]:


sample_df.diagnosis = test_predictions.astype(int)
sample_df.head()


# In[34]:


sample_df.to_csv('submission.csv',index=False)

