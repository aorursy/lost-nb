#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Reloads required modules
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#This will include the plots generated by matplotlib in your notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Imports everything needed
from fastai.conv_learner import *


# In[ ]:


#Change, if necessary. For this notebook, you don't have to!
PATH = '../input/planet-understanding-the-amazon-from-space/'


# In[ ]:


ls {PATH}


# In[ ]:


from fastai.plots import *


# In[ ]:


def get_1st(path, pattern): 
    return glob(f'{path}/*{pattern}.*')[0]


# In[ ]:


dc_path = "../input/dogs-vs-cats-redux-kernels-edition/train"
list_paths = [get_1st(f"{dc_path}", "cat"), get_1st(f"{dc_path}", "dog")]
plots_from_files(list_paths, titles=["cat", "dog"], maintitle="Single-label classification")


# In[ ]:


list_paths = [f"{PATH}train-jpg/train_0.jpg", f"{PATH}train-jpg/train_1.jpg"]
titles=["haze primary", "agriculture clear primary water"]
plots_from_files(list_paths, titles=titles, maintitle="Multi-label classification")


# In[ ]:


# planet.py

from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])


# In[ ]:


#metrics and the model used
metrics=[f2]
f_model = resnet34


# In[ ]:


#The csv file where we have our labels
label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)


# In[ ]:


'''
Top down augmentation
sz : data loader  size
'''
def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg-v2')


# In[ ]:


#get image of size 64x64
data = get_data(64)


# In[ ]:


#data loader
x,y = next(iter(data.val_dl))


# In[ ]:


print(np.shape(x))


# In[ ]:


#labels
y


# In[ ]:


list(zip(data.classes, y[0]))


# In[ ]:


plt.imshow(data.val_ds.denorm(to_np(x))[1]);


# In[ ]:


# Image looks washed out. So, we multiple the image with 1.4 that increases teh contrast
plt.imshow(data.val_ds.denorm(to_np(x))[1]*1.4);


# In[ ]:


data = data.resize(int(sz*1.3), '/tmp')


# In[ ]:


TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"


# In[ ]:


learn = ConvLearner.pretrained(f_model, data, metrics=metrics, tmp_name=TMP_PATH, models_name=MODEL_PATH)


# In[ ]:


lrf=learn.lr_find()
learn.sched.plot()


# In[ ]:


lr = 0.2


# In[ ]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


lrs = np.array([lr/9,lr/3,lr])


# In[ ]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.save(f'{sz}')


# In[ ]:


learn.sched.plot_loss()


# In[ ]:


sz=128


# In[ ]:


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


# In[ ]:


sz=256


# In[ ]:


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


# In[ ]:


multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)


# In[ ]:


f2(preds,y)


# In[ ]:





# In[ ]:




