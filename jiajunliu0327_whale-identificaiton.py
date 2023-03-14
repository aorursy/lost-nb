#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd

from utils import *

df = pd.read_csv('data/train.csv')
df.head()

df.Id.value_counts().head()

(df.Id == 'new_whale').mean()

(df.Id.value_counts() == 1).mean()

df.Id.nunique()

df.shape

fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}

SZ = 224
BS = 64
NUM_WORKERS = 12
SEED=0

data = (
    ImageList
        .from_folder('data/train-224')
        .random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path.name])
        .add_test(ImageList.from_folder('data/test-224'))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
)

data.show_batch(rows=3)

name = f'res50-{SZ}'

learn = create_cnn(data, models.resnet50, metrics=[accuracy, map5])

learn.fit_one_cycle(1)

learn.recorder.plot_losses()

learn.save(f'{name}-stage-1')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

max_lr = 1e-4
lrs = [max_lr/100, max_lr/10, max_lr]

learn.fit_one_cycle(5, lrs)

learn.save(f'{name}-stage-2')

learn.recorder.plot_losses()

preds, _ = learn.get_preds(DatasetType.Test)

mkdir -p subs

create_submission(preds, learn.data, name)

pd.read_csv(f'subs/{name}.csv.gz').head()

get_ipython().system('kaggle competitions submit -c humpback-whale-identification -f subs/{name}.csv.gz -m "{name}"')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
import re

from utils import *


# In[3]:


df = pd.read_csv('data/train.csv')
df.head()


# In[4]:


(df.Id != 'new_whale').mean()


# In[5]:


im_count = df[df.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
df = df.join(im_count, on='Id'); df.head()


# In[6]:


val_fns = set(df.sample(frac=1)[(df.Id != 'new_whale') & (df.sighting_count > 1)].groupby('Id').first().Image)


# In[7]:


len(val_fns)


# In[8]:


val_fns = val_fns.union(set(df[df.Id == 'new_whale'].sample(frac=1).Image.values[:1000]))


# In[9]:


len(val_fns)


# In[10]:


fn2label = {row[1].Image: 'new_whale' if row[1].Id == 'new_whale' else 'known_whale' for row in df.iterrows()}


# In[11]:


SZ = 224
BS = 64
NUM_WORKERS = 12
SEED=0


# In[12]:


path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)


# In[13]:


data = (
    ImageItemList
        .from_df(df, 'data/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('data/test'))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)


# In[14]:


data


# In[15]:


data.show_batch(rows=3)


# In[16]:


name = f'res50-{SZ}'


# In[17]:


learn = create_cnn(data, models.resnet50, metrics=[accuracy])


# In[18]:


learn.lr_find()


# In[19]:


learn.recorder.plot()


# In[20]:


learn.fit_one_cycle(4, 1e-2)


# In[21]:


learn.save(f'{name}-stage-1')


# In[22]:


learn.unfreeze()


# In[23]:


learn.lr_find()


# In[24]:


learn.recorder.plot()


# In[25]:


learn.fit_one_cycle(4, [1e-6, 1e-5, 1e-4])


# In[26]:


learn.save(f'{name}-stage-2')


# In[27]:


preds, targs = learn.get_preds()


# In[28]:


learn.data.classes


# In[29]:


# https://en.wikipedia.org/wiki/Precision_and_recall
tp = ((preds.argmax(1) == 1).long() * targs).sum()
tn = ((preds.argmax(1) == 0).long() * (targs==0).long()).sum()
fn = ((preds.argmax(1) == 0).long() * targs).sum()
fp = ((preds.argmax(1) == 1).long() * (targs==0).long()).sum()


# In[30]:


# recall of new_whale
tp/(tp+fn).float()


# In[31]:


# precision
tp/(tp+fp).float()


# In[32]:


# accuracy
(tp+tn)/(tp+tn+fp+fn).float()


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
import pandas as pd
from torch import optim
import re

from utils import *

df = pd.read_csv('data/train.csv')
df.head()
im_count = df[df.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
df = df.join(im_count, on='Id')
val_fns = set(df.sample(frac=1)[(df.Id != 'new_whale') & (df.sighting_count > 1)].groupby('Id').first().Image)

# pd.to_pickle(val_fns, 'data/val_fns')
val_fns = pd.read_pickle('data/val_fns')

fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}

SZ = 224
BS = 64
NUM_WORKERS = 12
SEED=0

path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)

df = df[df.Id != 'new_whale']

df.shape

df.sighting_count.max()

df_val = df[df.Image.isin(val_fns)]
df_train = df[~df.Image.isin(val_fns)]
df_train_with_val = df

df_val.shape, df_train.shape, df_train_with_val.shape

%%time

res = None
sample_to = 15

for grp in df_train.groupby('Id'):
    n = grp[1].shape[0]
    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)
    rows = pd.concat((grp[1], additional_rows))
    
    if res is None: res = rows
    else: res = pd.concat((res, rows))
        
%%time

res_with_val = None
sample_to = 15

for grp in df_train_with_val.groupby('Id'):
    n = grp[1].shape[0]
    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)
    rows = pd.concat((grp[1], additional_rows))
    
    if res_with_val is None: res_with_val = rows
    else: res_with_val = pd.concat((res_with_val, rows))

res.shape, res_with_val.shape

pd.concat((res, df_val))[['Image', 'Id']].to_csv('data/oversampled_train.csv', index=False)
res_with_val[['Image', 'Id']].to_csv('data/oversampled_train_and_val.csv', index=False)

df = pd.read_csv('data/oversampled_train.csv')

data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], 'data/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('data/test'))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)

data


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
import pandas as pd
from torch import optim
import re

from utils import *

import fastai
from fastprogress import force_console_behavior
import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

df = pd.read_csv('data/train.csv')
val_fns = {'69823499d.jpg'}

fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)

name = f'res50-full-train'

SZ = 224
BS = 64
NUM_WORKERS = 12
SEED=0

data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], 'data/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('data/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)

%%time

learn = create_cnn(data, models.resnet50, lin_ftrs=[2048])
learn.clip_grad();

learn.fit_one_cycle(14, 1e-2)
learn.save(f'{name}-stage-1')

learn.unfreeze()

max_lr = 1e-3
lrs = [max_lr/100, max_lr/10, max_lr]

learn.fit_one_cycle(24, lrs)
learn.save(f'{name}-stage-2')

SZ = 224 * 2
BS = 64 // 4
NUM_WORKERS = 12
SEED=0

data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], 'data/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('data/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)

%%time
learn = create_cnn(data, models.resnet50, lin_ftrs=[2048])
learn.clip_grad();
learn.load(f'{name}-stage-2')
learn.freeze_to(-1)

learn.fit_one_cycle(12, 1e-2 / 4)
learn.save(f'{name}-stage-3')

learn.unfreeze()

max_lr = 1e-3 / 4
lrs = [max_lr/100, max_lr/10, max_lr]

learn.fit_one_cycle(22, lrs)
learn.save(f'{name}-stage-4')

# with oversampling
df = pd.read_csv('data/oversampled_train_and_val.csv')

data = (
    ImageItemList
        .from_df(df, 'data/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('data/test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)

%%time
learn = create_cnn(data, models.resnet50, lin_ftrs=[2048])
learn.clip_grad();
learn.load(f'{name}-stage-4')
learn.freeze_to(-1)

learn.fit_one_cycle(2, 1e-2 / 4)
learn.save(f'{name}-stage-5')

learn.unfreeze()

max_lr = 1e-3 / 4
lrs = [max_lr/100, max_lr/10, max_lr]

learn.fit_one_cycle(3, lrs)
learn.save(f'{name}-stage-6')

preds, _ = learn.get_preds(DatasetType.Test)

preds = torch.cat((preds, torch.ones_like(preds[:, :1])), 1)

preds[:, 5004] = 0.06
classes = learn.data.classes + ['new_whale']

create_submission(preds, learn.data, name, classes)

pd.read_csv(f'subs/{name}.csv.gz').head()

pd.read_csv(f'subs/{name}.csv.gz').Id.str.split().apply(lambda x: x[0] == 'new_whale').mean()

get_ipython().system('kaggle competitions submit -c humpback-whale-identification -f subs/{name}.csv.gz -m "{name}"')


# In[35]:




