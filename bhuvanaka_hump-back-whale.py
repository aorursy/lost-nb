#!/usr/bin/env python
# coding: utf-8



# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

PATH = "../input/humpback-whale-identification/"

MODEL_NAME = 'resnet34'
TRAIN = '../input/humpback-whale-identification/train/'
TEST = '../input/humpback-whale-identification/test/'
LABELS = '../input/humpback-whale-identification/train.csv'
SAMPLE_SUB = '../input/humpback-whale-identification/sample_submission.csv'




print(os.listdir("../input/humpback-whale-identification/test/"))
      




from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
import numpy as np
import torch
from utils import *
print(os.listdir("../input/"))




get_ipython().system('pip show fastai')




cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# copy time!
get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')




def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)




def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])




def map5(preds, targs):
    predicted_idxs = preds.sort(descending=True)[1]
    top_5 = predicted_idxs[:, :5]
    res = mapk([[t] for t in targs.cpu().numpy()], top_5.cpu().numpy(), 5)




def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels




def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]




def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels




def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')




get_ipython().system('pip show fastai')





get_ipython().system('pwd')
get_ipython().system('ls')




df = pd.read_csv('../input/humpback-whale-identification/train.csv')
df.head()




df.Id.value_counts().head()




(df.Id == 'new_whale').mean()




(df.Id.value_counts() == 1).mean()




df.Id.nunique()




df.shape




fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}




SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0




data = (
    ImageItemList
        .from_folder(TRAIN)
        .random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path.name])
        .add_test(ImageItemList.from_folder(TEST))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS)
)




data.show_batch(rows=3)




name = f'res34-{SZ}'
import pathlib
data.path = pathlib.Path('.')




learn = create_cnn(data, models.resnet34, metrics=[accuracy, map5])




learn.fit_one_cycle(2)




learn.recorder.plot_losses()




learn.save(f'{name}-stage-1')




learn.unfreeze()




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

