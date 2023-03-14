#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
PATH = '/kaggle/input/bengaliai-cv19/'
# Any results you write to the current directory are saved as output.




get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
#from csvlogger import *
#from radam import *
#from mish_activation import *
import warnings
warnings.filterwarnings("ignore")

fastai.__version__




import cv2
import zipfile
from tqdm import tqdm_notebook as tqdm
import random
import torchvision

SEED = 42
LABELS = 'train.csv'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)




HEIGHT = 137
WIDTH = 236
SIZE = (128,128)
BATCH = 128

TRAIN = [PATH+'train_image_data_0.parquet',
         PATH+'train_image_data_1.parquet',
         PATH+'train_image_data_2.parquet',
         PATH+'train_image_data_3.parquet']




df_label = pd.read_csv(PATH+LABELS)
nunique = list(df_label.nunique())[1:-1]
print(nunique)
df_label['components'] = 'r_'+df_label['grapheme_root'].astype(str)+','                         +'v_'+df_label['vowel_diacritic'].astype(str)+','                         +'c_'+df_label['consonant_diacritic'].astype(str)
df_label.head()




stats128, stats137, fold, nfolds = ([0.08547], [0.22490]), ([0.06922], [0.20514]), 0, 4
FOLDER = '../input/bengali-grapheme'

src = (ImageList.from_df(df_label, path='.', folder=FOLDER, suffix='.png', cols='image_id')
       .split_by_idx(range(fold*len(df_label)//nfolds,(fold+1)*len(df_label)//nfolds))
        #.split_from_df(col='is_valid')
        .label_from_df(cols=['components'],label_delim=','))




data = (src.transform(get_transforms(do_flip=False,max_warp=0.1), size=SIZE, padding_mode='zeros')
        .databunch(bs=BATCH)
        .normalize(imagenet_stats))

data.show_batch()




# Model 
arch = models.resnet34

acc_02 = partial(accuracy_thresh)
f_score = partial(fbeta)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])




learn.lr_find() 
learn.recorder.plot() 




lr = 0.03




learn.fit_one_cycle(6, slice(lr))




learn.unfreeze()




learn.lr_find()
learn.recorder.plot()




learn.fit_one_cycle(5, slice(1e-5,lr/5))




learn.export(Path('/kaggle/working')/'try3-rn34-im128.pkl')

