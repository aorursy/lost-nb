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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




from fastai.vision import *




get_ipython().system('unzip -q /kaggle/input/tgs-salt-identification-challenge/train.zip')




path_img = "/kaggle/working/images"
path_lbl = "/kaggle/working/masks"




fnames = get_image_files(path_img)
fnames[:3]




lbl_names = get_image_files(path_lbl)
lbl_names[:3]




get_y_fn = lambda x: path_lbl + '/'+ f'{x.stem}{x.suffix}'




# Function to get label masks is running fine
x = fnames[0]       
get_y_fn(x)




# Load an image
img_f = fnames[2]
img = open_image(img_f, div=True)
img.show(figsize=(5,5))
print(img.shape)




# Load corresponding masks

mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), cmap='gray')
print(mask.shape)




# Check the mask data
mask.data




# Resize all the masks by dividing by 255 and replacing the original masks
for i in fnames:
    mask = open_mask(get_y_fn(i), div=True)
    mask.save(get_y_fn(i))




print(len(fnames))




i = fnames[8]
img = open_image(i)
img.show()





mask = open_mask(get_y_fn(i))
mask.show()




mask.data




bs = 4




data = (SegmentationItemList.from_folder(path_img)
       .split_by_rand_pct(0.2)
       .label_from_func(get_y_fn, classes = ['0','255'])
       .transform(get_transforms(), tfm_y=True)
       .databunch(bs = bs)
       .normalize(imagenet_stats))




data.train_ds.x[1].data




data.train_ds.y[1].data




data.show_batch(2, cmap='gray')     # Shows 2 rows and 2 cols




data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)   # Display valid data




metrics = dice
wd = 1e-2




# learn.destroy()         # If you are reusing the same learner




learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)




learn.lr_find()                
learn.recorder.plot()




lr = 3e-4




learn.fit_one_cycle(5, slice(lr))




learn.save('stage-1')
learn.load('stage-1');




learn.show_results()  #rows=10, figsize=(8,9), cmap='Gray')




get_ipython().run_line_magic('pinfo2', 'learn.freeze_to')




learn.freeze_to(2)




lrs = slice(lr/100, lr/10)
lrs




learn.fit_one_cycle(3, lrs)




learn.show_results()




learn.summary()




learn.recorder.plot_losses()






