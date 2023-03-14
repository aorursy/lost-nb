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
#for dirname, _, filenames in os.walk('/kaggle/input'):
     #   print(os.path.join(dirname, filename))
files = os.listdir("../input")
print(files)
print('trainlabels.csv' in files)
print(len(files))
# Any results you write to the current directory are saved as output.




from fastai import *
from fastai.vision import *
import matplotlib as plt
import pandas as pd
from fastai.widgets import ClassConfusion
from fastai.widgets import *




print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)




train_df =  pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
valid_df =  pd.read_csv("../input/aptos2019-blindness-detection/test.csv")




train_df.head(10)




valid_df.head(10)




train_df['diagnosis'].hist(figsize = (10, 5))




#tfms = get_tranforms(do_flip=True,)
tfms=get_transforms(do_flip = True,flip_vert = True,max_rotate=360,max_zoom = 1.1)




data = (ImageList.from_df(train_df,"../input/aptos2019-blindness-detection/train_images",suffix='.png').
       split_by_rand_pct(0.1).
       label_from_df(1).
       transform(tfms,size=256).
       databunch(bs = 16).
       normalize(imagenet_stats))




data.show_batch()




learn = cnn_learner(data, models.resnet101, metrics=accuracy,model_dir="/kaggle/working")




learn.lr_find()




learn.recorder.plot()




learn.fit_one_cycle(5,slice(2e-5,2e-3),wd=0.1,moms=(0.8,0.9))




learn.fit_one_cycle(2,max_lr=slice(2.5e-3),wd=0.1,moms=(0.8,0.9))




learn.fit_one_cycle(2,slice(2.5e-5),wd=0.01,moms=(0.8,0.9))




learn.fit_one_cycle(6,max_lr=slice(1e-3,1e-4),wd=0.1,moms=0.9)




learn.save('stage-1')




learn.load('stage-1')




learn.freeze()




learn.recorder.plot_losses()




learn.show_results()




learn.get_preds()




interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)




interp.plot_top_losses(9, figsize=(15,11))




interp.plot_confusion_matrix()




interp.most_confused()




Tf = partial(Image.apply_tfms,tfms=get_transforms(do_flip=True, flip_vert = True)[0][1:]+get_transforms(do_flip=True, flip_vert = True)[1],size = 512)  




sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")




print(sub)




img = open_image("../input/aptos2019-blindness-detection/test_images/020f6983114d.png")
pre = learn.predict(img)
x = pre[1]
x = int(x)
print(x)





for i in range(len(sub.id_code)):
    s=0
    id = sub.id_code[i]
    img=open_image("../input/aptos2019-blindness-detection/test_images/"+sub.id_code[i]+".png")
    """
    for i in range(10):
            Img = Tf(img)
            p = learn.predict(Img)
            p = p[1]
            p = int(p)
            #print(p) 
            s=s+p
    """
            
    Img = Tf(img)
    s = learn.predict(Img)
    s = s[1]
    s = int(s)
    print(s)
    sub.diagnosis[i]=s
    print(sub.diagnosis[i])





sub.to_csv("submission.csv",index=False)




print(sub) 






