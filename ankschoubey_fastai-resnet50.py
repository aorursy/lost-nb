#!/usr/bin/env python
# coding: utf-8



from fastai import *
from fastai.vision import *

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

base_dir = Path("../input")
save_dir = Path('/kaggle/working')
model_dir= Path('/tmp/models')
base = base_dir
base.ls()




train_folder, train_path, sample_sub_path, test_path= base.ls()




train_folder.ls()




train_folder = train_folder.ls()[0]; train_folder




train_df = pd.read_csv(train_path); train_df.head()




np.random.seed(42)
data_il = ImageList.from_df(train_df,train_folder)
data_il = data_il.split_by_rand_pct()
data_il = data_il.label_from_df()
data_il = data_il.transform(get_transforms(),size=32)




data = data_il.databunch().normalize(imagenet_stats)




data.show_batch()




learn = cnn_learner(data, models.resnet50, model_dir='/temp/model',metrics=[accuracy]).to_fp16()




learn.freeze()
learn.lr_find()




learn.recorder.plot()




learn.fit_one_cycle(3, 1e-2)




learn.recorder.plot_losses()




learn.save(save_dir/'stage1-64-resnet50')




learn.load(save_dir/'stage1-64-resnet50')




learn.unfreeze()




learn.lr_find()




learn.recorder.plot()




learn.fit_one_cycle(2, slice(1e-8,1e-7/5))




learn.recorder.plot_losses()




learn.recorder.plot_metrics()




learn.recorder.plot_lr()




learn.save(save_dir/'stage2-32-resnet34')




learn.load(save_dir/'stage2-32-resnet34')




test_df = pd.read_csv("../input/sample_submission.csv")




test_path = test_path.ls()[0]; test_path.ls()




test_img = ImageList.from_df(test_df, path=test_path, folder='')




test_img[0]




learn.data.add_test(test_img)




preds,y = learn.get_preds(ds_type=DatasetType.Test)




test_df.has_cactus = preds.numpy()[:, 0]




test_df.to_csv('submission.csv', index=False)






