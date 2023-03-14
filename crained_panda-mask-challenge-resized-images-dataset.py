#!/usr/bin/env python
# coding: utf-8



import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm




data_path = Path('../input/prostate-cancer-grade-assessment/')
os.listdir(data_path)




mask_path = Path('../input/panda-train-mask/')
os.listdir(mask_path)




get_ipython().system('cd ../input/prostate-cancer-grade-assessment/; du -h')




os.listdir(data_path/'train_label_masks')




import pandas as pd
train_df = pd.read_csv(mask_path/'train_mask.csv')
train_df.head(10)




print('Number of whole-slide images in training set: ', len(train_df))




sample_image = train_df.iloc[np.random.choice(len(train_df))].image_id
print(sample_image)




import openslide




openslide_image = openslide.OpenSlide(str(data_path/'train_label_masks'/(sample_image+'.tiff')))




openslide_image.properties




img = openslide_image.read_region(location=(0,0),level=2,size=(openslide_image.level_dimensions[2][0],openslide_image.level_dimensions[2][1]))
img




Image.fromarray(np.array(img.resize((512,512)))[:,:,:3])




get_ipython().run_line_magic('pinfo', 'Image.save')




for i in tqdm(train_df['image_id'],total=len(train_df)):
    openslide_image
    img
    Image.fromarray(np.array(img.resize((256,256)))[:,:,:3]).save(i+'.jpeg')
    

