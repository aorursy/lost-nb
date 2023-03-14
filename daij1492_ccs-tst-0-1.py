#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import keras




def read_jpg(path):
    return np.asarray(PIL.Image.open(path),dtype=np.uint8)    




import os




tr_ty1 = [x for x in os.listdir('../input/train/Type_1') if '.jpg' in x]




for j,f in enumerate(tr_ty1):
    img = read_jpg(path='../input/train/Type_1/'+f)
    print(j,f,img.shape)




# Something wrong /w this particular image
read_jpg('../input/train/Type_1/'+tr_ty1[60])




read_jpg('../input/train/Type_1/'+tr_ty1[61])




tr_ty1[61]




import cv2
def get_image_data(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img




img = get_image_data('../input/train/Type_1/'+tr_ty1[61])




plt.imshow(img)




for j,f in enumerate(tr_ty1):
    img = get_image_data(fname='../input/train/Type_1/'+f)
    print(j,f,img.shape)




# The following is from previous version of testing notebook




len([x for x in os.listdir('../input/train/Type_3') if '.jpg' in x])




[x for x in os.listdir('../input/train/Type_1') if '.jpg' not in x] + [x for x in os.listdir('../input/train/Type_2') if '.jpg' not in x] + [x for x in os.listdir('../input/train/Type_3') if '.jpg' not in x]




len([x for x in os.listdir('../input/additional/Type_1') if '.jpg' in x]), len([x for x in os.listdir('../input/additional/Type_2') if '.jpg' in x]), len([x for x in os.listdir('../input/additional/Type_3') if '.jpg' in x])




[x for x in os.listdir('../input/additional/Type_1') if '.jpg' not in x] + [x for x in os.listdir('../input/additional/Type_2') if '.jpg' not in x] + [x for x in os.listdir('../input/additional/Type_3') if '.jpg' not in x]




test_files = os.listdir('../input/test/')
len(test_files)




[x for x in test_files if 'jpg' not in x]




#plt.imshow('../input/test/0.jpg') # Doesn't work directly









img = read_jpg('../input/test/0.jpg')




plt.imshow(img)




img.shape




plt.imshow(read_jpg('../input/test/1.jpg'))




plt.imshow(read_jpg('../input/test/2.jpg'))




plt.imshow(read_jpg('../input/test/3.jpg'))




subm = pd.read_csv('../input/sample_submission.csv')









subm.columns




subm.head()




0.168805	+ 0.527346	+ 0.303849




subm.shape




os.listdir('../config')









os.listdir('../lib')


























