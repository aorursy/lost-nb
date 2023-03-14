#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('cd', '../input/python3gdcm')




get_ipython().system('dpkg -i build_1-1_amd64.deb')
get_ipython().system('apt-get install -f')




get_ipython().system('cp /usr/local/lib/gdcm.py /opt/conda/lib/python3.7/site-packages/.')
get_ipython().system('cp /usr/local/lib/gdcmswig.py /opt/conda/lib/python3.7/site-packages/.')
get_ipython().system('cp /usr/local/lib/_gdcmswig.so /opt/conda/lib/python3.7/site-packages/.')
get_ipython().system('cp /usr/local/lib/libgdcm* /opt/conda/lib/python3.7/site-packages/.')
get_ipython().system('ldconfig')




get_ipython().run_line_magic('cd', '-')




ls ../input/




get_ipython().system('pip install ../input/dicom-parser/dicom_parser-0.1.3-py3-none-any.whl > /dev/null')




get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys
from pathlib import Path
from tqdm import tqdm
import traceback
import gc
import joblib as jl
gc.enable()

import numpy as np
import pandas as pd

from dicom_parser import Image
import PIL
import cv2

import matplotlib.pyplot as plt


TRAIN_ROOT_PATH = Path('/kaggle/input/osic-pulmonary-fibrosis-progression/train/')




path_dict = {p.stem: list(TRAIN_ROOT_PATH.glob('%s/*.dcm'%(p.stem,))) for p in TRAIN_ROOT_PATH.glob('*')}




"""
'ID00149637202232704462834'

'ID00122637202216437668965']

'ID00122637202216437668965'
'ID00117637202212360228007',

"""

def get_smpl(idx):
    image = Image(path_dict['ID00122637202216437668965'][idx]).data
    
    h, w = image.shape
    image = image[image[:,w//2]!=0][:,image[h//2,:]!=0]
    h, w = image.shape
    image = image[w//3:w*7//10, h//2:]
    
    return image




image = get_smpl(8)
h, w = image.shape
print(cv2.blur(image[w//3:-w//3, h//3:-h//3], (5,5)).std())

plt.imshow(np.flip(image, axis=1), cmap='gray')
plt.show()




image = get_smpl(4)
h, w = image.shape
print(cv2.blur(image[w//3:-w//3, h//3:-h//3], (5,5)).std())

plt.imshow(np.flip(image, axis=1), cmap='gray')
plt.title("Original Slice")
plt.show()




len(path_dict['ID00122637202216437668965'])




image = get_smpl(16)

h, w = image.shape

print(cv2.blur(image[w//3:-w//3, h//3:-h//3], (5,5)).std())

plt.imshow(np.flip(image, axis=1), cmap='gray')
plt.show()




no_lung = []
for patient_id, pathes_dcm in path_dict.items():
    
    imgs = []
    
    for path_dcm in  pathes_dcm:
        image = Image(path_dcm)
        img_np = image.data
        h, w = img_np.shape
        
        clipped = img_np[img_np[:,w//2]!=0][:,img_np[h//2,:]!=0]
        h, w = clipped.shape
        lung_R = clipped[w//3:w*7//10, h//2:]
        lung_L = np.flip(clipped[w//3:w*7//10, :h//2], axis=1)
        h, w = lung_R.shape
        
        try:
        
            if 350 < np.std(cv2.blur(lung_R, ksize=(5, 5))[w//3:-w//3, h//3:-h//3]) < 450:
                imgs.append(lung_R)
            
            if 350 < np.std(cv2.blur(lung_L, ksize=(5, 5))[w//3:-w//3, h//3:-h//3]) < 450:
                imgs.append(lung_L)
                
        except:
            
            print(path_dcm)            
            traceback.print_exc()
            
        del image
        
    if len(imgs) == 0:
        no_lung.append(patient_id)
        
    jl.dump(imgs, patient_id+'.xz', 9)
        

    gc.collect()
            




no_lung




def get_smpl(idx):
    image = Image(path_dict['ID00371637202296828615743'][idx]).data
    
    h, w = image.shape
    image = image[image[:,w//2]!=0][:,image[h//2,:]!=0]
    h, w = image.shape
    image = image[w//3:w*7//10, h//2:]
    
    return image




image = get_smpl(4)

h, w = image.shape

print(cv2.blur(image, (5,5)).std())

plt.imshow(np.flip(image, axis=1), cmap='gray')
plt.show()




image = get_smpl(30)

h, w = image.shape

print(cv2.blur(image, (5,5)).std())

plt.imshow(np.flip(image, axis=1), cmap='gray')
plt.show()




plt.imshow(jl.load(list(Path('.').glob('*.xz'))[np.random.randint(100)])[0], cmap='gray')
plt.show()

