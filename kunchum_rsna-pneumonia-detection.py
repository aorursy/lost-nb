#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os
from matplotlib.patches import Rectangle

datapath = '../input/rsna-pneumonia-detection-challenge/'


# In[2]:


get_ipython().system('ls ../input/rsna-pneumonia-detection-challenge/stage_2_train_images/ | wc -l')


# In[3]:


def parse_data(df):
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': datapath + 'stage_2_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    plt.imshow(im, cmap=plt.cm.gist_gray)
    
    
def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im


# In[4]:


df_box = pd.read_csv(datapath+'stage_2_train_labels.csv')
print('Number of rows (unique boxes per patient) in main train dataset:', df_box.shape[0])
print('Number of unique patient IDs:', df_box['patientId'].nunique())
df_box.head(10)


# In[5]:


df_box.groupby('Target').size().plot.bar()
print(df_box.groupby('Target').size() / df_box.shape[0])


# In[6]:


df_det = pd.read_csv(datapath+'stage_2_detailed_class_info.csv')
print('Number of rows in auxiliary dataset:', df_det.shape[0])
print('Number of unique patient IDs:', df_det['patientId'].nunique())
df_det.head(10)


# In[7]:


df_det.groupby('class').size().plot.bar()
print(df_det.groupby('class').size() / df_det.shape[0])
assert df_det.loc[df_box['Target']==0].shape[0] == df_det.loc[df_det['class'].isin(['Normal',     'No Lung Opacity / Not Normal'])].shape[0], 'Number of negative targets does not match between main and detailed dataset.'
assert df_box.loc[df_box['Target']==1].shape[0] == df_det.loc[df_det['class'] == 'Lung Opacity'].shape[0],     'Number of positive targets does not match between main and detailed dataset.'


# In[8]:


assert df_box['patientId'].values.tolist() == df_det['patientId'].values.tolist(), 'PatientId columns are different.'
df_train = pd.concat([df_box, df_det.drop(labels=['patientId'], axis=1)], axis=1)
df_train.head(10)


# In[9]:


pId = "003d8fa0-6bf1-40ed-b54c-ac657f8495c5"    
dcmdata = pydicom.read_file(datapath+'stage_2_train_images/'+pId+'.dcm')
print(dcmdata)


# In[10]:


dcmimg = dcmdata.pixel_array
plt.figure(figsize=(7,7))
plt.imshow(dcmimg, cmap=pylab.cm.binary)


# In[11]:


parsed = parse_data(df_box)

patientId = df_box['patientId'][8]
#print(df_det.loc[patientId])

plt.figure(figsize=(7,7))
plt.title("Sample Patient - Lung Opacity")

draw(parsed[patientId])


# In[12]:


plt.figure(figsize=(20,10))

plt.subplot(131)
plt.title("Normal Image")
draw(parsed[df_box['patientId'][3]])

plt.subplot(132)
plt.title("Lung Opacity")
draw(parsed[df_box['patientId'][16]])

plt.subplot(133)
plt.title("No Lung Opacity / Not Normal")
draw(parsed[df_box['patientId'][1]])


# In[ ]:




