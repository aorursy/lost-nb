#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


raw_csv = pd.read_csv("../input/3d-object-detection-for-autonomous-vehicles/train.csv")


# In[3]:


raw_csv.head()


# In[4]:


raw_csv.sample(5)


# In[5]:


raw_csv[['PredictionString']]


# In[6]:


raw_csv.shape


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[8]:


image=plt.imread("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/host-a101_cam3_1242749262599200006.jpeg", format=None)


# In[9]:


test_image=plt.imread("/kaggle/input/3d-object-detection-for-autonomous-vehicles/test_images/host-a011_cam3_1232841055800995006.jpeg", format=None)


# In[10]:


plt.imshow(image,cmap='gray')


# In[11]:


plt.imshow(test_image,cmap='gray')


# In[12]:


image.shape


# In[13]:


test_image.shape


# In[14]:


pip install -U lyft_dataset_sdk


# In[15]:


pip install -U git+https://github.com/lyft/nuscenes-devkit


# In[16]:


get_ipython().system('pip install -U git+https://github.com/lyft/nuscenes-devkit moviepy >> /dev/tmp')


# In[17]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from IPython.display import HTML


# In[18]:


import pdb
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

from moviepy.editor import ImageSequenceClip
from tqdm import tqdm_notebook as tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


# gotta do this for LyftDataset SDK, it expects folders to be named as `images`, `maps`, `lidar`

get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')


# In[20]:


lyftdata = LyftDataset(data_path='.', json_path='data/', verbose=True)


# In[21]:


lyftdata.category[0]


# In[22]:


car_token = lyftdata.category[0]['token']
car_token


# In[23]:


lyftdata.get('category',car_token)


# In[24]:


lyftdata.sample_annotation[0]


# In[25]:


#sample_annotation
#sample_annotation refers to any bounding box defining the position of an object seen in a sample. 
#All location data is given with respect to the global coordinate system. Let's examine an example from our sample above.
my_annotation_token = my_sample['anns'][16]
my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)
my_annotation


# In[26]:


#We can also render an annotation to have a closer look.
lyftdata.render_annotation(my_annotation_token)


# In[27]:


#The attribute record indicates about what was the state of the concerned object when it was annotated
lyftdata.get('attribute', lyftdata.sample_annotation[0]['attribute_tokens'][0])


# In[28]:


lyftdata.scene[0]


# In[29]:


my_scene = lyftdata.scene[1]
my_sample_token= my_scene['first_sample_token']
lyftdata.render_sample(my_sample_token)


# In[30]:


train = pd.read_csv('../input/3d-object-detection-for-autonomous-vehicles/train.csv')
train.head()


# In[31]:


#We'll be using token0 to as our reference sample token
token0 = train.iloc[0]['Id']
token0


# In[32]:


my_sample = lyftdata.get('sample', my_sample_token)
my_sample


# In[33]:


#A useful method is list_sample() which lists all related sample_data keyframes and sample_annotation associated with a sample which we will discuss in detail in the subsequent parts.

lyftdata.list_sample(my_sample['token'])


# In[34]:


lyftdata.render_sample_3d_interactive(my_sample['token'], render_sample=False)


# In[35]:


#Instead of looking at camera and lidar data separately, we can also project the lidar pointcloud into camera images
lyftdata.render_pointcloud_in_image(sample_token = my_sample["token"],
                                      dot_size = 1,
                                      camera_channel = 'CAM_FRONT')


# In[36]:


#The dataset contains data that is collected from a full sensor suite. Hence, for each snapshot of a scene, we provide references to a family of data that is collected from these sensors.

#We provide a data key to access these:
my_sample['data']


# In[37]:


#Notice that the keys are referring to the different sensors that form our sensor suite. Let's take a look at the metadata of a sample_data taken from CAM_FRONT.
sensor_channel = 'CAM_FRONT' 
my_sample_data = lyftdata.get('sample_data', my_sample['data'][sensor_channel])
my_sample_data


# In[38]:


# also try this e.g. with 'LIDAR_TOP'
sensor_channel = 'LIDAR_TOP'  
my_sample_data_lidar = lyftdata.get('sample_data', my_sample['data'][sensor_channel])
my_sample_data_lidar


# In[ ]:




