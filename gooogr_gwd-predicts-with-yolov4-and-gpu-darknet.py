#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import re
from collections import defaultdict


# In[2]:


#Constants
CONFIDENCE_THRESHOLD = 0.1 # Filter predicted bboxes

#Seeds
SEED = 42
np.random.seed(seed=SEED)


# In[3]:


## Use version from github
# ! git clone https://github.com/AlexeyAB/darknet.git 

# Use Darknet with CPU and make it from source
# ! cp -a /kaggle/input/global-wheat-detection-models/darknet/darknet/. /kaggle/darknet/

## Use pre-built Daknet binaries with GPU support
get_ipython().system(' cp -a /kaggle/input/global-wheat-detection-models/darknet_gpu_prebuilt/darknet_gpu_prebuilt/. /kaggle/darknet/')


# In[4]:


# %cd /kaggle/darknet

# ## Uncomment if you want to use Darknet with GPU.

# !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
# # !sed -i 's/GPU=0/GPU=1/' Makefile
# # !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
# # !sed -i 's/OPENMP=0/OPENMP=1/' Makefile

# !head Makefile

# %%capture 
# #Use  %%caputure to hide huge terminal output
# ! make clean
# ! make --silent


# In[5]:


get_ipython().system(' mkdir /kaggle/darknet/weights')
get_ipython().system(' cp -a /kaggle/input/global-wheat-detection-models/yolov4.weights /kaggle/darknet/weights')


# In[6]:


get_ipython().run_cell_magic('capture', '', '%cd /kaggle/darknet\n! chmod 777 ./darknet\n! ./darknet detect cfg/yolov4.cfg weights/yolov4.weights data/dog.jpg -dont_show')


# In[7]:


sample_preds = cv2.imread('predictions.jpg')
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(sample_preds)
fig.show()


# In[8]:


get_ipython().system(' ls /kaggle/input/global-wheat-detection-models/competition_files/competition_files')


# In[9]:


get_ipython().system(' mkdir /kaggle/darknet/my_files')
# cfg file and txt2json
get_ipython().system(' cp -a /kaggle/input/global-wheat-detection-models/competition_files/competition_files/. /kaggle/darknet/my_files')
# yolo weights (CHANGE LINK TO YOUR WEIGHTS HERE IF YOU NEED)
get_ipython().system(' cp -a /kaggle/input/global-wheat-detection-models/yolov4_naive.weights /kaggle/darknet/weights')


# In[10]:


get_ipython().system('mv /kaggle/darknet/my_files/yolov4-custom.cfg /kaggle/darknet/my_files/yolov4.cfg ')


# In[11]:


get_ipython().run_line_magic('cd', '/kaggle/darknet/my_files')


# In[12]:


get_ipython().run_cell_magic('writefile', 'obj.names', 'Wheat head')


# In[13]:


get_ipython().run_cell_magic('writefile', 'yolo.data', '#classses = 1\nnames = /kaggle/darknet/my_files/obj.names')


# In[14]:


def create_path_file(files_dir, save_dir):
    get_ipython().run_line_magic('cd', '/kaggle/working/')
    # from https://stackoverflow.com/questions/9816816/get-absolute-paths-of-all-files-in-a-directory
    file = open(os.path.join(save_dir, "predict.txt"), "w")
    for root, dirs, files in os.walk(os.path.abspath(files_dir)):
        for item in files:
            row = os.path.join(root, item)
            file.write(row)
            file.write('\n')
    file.close()


# In[15]:


create_path_file(files_dir='/kaggle/input/global-wheat-detection/test', 
                 save_dir='/kaggle/darknet/my_files/')


# In[16]:


get_ipython().system(' head /kaggle/darknet/my_files/predict.txt ')


# In[17]:


get_ipython().system(' ls /kaggle/darknet')


# In[18]:


get_ipython().run_cell_magic('capture', '', '%cd /kaggle/darknet\n\n! ./darknet detector test \\\nmy_files/yolo.data \\\nmy_files/yolov4.cfg \\\nweights/yolov4_naive.weights \\\n/kaggle/input/global-wheat-detection/test/2fd875eaa.jpg -dont_show')


# In[19]:


sample_preds = cv2.imread('predictions.jpg')
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(sample_preds)
fig.show()


# In[20]:


get_ipython().run_cell_magic('capture', '', '%cd /kaggle/darknet\n\n! ./darknet detector test \\\nmy_files/yolo.data \\\nmy_files/yolov4.cfg \\\nweights/yolov4_naive.weights \\\n-dont_show -ext_output < my_files/predict.txt > log.txt')


# In[21]:


# # Uncomment to see result log file
# ! cat log.txt


# In[22]:


def txt2json(file_path):
    file_lines = open(file_path, 'r').read()
    table_dict = defaultdict()
    current_jpg_name = ''

    jpg_delimiters = " ", "/", ":"
    jpg_regexPattern = '|'.join(map(re.escape, jpg_delimiters))

    for line in file_lines.splitlines():
        if '.jpg' in line:
            for item in re.split(jpg_regexPattern, line):
                if '.jpg' in item:
                    current_jpg_name = item
                    table_dict[item] = []
        if '%' in line:
            split_string = (re.findall('-?\d+', line))
            split_string = list(filter(lambda x: x != "", split_string)) # remove empty strings from list
            int_string = list(map(int, split_string))
            sub_dict_keys = ['proba_%', 'left_x', 'top_y', 'width', 'height']
            table_dict[current_jpg_name].append(dict(zip(sub_dict_keys, int_string)))
    return table_dict


# In[23]:


data = txt2json('/kaggle/darknet/log.txt')


# In[24]:


# data['empty_sample'] = list()  #ONLY FOR NEGATIVE TEST, DON'T UNCOMMENT


# In[25]:


img_id, proba, left_x, top_y, width, height = list([]), [], [], [], [], []
for key in data.keys():
    try:
        df = pd.DataFrame(data[key])
        img_id.extend([key] * len(df))
        proba.extend(df['proba_%'].values)
        left_x.extend(df['left_x'].values)
        top_y.extend(df['top_y'].values)
        width.extend(df['width'].values)
        height.extend(df['height'].values)
    except: # in case of no detections
        img_id.extend([key])
        proba.extend([np.nan])
        left_x.extend([np.nan])
        top_y.extend([np.nan])
        width.extend([np.nan])
        height.extend([np.nan])

result_df = pd.DataFrame(list(zip(img_id, proba, left_x, top_y, width, height)), 
                         columns = ['img', 'proba_%', 'left_x', 'top_y', 'width', 'height'])
result_df.head()


# In[26]:


sample_submission = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')
sample_submission.head().T


# In[27]:


result_df['proba_ratio'] = result_df['proba_%'] / 100


# In[28]:


def format_list(confidence, x, y, width, height):
    temp_list =  [confidence, x, y, width, height]
    if not np.isnan(confidence):
        return ' '.join(str(item) for item in temp_list)
    else:
        return np.nan


# In[29]:


result_df['sub_list'] = result_df.apply(lambda x: format_list(x.proba_ratio, 
                                                              x.left_x, 
                                                              x.top_y, 
                                                              x.width, 
                                                              x.height), axis = 1)


# In[30]:


filter_condition = (result_df['proba_ratio'] > CONFIDENCE_THRESHOLD) | (result_df['proba_ratio'].isna())
result_df = result_df[filter_condition]

result_df.fillna('', inplace=True)

img_pred_list = []
for img_name in result_df['img'].unique():
    img_pred_list.append(' '.join(str(item) for item in result_df[result_df['img']==img_name].sub_list))

img_names = [item.split('.')[0] for item in result_df['img'].unique()]

submission = pd.DataFrame(zip(img_names, img_pred_list), 
                          columns = ['image_id', 'PredictionString'])


# In[31]:


submission.head()


# In[32]:


submission.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:




