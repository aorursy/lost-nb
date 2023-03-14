#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc #garbage collector
import ast # operate with string representation of list
import os

import pandas as pd
import numpy as np
import cv2
from scipy.stats import shapiro # for normal distribution checks 

import plotly.express as px
import plotly.graph_objects as go
import plotly
plotly.offline.init_notebook_mode(connected = True)

import matplotlib.pyplot as plt


# In[2]:


def convert_coords(bbox):
    '''
    Transform boundary box coordinates from pandas dataframe to cv2.rectangle values
    Pandas df values: x, y  width, height
    '''
    x, y, width, height = bbox
    start_point = x, y
    end_point = (x + width), (y + height)
    return start_point, end_point

def plot_samples(df, img_ids=None, threshold=6, title=''):
    '''
    Plot image grid from seleted dataframe 
    https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
    '''
    if img_ids is None:
        img_ids = df['image_id_ext'].unique()[:threshold]
    cols = 3
    rows = len(img_ids) // cols + 1
    fig = plt.figure(figsize = (15, 5 * rows))
    for i, img_id in enumerate(img_ids):
        bboxes_list = df[df['image_id_ext'] == img_id].bbox.to_list()
        img = cv2.imread(os.path.join(TRAIN_DIR_PATH, img_id))
        for item in bboxes_list:
            bbox = list(map(int, ast.literal_eval(item)))
            strart_point, end_point = convert_coords(bbox)
            color = (255, 0, 0) #RGB
            thickness = 2
            img = cv2.rectangle(img, strart_point, end_point, color, thickness)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)
    plt.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()
    
# Took ideas of these functions from https://www.kaggle.com/aleksandradeis/globalwheatdetection-eda

def get_image_brightness(image_id_ext):
    img = cv2.imread(os.path.join(TRAIN_DIR_PATH, image_id_ext))
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get average brightness
    return np.array(gray).mean()

def get_percentage_of_green_pixels(image_id_ext):
    img = cv2.imread(os.path.join(TRAIN_DIR_PATH, image_id_ext))
    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # get the green mask
    hsv_lower = (40, 40, 40) 
    hsv_higher = (70, 255, 255)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)
    
    return float(np.sum(green_mask)) / 255 / (1024 * 1024)

def get_percentage_of_yellow_pixels(image_id_ext):
    img = cv2.imread(os.path.join(TRAIN_DIR_PATH, image_id_ext))
    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # get the green mask
    hsv_lower = (25, 40, 40) 
    hsv_higher = (35, 255, 255)
    yellow_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)
    
    return float(np.sum(yellow_mask)) / 255 / (1024 * 1024)


# In[3]:


MAIN_PATH = '/kaggle/input/global-wheat-detection/'
TRAIN_DIR_PATH = '/kaggle/input/global-wheat-detection/train/'
TEST_DIR_PATH = '/kaggle/input/global-wheat-detection/test/'


# In[4]:


print('Train images amount', len(os.listdir(os.path.join(MAIN_PATH, 'train'))))
print('Test images amount', len(os.listdir(os.path.join(MAIN_PATH, 'test'))))


# In[5]:


train_df = pd.read_csv(os.path.join(MAIN_PATH, 'train.csv'))
sample_submission = pd.read_csv(os.path.join(MAIN_PATH, 'sample_submission.csv'))


# In[6]:


train_df.head().T


# In[7]:


sample_submission.head().T


# In[8]:


train_df['image_id_ext'] = train_df['image_id'] + '.jpg'


# In[9]:


ser = train_df['image_id']

fig = px.histogram(ser, title = 'Amunt of bundary boxes on the each picture', 
                   labels={'x':'image id', 'y':'bbox amount'})
fig.update_xaxes(categoryorder='total descending')
fig.show()


# In[10]:


temp_df = train_df.copy()
temp_df['dummy_column'] = 1
ser = temp_df.groupby(['image_id']).sum()['dummy_column']


# In[11]:


fig = px.histogram(ser, title = 'Sum distribution of bboxes amount', 
                   labels={'y':'bbox amount'})
fig.show()


# In[12]:


print('Shapiro-Wilk test for normality result\n statistic:{:.3f}, p-value:{:.3E}'.format(*shapiro(ser.to_list())))


# In[13]:


del temp_df
gc.collect()


# In[14]:


print('How many no-bbox images does exist in train dataset?', 
      len(os.listdir(TRAIN_DIR_PATH)) - len(train_df['image_id'].unique()))


# In[15]:


no_bbox_img_ids = set(os.listdir(TRAIN_DIR_PATH)) - set((train_df['image_id_ext']).unique().tolist())


# In[16]:


get_ipython().run_cell_magic('time', '', "# convert column '[x, y, width, height]' to the separate pandas dataframe\nsplitted_data = train_df['bbox'].str.split(r'[^\\d.]+')\nbbox_df = pd.DataFrame.from_dict(dict(zip(splitted_data.index,splitted_data))).T\nbbox_df.drop(columns = [0, 5], inplace = True) #drop empty columns\nbbox_df.columns = ['x', 'y', 'bwidth', 'bheight']\nbbox_df = bbox_df.astype(float)\nbbox_df['size'] = bbox_df['bwidth'] * bbox_df['bheight']")


# In[17]:


train_df = train_df.join(bbox_df)
train_df.head()


# In[18]:


train_df['size'].describe()


# In[19]:


max_bbox_size = train_df.groupby(['image_id']).max()['size'].to_frame()

fig = px.histogram(max_bbox_size, title = 'Max distribution of bboxes area size', 
                   labels={'y':'bbox area size'})
fig.show()


# In[20]:


MAX_ANOMALY_THRESHOLD = 120000
print('Size of anomaly frame: ', train_df[train_df['size'] > MAX_ANOMALY_THRESHOLD].shape)


# In[21]:


plot_samples(train_df[train_df['size'] > MAX_ANOMALY_THRESHOLD], 
             threshold=15, 
             title='Images with anomaly big bboxes')


# In[22]:


min_bbox_size = train_df.groupby(['image_id']).min()['size'].to_frame()

fig = px.histogram(min_bbox_size, title = 'Min distribution of bboxes area size', 
                   labels={'y':'bbox area size'})
fig.show()


# In[23]:


MIN_ANOMALY_THRESHOLD = 1000
print('Size of anomaly frame: ', train_df[train_df['size'] < MIN_ANOMALY_THRESHOLD].shape)


# In[24]:


plot_samples(train_df[train_df['size'] < MIN_ANOMALY_THRESHOLD], 
             threshold=9, 
             title = 'Images with anomaly small bboxes')


# In[25]:


sum_bbox_areas =  train_df.groupby(['image_id']).sum()['size']/(1024 * 1024)

fig = px.histogram(sum_bbox_areas, title = 'Percentage distribution of image box coverage', 
                   labels={'y':'bbox area size'})
fig.show()


# In[26]:


print('Shapiro-Wilk test for normality result \n statistic:{:.3f}, p-value:{:.3E}'.format(*shapiro(sum_bbox_areas.to_list())))


# In[27]:


fig = px.histogram(train_df.groupby(['image_id', 'source']).mean().index.to_frame()['source'],
                   title = 'Image sources distribution',
                   labels={'x':'source name', 'y':'images amount'})
fig.update_xaxes(categoryorder='total descending')
fig.show()


# In[28]:


max_bbox_ids = train_df['image_id_ext'].value_counts().index.to_list()[:6]
plot_samples(train_df, img_ids=max_bbox_ids, title='Samples with the maximum amount of boundary boxes')


# In[29]:


min_bbox_ids = train_df['image_id_ext'].value_counts(ascending=True).index.to_list()
plot_samples(train_df, img_ids=min_bbox_ids[:6], title='Samples with the minimum amount of boundary boxes')


# In[30]:


print('Sources of top 100 photost with the least amount of bboxes: ', 
      train_df[train_df['image_id_ext'].isin(min_bbox_ids[:100])]['source'].unique())    


# In[31]:


fig = px.histogram(train_df[train_df['source'] == 'arvalis_3'].groupby(['image_id']).mean()['size'], 
                   title = 'Percentage distribution of image box coverage in arvalis_3 dataset')                 
fig.show()


# In[32]:


sources_list = list(train_df['source'].unique())
print('Complete list of sources: \n', sources_list)


# In[33]:


plot_samples(train_df[train_df['source'] == 'arvalis_1'], 
             title='Source: ARVALIS (Institut du vegetal is an applied agricultural research)')


# In[34]:


plot_samples(train_df[train_df['source'] == 'arvalis_2'])


# In[35]:


plot_samples(train_df[train_df['source'] == 'arvalis_3'])


# In[36]:


plot_samples(train_df[train_df['source'] == 'inrae_1'],
             title='Source: INRAE (National Research Institute for Agriculture, Food and Environment)')


# In[37]:


plot_samples(train_df[train_df['source'] == 'ethz_1'],
             title='Source: ETHZ (Swiss Federal Institute of Technology in Zurich)')


# In[38]:


plot_samples(train_df[train_df['source'] == 'rres_1'],
             title='Soure: Rothamsted Research Institute')


# In[39]:


plot_samples(train_df[train_df['source'] == 'usask_1'],
             title='Source: University of Saskatchewan')


# In[40]:


plot_samples(train_df, img_ids=list(no_bbox_img_ids)[:6], 
             title='Examples of empty images without boundary boxes')


# In[41]:


get_ipython().run_cell_magic('time', '', "#Calculate mean brightness\nser = train_df.groupby(['image_id_ext']).mean().reset_index()['image_id_ext']\nmean_brightness = ser.apply(get_image_brightness)\n# Add results to train_df\nbright_df = pd.DataFrame({'image_id_ext': ser, 'mean brightness': mean_brightness})\ntrain_df = train_df.merge(bright_df, on='image_id_ext')")


# In[42]:


# Brightness (min - max)
fig = px.histogram(mean_brightness, title = 'Mean brightness distribution')
fig.show()


# In[43]:


first_group_id = train_df[(train_df['mean brightness'] >= 76) & (train_df['mean brightness'] <= 78)]['image_id_ext'].unique()
second_group_id = train_df[(train_df['mean brightness'] >= 106) & (train_df['mean brightness'] <= 108)]['image_id_ext'].unique()


# In[44]:


plot_samples(train_df, img_ids=first_group_id[:6], 
             title='Examples of images with brightness [76 - 78]')


# In[45]:


plot_samples(train_df, img_ids=second_group_id[:6], 
             title='Examples of images with brightness [106 - 108]')


# In[46]:


sorted_bright_images = train_df.groupby(['image_id_ext']).mean()['mean brightness']                       .sort_values(ascending=False).index.to_list()


# In[47]:


plot_samples(train_df, img_ids=sorted_bright_images[:6], 
             title='The most bright images')


# In[48]:


plot_samples(train_df, img_ids=sorted_bright_images[:-7:-1], 
             title='The most dark images')


# In[49]:


get_ipython().run_cell_magic('time', '', "#Calculate color percentage \nser = train_df.groupby(['image_id_ext']).mean().reset_index()['image_id_ext']\ngreen_percentage = ser.apply(get_percentage_of_green_pixels)\nyellow_percentage = ser.apply(get_percentage_of_yellow_pixels)\n# Add results to train_df\ncolors_df = pd.DataFrame({'image_id_ext': ser, 'green %': green_percentage, 'yellow %': yellow_percentage})\ntrain_df = train_df.merge(colors_df, on='image_id_ext')")


# In[50]:


gc.collect()


# In[51]:


yellow_means = train_df.groupby(['image_id']).mean()['yellow %']
green_means = train_df.groupby(['image_id']).mean()['green %']


# In[52]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=yellow_means, marker_color='#eeff00'))
fig.add_trace(go.Histogram(x=green_means, marker_color='#55ff00'))

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.7)
fig.update_layout(
    #barmode='overlay',
    title_text='Yellow and green percentage distribution', 
    xaxis_title_text='% of colored pixels', # xaxis label
    yaxis_title_text='Count', # yaxis label
    #bargap=0.2, # gap between bars of adjacent location coordinates
    #bargroupgap=0.1 # gap between bars of the same location coordinates
)

fig.show()


# In[53]:


sorted_yellow_images = train_df.groupby(['image_id_ext']).mean()['yellow %']                       .sort_values(ascending=False).index.to_list()
sorted_green_images = train_df.groupby(['image_id_ext']).mean()['green %']                       .sort_values(ascending=False).index.to_list()


# In[54]:


plot_samples(train_df, img_ids=sorted_yellow_images[:3], 
             title='The most yellow images')


# In[55]:


plot_samples(train_df, img_ids=sorted_yellow_images[:-4:-1], 
             title='The least yellow images')


# In[56]:


plot_samples(train_df, img_ids=sorted_green_images[:3], 
             title='The most green images')


# In[57]:


plot_samples(train_df, img_ids=sorted_green_images[:-4:-1], 
             title='The least green images')

