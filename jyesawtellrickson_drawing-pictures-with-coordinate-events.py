#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import NonUniformImage

from math import floor
from random import sample

import json

import missingno as msno

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)


# In[2]:


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(
        train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(
        test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(
        train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(
        specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(
        sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


# read data
train, test, train_labels, specs, sample_submission = read_data()


# In[3]:


titles = [["Welcome to Lost Lagoon!"], ["Tree Top City - Level 1"], ["Ordering Spheres"], ["All Star Sorting"], ["Costume Box"], ["Fireworks (Activity)"], ["12 Monkeys"], ["Tree Top City - Level 2"], ["Flower Waterer (Activity)"], ["Pirate's Tale"], ["Mushroom Sorter (Assessment)"], ["Air Show"], ["Treasure Map"], ["Tree Top City - Level 3"], ["Crystals Rule"], ["Rulers"], ["Bug Measurer (Activity)"], ["Bird Measurer (Assessment)"], ["Magma Peak - Level 1"], ["Sandcastle Builder (Activity)"], ["Slop Problem"], ["Scrub-A-Dub"], ["Watering Hole (Activity)"], ["Magma Peak - Level 2"], ["Dino Drink"], ["Bubble Bath"], ["Bottle Filler (Activity)"], ["Dino Dive"], ["Cauldron Filler (Assessment)"], ["Crystal Caves - Level 1"], ["Chow Time"], ["Balancing Act"], ["Chicken Balancer (Activity)"], ["Lifting Heavy Things"], ["Crystal Caves - Level 2"], ["Honey Cake"], ["Happy Camel"], ["Cart Balancer (Assessment)"], ["Leaf Leader"], ["Crystal Caves - Level 3"], ["Heavy, Heavier, Heaviest"], ["Pan Balance"], ["Egg Dropper (Activity)"], ["Chest Sorter (Assessment)"]]
ordered_titles = pd.DataFrame.from_records(titles, columns=['title'])
ordered_titles = ordered_titles.reset_index().rename(columns={'index': 'order'}).set_index('title')
ordered_titles.head()


# In[4]:


train = train[(train["event_data"].notnull())]
train = train[(train["type"] != "Clip")]
train = train[train["event_data"].apply(lambda x: 'coordinates' in x)]

train = train.sample(1000000)

event_data = pd.DataFrame.from_records(
    train.event_data.apply(json.loads).tolist(),
    index=train.index
)
# Sort the most non-null columns at the start
event_data = pd.merge(
    event_data[event_data.isnull().sum().sort_values().index],
    train[['title', 'type', 'world']],
    left_index=True,
    right_index=True)

del train


# In[5]:


print("Total of {} rows and {} features.".format(*event_data.shape))


# In[6]:


event_data.head()


# In[7]:


msno.matrix(event_data.iloc[:, :50].sample(250))
fig = plt.gcf()


# In[8]:


event_data = event_data[['title', 'world', 'type', 'coordinates']]
event_data.head()


# In[9]:


event_data = pd.merge(pd.DataFrame.from_records(
    event_data.coordinates.values.tolist(), index=event_data.index),
                      event_data.drop('coordinates', axis=1),
                      left_index=True,
                      right_index=True)
event_data.head()


# In[10]:


event_data['scale'] = 100 / event_data.stage_width

event_data[['x', 'y', 'stage_width', 'stage_height']] =     event_data[['x', 'y', 'stage_width', 'stage_height']]     .multiply(event_data['scale'], axis=0)


# In[11]:


event_data.head()


# In[12]:


print(f'{event_data.title.nunique()} unique maps to plot')


# In[13]:


event_data = pd.merge(
    event_data,
    ordered_titles,
    left_on='title',
    right_index=True)


# In[14]:


event_data.query('title == "Welcome to Lost Lagoon!"')


# In[15]:


def plot_heatmaps(event_data):
    xedges = np.linspace(0, 100, 51)
    yedges = np.linspace(0, 100, 51)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    cols = 4
    rows = int(24/cols)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 22))

    color_maps = {'MAGMAPEAK': 'Reds',
                  'CRYSTALCAVES': 'Blues',
                  'TREETOPCITY': 'Greens'}

    for i, _ in enumerate(event_data[['title', 'world', 'type']].drop_duplicates().iterrows()):
        game, world, game_type = _[1].values
        game_data = event_data.query('title == @game')
        # Linear x array for cell centers:
        H, xedges, yedges = np.histogram2d(game_data.x,
                                           game_data.y,
                                           bins=(xedges, yedges))
        H = H.T

        ax = axs[floor(i/cols), i%cols]
        # ax = fig.add_subplot(int('44'+str(i+1)))

        interp = 'nearest' # bilinear, nearest

        im = NonUniformImage(ax, interpolation=interp, extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()),
                            cmap=plt.get_cmap(color_maps[world]))
        im.set_data(xcenters, ycenters, H)
        ax.images.append(im)
        ax.set_xlim(xedges.min(), xedges.max())
        ax.set_ylim(yedges.min(), yedges.max())
        ax.set_title('{} ({})'.format(game, game_type) if game_type=='Game' else game)
        ax.set_aspect(aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()


# In[16]:


event_data = event_data.sort_values(by=['order'])

plot_heatmaps(event_data)


# In[17]:


event_data = event_data.sort_values(by=['type', 'order'])

plot_heatmaps(event_data)

