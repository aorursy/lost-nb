#!/usr/bin/env python
# coding: utf-8



#-*- encoding: utf8 -*-
get_ipython().run_line_magic('matplotlib', 'inline')




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import json
import zipfile
from tqdm import tqdm
from matplotlib import pyplot as plt
import ast
# Any results you write to the current directory are saved as output.




get_ipython().system('mkdir ../working/main_data')




./




SEED = 42
INPUT_DIR = "../input/"
train_simple_eiffel = pd.read_csv(os.path.join(INPUT_DIR, 'train_simplified/The Eiffel Tower.csv'))
train_simple_eiffel.head()




train_simple_eiffel['recognized'].value_counts()




train_simple_eiffel['drawing'] = train_simple_eiffel['drawing'].apply(ast.literal_eval)




n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
for i, drawing in enumerate(train_simple_eiffel[train_simple_eiffel['recognized']==True][:100].drawing):
    ax = axs[i // n, i % n]
    for x, y in drawing:
        ax.plot(x, -np.array(y), lw=3)
    ax.axis('off')
fig.savefig('eiffe_true.png', dpi=200)
plt.show();




n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
for i, drawing in enumerate(train_simple_eiffel[train_simple_eiffel['recognized']==False][:100].drawing):
    ax = axs[i // n, i % n]
    for x, y in drawing:
        ax.plot(x, -np.array(y), lw=3)
    ax.axis('off')
fig.savefig('eiffe_false.png', dpi=200)
plt.show();




CHUNK_DIR = '../working/main_data/'




np.random.seed(seed = SEED)




len(os.listdir(os.path.join(INPUT_DIR, 'train_simplified/')))




print(len(os.listdir(os.path.join(INPUT_DIR, 'train_simplified/'))))




def file2cat(filename):
    return filename.split('.')[0]




class Simplified():
    def __init__(self, input_path = '../input'):
        self.input_path = input_path
    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return sorted([file2cat(f) for f in files], key = str.lower)
    def read_training_csv(self, category, nrows = None, usecols = None, drawing_transform = False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),                        parse_dates = ['timestamp'],usecols=usecols)
        df = df[df['recognized']==True].reset_index(drop=True)
        df = df.iloc[:nrows]
        
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(json.loads)
        return df




s = Simplified(INPUT_DIR)
DATA_CHUNK = 100
categories = s.list_all_categories()
print(len(categories))




for y, cat in tqdm(enumerate(categories)):
    df = s.read_training_csv(cat, nrows=30000)
    df['y'] = y
    df['cv'] = (df.key_id//10 ** 7) % DATA_CHUNK
    for k in range(DATA_CHUNK):
        filename = 'train_k{}.csv'.format(k)
        filename = os.path.join(CHUNK_DIR, filename)
        chunk = df[df.cv==k]
        chunk = chunk.drop(['key_id'],axis = 1)
        if y==0:
            chunk.to_csv(filename, index =False)
        else:
            chunk.to_csv(filename, mode = 'a', header = False, index = False)




import gc
from multiprocessing import Pool
from functools import partial
from itertools import repeat
from itertools import product




get_ipython().system('grep -c processor /proc/cpuinfo')




NJOBS = get_ipython().getoutput('grep -c processor /proc/cpuinfo')




def chunk2gzip(chunk_number):
#     for k in range(chunk_number,JOBS):
#         try:
    filename = 'train_k{}.csv'.format(chunk_number)
    filename = os.path.join(CHUNK_DIR, filename)
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        print(df.shape)
        df['rnd'] = np.random.rand(len(df))
        df = df.sort_values(by ='rnd').drop('rnd', axis =1)
        df.to_csv(filename + '.gz', compression='gzip', index = False)
        os.remove(filename)
#         except IndexError:
#             print("file no. {} not in".format(k))
#             pass
    




with Pool(processes=int(NJOBS[0])) as p:
    max_ = DATA_CHUNK
    with tqdm(total=max_) as pbar:
        for i, _ in tqdm(enumerate(p.imap(chunk2gzip, list(range(0,100))))):
            pbar.update()

