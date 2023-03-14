#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for plotting
import matplotlib.gridspec as gridspec # to do the grid of plots
# jupyter cell magic for inline visualization
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns # for plotting
sns.set(style='whitegrid') # for plotting style

from IPython.display import display

import gc; gc.enable()

#setting to suppress SettingWithCopy
pd.set_option('mode.chained_assignment', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Any results you write to the current directory are saved as output.

#set random seed
SEED = 42
np.random.seed(SEED)




##############################
#######custom  functions######
##############################
def plot_target_dist(df, cols, figsize = (16,10), grid_r=3, grid_c=3):
    
    grid = gridspec.GridSpec(grid_r,grid_c) # The grid of chart
    fig = plt.figure(figsize=figsize) # size of figure
    total = df.shape[0] # total number of observations

    # loop to get column and the count of plots
    for n, col in enumerate(df[cols]): 
        ax = plt.subplot(grid[n]) # feeding the figure of grid
        
        #for low cardinality data
        if df[col].nunique() < 14:
            #count plot
            sns.countplot(x=col, data=df, hue='target', palette='Paired',
                          order=df[col].sort_values().unique(),ax=ax) 
            #df.groupby([col,'target'])[col].count().unstack(level=1)\
            #  .plot(kind='bar', color = ["#a6cee3", "#1f78b4"], width=0.8, ax=ax)
            sizes=[] # Get highest values in y
            for p in ax.patches: # loop to all objects
                height = p.get_height()
                sizes.append(height)
                ax.text(p.get_x()+p.get_width()/2.,
                        height * 1.02,
                        '{:1.1f}%'.format(height/total*100),
                        ha="center", fontsize=14) 
            ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

        #for high cardinality data
        else:
            df_col = df.groupby([col,'target'])[col].count()                       .unstack(level=1).fillna(0).sort_index()
            #define the plot type 
            if df[col].nunique() < 200:
                df_col.plot(kind='bar', ax=ax, stacked=True,
                            color = ["#a6cee3", "#1f78b4"], width=1)
            else: 
                df_col.plot(kind='line', ax=ax, 
                            color = ["#a6cee3", "#1f78b4"])
                        
            #force number of xticks to show
            ax.xaxis.set_major_locator(plt.MaxNLocator(20) )
            
            
        #set labels
        ax.set_ylabel('Count', fontsize=15) # y axis label
        ax.set_title(f'{col} distribution by target', fontsize=16) # title label
        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label
        _xlim = ax.get_xlim()
        
        #calculate pct of class 1
        d = df.groupby([col,'target'])[col].count()              .unstack(level=1).fillna(0).sort_index()
        d['class1_pct'] = d[1] / (d.sum(axis=1))
        if (d.index.dtype == 'int'):
            d.index = d.index - d.index.min()
        elif (d.index.dtype == 'float'):
            #to fix the scale
            d.index = d.index * df[col].nunique()
        
        #add another y-axis to show the pct of class 1
        ax2 = ax.twinx()
        if df[col].nunique() < 200:
            d.class1_pct.plot(marker='o',markersize=5,ax=ax2,color=["#6a3d9a"])
        else:
            d.class1_pct.plot(marker='o',markersize=5,linewidth=0,
                              ax=ax2,color=["#6a3d9a"])
            
        ax2.set_ylabel('class 1 fraction', color="#6a3d9a", fontsize=15)
        ax2.set_xlim(_xlim)
        ax2.set_ylim([-0.1,1.1])
        ax2.grid(False)
        
    #!!!!!!!need to fix missing xlabels, problem appears when twinx axis is added
    plt.tight_layout()
    plt.show()
    
#calculate mean class 1 proporsion and average contrast between values
def mean_1_contrast(df,cols):
    contrast = {}
    mean_1 = {}

    for col in cols:
        d = df.groupby([col,'target'])[col].count().unstack(level=1)
        d['class1_pct'] = d[1] / (d.sum(axis=1))
        contrast[col] = abs(d['class1_pct'].pct_change().dropna().values).mean()
        mean_1[col] = d['class1_pct'].mean()
    
    return pd.concat([pd.DataFrame(mean_1,index=['mean_class1']),
          pd.DataFrame(contrast,index=['contrast'])])




'''Read in train and test data from csv files'''
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv',index_col=0)
df_test = pd.read_csv('../input/cat-in-the-dat/test.csv',index_col=0)

#From df.info():
print('There are no missing values in the dataset.')

#Check the size of the dataset:
print(f"There are {df_train.shape[1]} features and {df_train.shape[0]:,} observation.")

#Display the statistic for categorical data
d = df_train.astype('str').describe().T
#add percentage for most frequently observed value in a given feature
d['top, %'] = d['freq']/df_train.shape[0]*100
#display the summary with some columns removed 
d.drop(['count','freq'],axis=1)




#group columns by type
target = 'target'
bin_cols = [col for col in df_train.columns if 'bin' in col]
cyclic_cols = ['day','month']
ord_cols = [col for col in df_train.columns if 'ord' in col]
nom_cols = [col for col in df_train.columns if 'nom' in col]
no_target = [col for col in df_train.columns if 'target' not in col]




plot_target_dist(df_train, bin_cols, figsize = (16,8), grid_r=2, grid_c=3)




#target: class 1 fraction
df_train.target.value_counts(normalize=True)[1].round(3)




#calculate contrast and avegare classes 1 proporsion
mean_1_contrast(df_train,bin_cols)




plot_target_dist(df_train,cyclic_cols, figsize = (18,5), grid_r=1, grid_c=2)




#calculate contrast and avegare classes 1 proporsion
mean_1_contrast(df_train,cyclic_cols)




plot_target_dist(df_train, ord_cols, figsize = (16,8), grid_r=3, grid_c=2)




#calculate contrast and avegare classes 1 proporsion
mean_1_contrast(df_train,ord_cols)




plot_target_dist(df_train, nom_cols[:5], figsize = (16,8), grid_r=3, grid_c=2)




#calculate contrast and avegare classes 1 proporsion
mean_1_contrast(df_train,nom_cols[:5])




plot_target_dist(df_train, nom_cols[5:], figsize = (16,8), grid_r=3, grid_c=2)




#calculate contrast and avegare classes 1 proporsion
mean_1_contrast(df_train,nom_cols[5:])




#distribution of class 1 fraction in nom_9 variable
col = 'nom_9'

fig, ax = plt.subplots(1,1,figsize=(6,3))

d = df_train.groupby([col,'target'])[col].count()            .unstack(level=1).sort_index()
#calculate the percentage of class 1
d['class1_pct'] = d[1] / (d.sum(axis=1))
d.fillna(0, inplace=True)
#plot a histogram
d.class1_pct.plot(kind = 'hist')

#label axis
plt.xlabel('Class 1 fraction', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
#add title
plt.title('Distribution of class 1 fraction for ' + col, fontsize=15)
plt.show()




#how many observation per given value in nom_9?
df_train.nom_9.value_counts().head()






