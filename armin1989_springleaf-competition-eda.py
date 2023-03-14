#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




# for the EDA I only use 20% of the training data, randomly sampled from original dataset, I have saved this in a different .csv file to make
# re-running of this notebook faster, for the first run un-comment the following line
#train_data = pd.read_csv('../input/train.csv').sample(frac=0.2, axis=0)
train_data = pd.read_csv('../input/springleaf-train-small/train_small.csv')
train_data.drop(['Unnamed: 0'], axis=1, inplace=True)  # this feature was created during subsampling
original_feat_dim = train_data.shape[1]
train_data.info()




train_data.head(10)




# looking at categorical columns
categorical = train_data.select_dtypes(include=[np.object]).columns
categorical, len(categorical)




# looking at numerical columns
numerical = train_data.select_dtypes(include=[np.number]).columns
numerical, len(numerical)




plt.hist(train_data["target"])




print("Ratio of positive samples (target == 1) to negative samples (target == 0) is {}"          .format(len(train_data[train_data["target"] == 1]) / len(train_data[train_data["target"] == 0])))




null_count = train_data.isnull().sum(axis=0).sort_values(ascending=False)
null_count.head(30)




# figuring out how many features have more than 10% of data missing them
np.sum(null_count > 0.1 * train_data.shape[0])




# I think its safe to remove these features, I will create a to_remove list to store all features that need to be dropped
missing = [feature for feature in train_data.columns if null_count[feature] > 0.1 * train_data.shape[0]]
train_data.drop(missing, axis=1, inplace=True)




# taking a second look
total = train_data.isnull().sum().sort_values(ascending=False)
percent = total / train_data.shape[0]
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)




remove_missing_examples = [feat for feat in percent.index if percent[feat] < 0.01]
fill_missing_examples = [feat for feat in percent.index if percent[feat] > 0.01]

# filling in the missing data for features containing many of them
for col in fill_missing_examples:
    if col in categorical:
        # fill missing data with mode
        train_data[col].fillna(train_data[col].mode(), inplace=True)
    else:
        # fill missing data with mean
        train_data[col].fillna(train_data[col].mean(), inplace=True)

# removing rows with missing data in them (only a few examples will be deleted at this point)
train_data.dropna(axis=0, inplace=True)




# final check for missing data
train_data.isnull().sum().max()




values_count = train_data.nunique(dropna=False).sort_values()
np.sum(values_count == 1)




# noting features with constant values for removal
constants = [feature for feature in train_data.columns if values_count[feature] == 1]
train_data.drop(constants, axis=1, inplace=True)




enc_data =  pd.DataFrame(index=train_data.index)
for col in train_data.columns:
    enc_data[col] = train_data[col].factorize()[0]




# noting and dropping duplicated features 
duplicates = []
data_mean = np.mean(enc_data)
data_std = np.std(enc_data)
for i, ref in enumerate(train_data.columns[:-1]):
    for other in train_data.columns[i + 1:-1]:
        if other not in duplicates and data_mean[ref] == data_mean[other] and            data_std[ref] == data_std[other] and np.all(enc_data[ref] == enc_data[other]):
            duplicates.append(other)    
train_data.drop(duplicates, axis=1, inplace=True)
len(duplicates)




print("Dimension of feature space before data cleaning : %d" % original_feat_dim)
print("Dimension of feature space after data cleaning : %d" % train_data.shape[1])
print("Number of features removed : %d" % (original_feat_dim - train_data.shape[1]))




categorical = train_data.select_dtypes(include=[np.object]).columns
numerical = train_data.select_dtypes(include=[np.number]).columns




train_data[categorical].describe()




potential_to_remove = ['VAR_0226', 'VAR_0230', 'VAR_0236']




train_data[categorical].head(20)




dates = ['VAR_0075', 'VAR_0204', 'VAR_0217']
locations = ['VAR_200', 'VAR_0237', 'VAR_274']

date1 = pd.to_datetime(train_data["VAR_0075"],format = '%d%b%y:%H:%M:%S')
date2 = pd.to_datetime(train_data["VAR_0217"],format = '%d%b%y:%H:%M:%S')
np.all(date2 > date1)




train_data[['VAR_0404', 'VAR_0466', 'VAR_0467', 'VAR_0493']].nunique(dropna=False)




train_data['VAR_0466'].value_counts() / train_data.shape[0]




train_data['VAR_0467'].value_counts() / train_data.shape[0]




# converting categorical features to encoded variables first, 
# for features represnting date, we separate year, month day, hour and minute, this is a very crude represntation, we may 
# need more interesting time features like time of the day, weekday and so on
from datetime import datetime
year_func = lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S" ).year
month_func = lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S" ).month
day_func = lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S" ).day
hour_func = lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S" ).hour
minute_func = lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S" ).minute

enc_data =  pd.DataFrame(index=train_data.index)
for col in categorical:
    enc_data[col] = train_data[col].factorize()[0]
    if col in dates:
        enc_data[col + '_year'] = train_data[col].map(year_func)
        enc_data[col + '_month'] = train_data[col].map(month_func)
        enc_data[col + '_day'] = train_data[col].map(day_func)
        enc_data[col + '_hour'] = train_data[col].map(hour_func)
        enc_data[col + '_minute'] = train_data[col].map(minute_func)
expanded_categoricals = list(enc_data.columns) # saving which variables are categorical for possible one hot encoding 
enc_data["target"] = train_data["target"]
# finding correlation and looking at it
corrmat = enc_data.corr()
corrmat["target"].sort_values(ascending=False)




enc_data.drop(['VAR_0075_hour', 'VAR_0075_minute', 'VAR_0204_year', 'VAR_0217_hour', 'VAR_0217_minute'], axis=1, inplace=True)
corrmat = enc_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True)




numerical = train_data.select_dtypes(include=[np.number]).columns
corrmat2 = pd.concat([train_data[numerical], train_data['target']], axis = 0).corr()
corrmat2["target"].sort_values(ascending=False)




# getting more important features
treshold = 0.17
importants = [feat for feat in corrmat2.columns if abs(corrmat2['target'].loc[feat]) > treshold]

# randomly selecting 6 features from importants
sample_feats = np.random.choice(importants[:-1], 6)

# looking at histograms
f, ax = plt.subplots(figsize=(20, 10))
for i in range(len(sample_feats)):
    plt.subplot(2, 3, i + 1)
    sns.distplot(train_data[sample_feats[i]])




f, ax = plt.subplots(figsize=(20, 10))
f.suptitle('Features With High Correlation (6 randomly chosen ones)', size=14)
for i in range(len(sample_feats)):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='target', y=sample_feats[i], data=train_data)




high_corrs = []
for i, c1 in enumerate(corrmat2.columns):
    for c2 in corrmat2.columns[i + 1:]:
        if abs(corrmat2[c1].loc[c2]) > 0.8:
            high_corrs.append((c1, c2))
len(high_corrs)




np.savez('./prep_info.npz', missing, fill_missing_examples, remove_missing_examples, constants, duplicates,
         dates, locations, potential_to_remove, expanded_categoricals)




from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA, KernelPCA, FastICA




# first step, separating features and targets, remember that we have already encoded categorical data and stored the result in enc_data 
num_samples = 1000 # number of samples to use from each class of target for dimensionality reduction, reduce this if you are impatient!
numerical_feats = train_data[numerical].drop(['target'], axis=1, inplace=False)
new_data = pd.concat([numerical_feats, enc_data], axis=1)
positive_samples = new_data[new_data['target'] == 1].sample(num_samples)
negative_samples = new_data[new_data['target'] == 0].sample(num_samples)
to_transform = pd.concat([positive_samples, negative_samples]).sample(frac=1).reset_index(drop=True)
to_transform.head(20)




features = to_transform.drop(['target'], axis=1, inplace=False)
labels = to_transform['target']




pca_embedding =  PCA(n_components=2) 
pca_emb_data = pca_embedding.fit_transform(features.values)
plt.figure(figsize=(15,15))
plt.scatter(pca_emb_data[labels == 1, 0], pca_emb_data[labels == 1, 1], color='red', label='positive samples')
plt.scatter(pca_emb_data[labels == 0, 0], pca_emb_data[labels == 0, 1], color='blue', label='negative samples')
plt.legend()




kpca_embedding =  KernelPCA(n_components=2, kernel='rbf')
kpca_emb_data = kpca_embedding.fit_transform(features.values)
plt.figure(figsize=(15,15))
plt.title('Reduced data with kernel PCA (RBF kernel)')
plt.scatter(kpca_emb_data[labels == 1, 0], kpca_emb_data[labels == 1, 1], color='red', label='positive samples')
plt.scatter(kpca_emb_data[labels == 0, 0], kpca_emb_data[labels == 0, 1], color='blue', label='negative samples')
plt.legend()




tsne_embedding =  TSNE(n_components=2) 
tsne_emb_data = tsne_embedding.fit_transform(features.values)
plt.figure(figsize=(15,15))
plt.title('Reduced data with tSNE')
plt.scatter(tsne_emb_data[labels == 1, 0], tsne_emb_data[labels == 1, 1], color='red', label='positive samples')
plt.scatter(tsne_emb_data[labels == 0, 0], tsne_emb_data[labels == 0, 1], color='blue', label='negative samples')
plt.legend()




spec_embedding = SpectralEmbedding(n_components=2, affinity='rbf')
transformed_data2 = spec_embedding.fit_transform(features.values)
fig = plt.figure(figsize=(30,10))
plt.subplot(1, 3, 1)
plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')
plt.legend()
plt.subplot(1, 3, 3)
plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')
plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')
plt.legend()




spec_embedding2 = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', n_neighbors=30)
transformed_data2 = spec_embedding2.fit_transform(features.values)
fig = plt.figure(figsize=(30,10))
plt.subplot(1, 3, 1)
plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')
plt.legend()
plt.subplot(1, 3, 3)
plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')
plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')
plt.legend()

