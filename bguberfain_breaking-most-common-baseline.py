#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import io
import bson
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import cpu_count


# In[2]:


num_images = 200000

def imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)

def img2feat(im):
    return cv2.resize(im, (8, 8), interpolation=cv2.INTER_AREA).ravel()

X = []
y = []

bar = tqdm_notebook(total=num_images)
with open('../input/train.bson', 'rb') as f:
    data = bson.decode_file_iter(f)

    i = 0
    try:
        for c, d in enumerate(data):
            target = d['category_id']
            for e, pic in enumerate(d['imgs']):
                picture = imread(pic['picture'])
                x = img2feat(picture)
                
                X.append(x)
                y.append(target)
                
                i = i + 1
                bar.update()

                if i >= num_images:
                    raise IndexError()

    except IndexError:
        pass;


# In[3]:


X = np.array(X, dtype=np.float32)
y = pd.Series(y)

X.shape, y.shape


# In[4]:


num_classes = 30  # This will reduce the max accuracy to just above 0.2

# Now we must find the most `num_classes-1` frequent classes
# (there will be an aditional 'other' class)
valid_targets = set(y.value_counts().index[:num_classes-1].tolist())
valid_y = y.isin(valid_targets)

# Set other classes to -1
y[~valid_y] = -1

max_acc = valid_y.mean()
print(max_acc)


# In[5]:


# Now we categorize the dataframe
y, rev_labels = pd.factorize(y)


# In[6]:


# Now we have a X,y pair. Let's train a simple KNN Classifier
# import xgboost as xgb  # This run out of time for this task

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(3)
knn.fit(X, y)


# In[7]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='_id')

most_frequent_guess = 1000018296
submission['category_id'] = most_frequent_guess # Most frequent guess


# In[8]:


num_images_test = 100000  # We only have time for a few test images..
num_cpus = cpu_count()

def predict(d, bar):
    picture = imread(d['imgs'][0]['picture'])
    x = img2feat(picture)
    y_cat = rev_labels[knn.predict(x[None])[0]]
    if y_cat == -1:
        y_cat = most_frequent_guess

    bar.update()
    
    return d['_id'], y_cat

bar = tqdm_notebook(total=num_images_test)
with open('../input/test.bson', 'rb') as f,          concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

    data = bson.decode_file_iter(f)

    future_predict = []
    
    for i,d in enumerate(data):
        if i >= num_images_test:
            break
        future_predict.append(executor.submit(predict, d, bar))

    for future in concurrent.futures.as_completed(future_predict):
        _id, y_cat = future.result()
        submission.loc[_id, 'category_id'] = y_cat


# In[9]:


submission.to_csv('new_submission.csv.gz', compression='gzip')

