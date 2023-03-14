#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob

import pandas as pd


# In[2]:


categories = [cat.split('/')[-1] for cat in glob.glob('../input/train/*')]


# In[3]:


categories


# In[4]:


df = pd.DataFrame()


# In[5]:


for cat in categories:
    cat_pat = '../input/train/{}/*'.format(cat)
    files_cat = glob.glob(cat_pat)
    
    cat_df =  pd.DataFrame([{'label': cat, 'feature': file} for file in files_cat])
    
    df = df.append(cat_df)


# In[6]:


df['label'].value_counts()


# In[7]:


df[df['feature'].str.contains('png')]


# In[8]:


df[df['feature'].str.contains('jpg')]


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test= pd.Series(), pd.Series(), pd.Series(), pd.Series()


# In[11]:


for cat in categories:
    # split each catagory into proportional training and test sizes.
    temp_df = df[df['label'] == cat]
    X, y = temp_df['feature'], temp_df['label']
    
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y, 
                                                                random_state=42, 
                                                                test_size=0.2)
    
    X_train = X_train.append(X_train_t)
    X_test = X_test.append(X_test_t)
    y_train = y_train.append(y_train_t)
    y_test = y_test.append(y_test_t)


# In[12]:


len(y_test)


# In[13]:


len(X_train)


# In[14]:


from sklearn.neural_network import MLPClassifier

 clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

