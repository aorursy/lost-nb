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


ls ../input/firstsubmission


# In[3]:


import numpy as np 
import pandas as pd 
import os 

my_result = pd.read_csv(os.path.join('/kaggle/input/firstsubmission', 'submission.csv'), index_col=False)
sam_submi = pd.read_csv(os.path.join('/kaggle/input/deepfake-detection-challenge','sample_submission.csv'), index_col=False)
new_result = pd.merge(sam_submi, my_result, on='filename', how='left')
new_result.fillna(0, inplace=True)
new_result['label'] = new_result['label_y']
new_result = new_result[['filename', 'label']]
new_result['label'] = new_result['label'].astype('int64')
new_result.to_csv('submission.csv', index=False)


# In[4]:


ls


# In[5]:


import pandas as pd
submission = pd.read_csv("../input/firstsubmission/submission.csv")
submission.head()


# In[6]:


import pandas as pd


# In[7]:


ls


# In[8]:


submitted = pd.read_csv('../input/firstsubmission/submission.csv')

submission = pd.DataFrame(submitted, columns=['filename', 'label']).fillna(0.5)
submission.sort_values('filename').to_csv('submission.csv', index=False)
submission.to_csv('submission.csv', index=False)


# In[9]:


ls


# In[10]:



file = open("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
file2 = open('../input/firstsubmission/submission.csv', 'r')

for line in file:
    tocheck = str(line).split(',')
    for secondline in file2:
        if 'filename' not in str(secondline) and str(tocheck[0]) in secondline:
            print('yes')

# file.close()
# file2.close()


# In[11]:




from pandas import *

original = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
submitted = pd.read_csv('../input/firstsubmission/submission.csv')
for i, row in enumerate(original.values):
    filename, label = row
    for x, rows in enumerate(submitted.values):
        file, lable = rows
        if str(filename) == str(file):
            original.replace(to_replace =label,  
                            value =lable) 
            #label = lable

original.to_csv('newsubmission.csv', index=False)


# In[12]:


from pandas import *
import pandas as pd

original = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
submitted = pd.read_csv('../input/firstsubmission/submission.csv')
originalar = []
submittedar = []
for i, row in enumerate(original.values):
    filename, label = row
    originalar.append(filename)
for x, rows in enumerate(submitted.values):
    file, lable = rows
    print(lable)


# In[13]:


ls


# In[ ]:




