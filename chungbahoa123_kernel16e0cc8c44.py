#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[1]:


import os, math, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
pd.set_option('display.max_colwidth', -1)


# In[5]:


# load data
pdf_data = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
print(pdf_data.shape)
pdf_data.head()


# In[6]:


get_ipython().run_cell_magic('time', '', 'profile = pandas_profiling.ProfileReport(pdf_data)\nprofile')


# In[7]:


pip install pandas_profiling


# In[ ]:




