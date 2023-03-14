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


# Replace 'kaggle-competitions-project' with YOUR OWN project id here --  
PROJECT_ID = 'kaggle-bq-quest'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID) # location="US")
#dataset = client.create_dataset('bqml_example', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

# create a reference to our table
table = client.get_table("kaggle-bq-quest.ms_maleware.train")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()


# In[3]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[4]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `ms_maleware.model1`\nOPTIONS\n( model_type='LOGISTIC_REG',\n    auto_class_weights=TRUE\n  ) AS\nSELECT\n    HasDetections as label,\n    SmartScreen,\n    AppVersion,\n    Census_InternalBatteryNumberOfCharges,\n    AVProductStatesIdentifier,\n    Census_TotalPhysicalRAM,\n    LocaleEnglishNameIdentifier,\n    Census_SystemVolumeTotalCapacity,\n    AVProductsEnabled,\n    Census_InternalPrimaryDiagonalDisplaySizeInInches,\n    Census_FirmwareVersionIdentifier,\n    Census_OSInstallTypeName,\n    Census_OSBuildNumber,\n    Census_FirmwareManufacturerIdentifier,\n    Census_ActivationChannel,\n    Census_OSArchitecture,\n    Census_ProcessorCoreCount,\n    Census_OSEdition,\n    Census_PrimaryDiskTypeName,\n    Census_IsSecureBootEnabled,\n    IsProtected,\n    Census_InternalPrimaryDisplayResolutionVertical,\n    Census_OSBuildRevision,\n    Census_InternalPrimaryDisplayResolutionHorizontal,\n    Census_HasOpticalDiskDrive,\n    OrganizationIdentifier,\n    EngineVersion,\n    ProductName\nFROM\n  `kaggle-bq-quest.ms_maleware.train`")


# In[5]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n    *\nFROM\n  ML.TRAINING_INFO(MODEL `ms_maleware.model1`)\nORDER BY iteration ')


# In[6]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `ms_maleware.model1`, (\n  SELECT\n    HasDetections as label,\n    SmartScreen,\n    AppVersion,\n    Census_InternalBatteryNumberOfCharges,\n    AVProductStatesIdentifier,\n    Census_TotalPhysicalRAM,\n    LocaleEnglishNameIdentifier,\n    Census_SystemVolumeTotalCapacity,\n    AVProductsEnabled,\n    Census_InternalPrimaryDiagonalDisplaySizeInInches,\n    Census_FirmwareVersionIdentifier,\n    Census_OSInstallTypeName,\n    Census_OSBuildNumber,\n    Census_FirmwareManufacturerIdentifier,\n    Census_ActivationChannel,\n    Census_OSArchitecture,\n    Census_ProcessorCoreCount,\n    Census_OSEdition,\n    Census_PrimaryDiskTypeName,\n    Census_IsSecureBootEnabled,\n    IsProtected,\n    Census_InternalPrimaryDisplayResolutionVertical,\n    Census_OSBuildRevision,\n    Census_InternalPrimaryDisplayResolutionHorizontal,\n    Census_HasOpticalDiskDrive,\n    OrganizationIdentifier,\n    EngineVersion,\n    ProductName\n  FROM\n    `kaggle-bq-quest.ms_maleware.train`\n    ))')


# In[7]:


get_ipython().run_cell_magic('bigquery', 'df', 'SELECT\n    *\nFROM\n  ML.PREDICT(MODEL `ms_maleware.model1`,\n    (\n    SELECT\n        MachineIdentifier,\n        SmartScreen,\n        AppVersion,\n        Census_InternalBatteryNumberOfCharges,\n        AVProductStatesIdentifier,\n        Census_TotalPhysicalRAM,\n        LocaleEnglishNameIdentifier,\n        Census_SystemVolumeTotalCapacity,\n        AVProductsEnabled,\n        Census_InternalPrimaryDiagonalDisplaySizeInInches,\n        Census_FirmwareVersionIdentifier,\n        Census_OSInstallTypeName,\n        Census_OSBuildNumber,\n        Census_FirmwareManufacturerIdentifier,\n        Census_ActivationChannel,\n        Census_OSArchitecture,\n        Census_ProcessorCoreCount,\n        Census_OSEdition,\n        Census_PrimaryDiskTypeName,\n        Census_IsSecureBootEnabled,\n        IsProtected,\n        Census_InternalPrimaryDisplayResolutionVertical,\n        Census_OSBuildRevision,\n        Census_InternalPrimaryDisplayResolutionHorizontal,\n        Census_HasOpticalDiskDrive,\n        OrganizationIdentifier,\n        EngineVersion,\n        ProductName\n    FROM\n      `kaggle-bq-quest.ms_maleware.test`))')


# In[8]:


df = df[['predicted_label', 'MachineIdentifier']]
df.head()


# In[9]:


df.rename(columns={'MachineIdentifier':'MachineIdentifier', 'predicted_label': 'HasDetections'}, inplace=True)


# In[10]:


df = df[['MachineIdentifier', 'HasDetections']]


# In[11]:


df.head()


# In[12]:


df.to_csv(r'submission.csv',index=False)

