#!/usr/bin/env python
# coding: utf-8

# In[1]:


# download runtime and unpack
get_ipython().system('wget https://download.knime.org/analytics-platform/linux/knime_4.1.2.linux.gtk.x86_64.tar.gz')
get_ipython().system('tar xvzf knime_4.1.2.linux.gtk.x86_64.tar.gz')
get_ipython().system('rm knime_4.1.2.linux.gtk.x86_64.tar.gz')
get_ipython().system('unzip ./knime_4.1.2/knime-workspace.zip -d ./knime_4.1.2/knime-workspace/')
get_ipython().system('rm ./knime_4.1.2/knime-workspace.zip')
# copy the workflow
get_ipython().system('cp -R /kaggle/input/knime-cat-publ ./knime_4.1.2/knime-workspace/')
# set memory settings for the runtime
get_ipython().system("sed -i 's/Xmx2048m/Xmx10240m/g' ./knime_4.1.2/knime.ini")
# install runtime extensions - python and H2O integrations
get_ipython().system("./knime_4.1.2/knime -application org.eclipse.equinox.p2.director -nosplash -consolelog -r 'http://update.knime.com/analytics-platform/4.1,http://update.knime.com/community-contributions/4.1,http://update.knime.com/community-contributions/trusted/4.1,http://update.knime.com/partner/4.1' -i 'org.knime.features.python2.feature.group,org.knime.features.ext.h2o.feature.group,org.knime.features.datageneration.feature.group' -d ./knime_4.1.2/")
# install the wrapper python package
get_ipython().system('pip install knime')
# used in the workflow
get_ipython().system('pip install dfply')


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import knime

# Any results you write to the current directory are saved as output.
knime.executable_path = "./knime_4.1.2/knime"
workspace = "./knime_4.1.2/knime-workspace"
workflow = "knime-cat-publ/cat_publ/cat_publ"


# In[3]:


knime.Workflow(workflow_path=workflow,workspace_path=workspace)


# In[4]:


with knime.Workflow(workflow_path=workflow,workspace_path=workspace) as wf:
    wf.execute()


# In[5]:


# Alternatively wf can be executed through the command-line processor directly (instead of Python wrapper - you'll get more detailed output)
#!./knime_4.1.2/knime -nosplash -application org.knime.product.KNIME_BATCH_APPLICATION -workflowDir="./knime_4.1.2/knime-workspace/knime-cat-publ/cat_publ"


# In[6]:


rm -rf knime_4.1.2

