#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')


# In[2]:


get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[3]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[4]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[5]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[6]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[7]:


# Latest Pandas version
get_ipython().system("pip install -q 'pandas==0.25' --force-reinstall")


# In[8]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[9]:


print("Pandas version:", pd.__version__)


# In[10]:


import warnings
warnings.filterwarnings("ignore")


# In[11]:


import gc
gc.enable()


# In[12]:


import lightgbm as lgb
print("LightGBM version:", lgb.__version__)


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')


# In[15]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)


# In[16]:


y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# In[17]:


# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()
del train, test
gc.collect()


# In[18]:


X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


# In[19]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))


# In[20]:


# LGBMClassifier with GPU
clf = lgb.LGBMClassifier(
    max_bin = 63,
    num_leaves = 255,
    num_iterations = 500,
    learning_rate = 0.01,
    tree_learner = 'serial',
    task = 'train',
    is_training_metric = False,
    min_data_in_leaf = 1,
    min_sum_hessian_in_leaf = 100,
    sparse_threshold=1.0,
    device = 'gpu',
    num_thread = -1,
    save_binary= True,
    seed= 42,
    feature_fraction_seed = 42,
    bagging_seed = 42,
    drop_seed = 42,
    data_random_seed = 42,
    objective = 'binary',
    boosting_type = 'gbdt',
    verbose = 1,
    metric = 'auc',
    is_unbalance = True,
    boost_from_average = False,
)


# In[21]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[22]:


gc.collect()


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:20])
plt.title('LightGBM Feature Importance - Top 20')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances.png')


# In[24]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_lightgbm_gpu.csv')

