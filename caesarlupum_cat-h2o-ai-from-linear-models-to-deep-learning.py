#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install gxx_linux-64 gcc_linux-64 swig


# In[2]:


import h2o
h2o.init(ip="localhost", port=54323)


# In[3]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[4]:



train_data = h2o.import_file("/kaggle/input/cat-in-the-dat/train.csv")
test_data = h2o.import_file("/kaggle/input/cat-in-the-dat/test.csv")


# In[5]:


test_id = h2o.import_file('/kaggle/input/cat-in-the-dat/test.csv')['id']


# In[6]:



from h2o.estimators.glm import H2OGeneralizedLinearEstimator


# In[7]:


glm_fit1 = H2OGeneralizedLinearEstimator(family='binomial', model_id='glm_fit1')


# In[8]:


train_data["target"] = train_data["target"].asfactor()


# In[9]:


train, valid, test = train_data.split_frame(ratios=[0.7, 0.15], seed=42)  
y = 'target'
x = list(train_data.columns)


# In[10]:


id_var = 'id'
x.remove(id_var)  #remove the response


# In[11]:


x.remove(y)  #remove the response
print(x)


# In[12]:


glm_fit1.train(x=x, y=y, training_frame=train)


# In[13]:


glm_fit2 = H2OGeneralizedLinearEstimator(family='binomial', model_id='glm_fit2', lambda_search=True,balance_classes = True)
glm_fit2.train(x=x, y=y, training_frame=train, validation_frame=valid)


# In[14]:


glm_perf1 = glm_fit1.model_performance(test)
glm_perf2 = glm_fit2.model_performance(test)


# In[15]:



print (glm_perf1.gini())
print (glm_perf2.gini())


# In[16]:



print (glm_fit2.gini(train=True))
print (glm_fit2.gini(valid=True))


# In[17]:



from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[18]:



rf_fit1 = H2ORandomForestEstimator(model_id='rf_fit1',   seed=1)


# In[19]:


rf_fit1.train(x=x, y=y, training_frame=train,validation_frame=valid)


# In[20]:


rf_fit2 = H2ORandomForestEstimator(model_id='rf_fit2', ntrees=100,   seed=1)
rf_fit2.train(x=x, y=y, training_frame=train,validation_frame=valid)


# In[21]:


rf_perf1 = rf_fit1.model_performance(test)
rf_perf2 = rf_fit2.model_performance(test)


# In[22]:



print(rf_perf1.gini())
print(rf_perf2.gini())


# In[23]:


rf_fit3 = H2ORandomForestEstimator(model_id='rf_fit3', seed=1, nfolds=5)
rf_fit3.train(x=x, y=y, training_frame=train)


# In[24]:


print( rf_fit3.gini(xval=True))


# In[25]:



from h2o.estimators.gbm import H2OGradientBoostingEstimator


# In[26]:



gbm_fit1 = H2OGradientBoostingEstimator(model_id='gbm_fit1',   seed=1)
gbm_fit1.train(x=x, y=y, training_frame=train, validation_frame=valid)


# In[27]:


gbm_fit2 = H2OGradientBoostingEstimator(model_id='gbm_fit2', ntrees=500,   seed=1)
gbm_fit2.train(x=x, y=y, training_frame=train,validation_frame=valid)


# In[28]:


# Now let's use early stopping to find optimal ntrees

gbm_fit3 = H2OGradientBoostingEstimator(model_id='gbm_fit3', 
                                        ntrees=1000, 
                                        score_tree_interval=5,     #used for early stopping
                                        stopping_rounds=3,         #used for early stopping
                                        stopping_metric='AUC',     #used for early stopping
                                        stopping_tolerance=0.0005, #used for early stopping
                                        seed=1)
# The use of a validation_frame is recommended with using early stopping
gbm_fit3.train(x=x, y=y, training_frame=train, validation_frame=valid)


# In[29]:


# Let's try XGBOOSTING
from h2o.estimators import H2OXGBoostEstimator
param = {
      "model_id": 'gbm_fit4'
    , "ntrees" : 100
    , "max_depth" : 10
    , "learn_rate" : 0.02
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 4241
    , "score_tree_interval": 100
}
gbm_fit4 = H2OXGBoostEstimator(**param)
gbm_fit4.train(x=x, y=y, training_frame=train, validation_frame=valid)


# In[30]:


gbm_perf1 = gbm_fit1.model_performance(test)
gbm_perf2 = gbm_fit2.model_performance(test)
gbm_perf3 = gbm_fit3.model_performance(test)
gbm_perf4 = gbm_fit4.model_performance(test)


# In[31]:



print (gbm_perf1.gini())
print (gbm_perf2.gini())
print (gbm_perf3.gini())
print (gbm_perf4.gini())


# In[32]:


# Import H2O DL:
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# In[33]:


# Initialize and train the DL estimator:

dl_fit1 = H2ODeepLearningEstimator(model_id='dl_fit1',   seed=1,  balance_classes = True)
dl_fit1.train(x=x, y=y, training_frame=train,validation_frame=valid)


# In[34]:


dl_fit2 = H2ODeepLearningEstimator(model_id='dl_fit2', 
                                   epochs=50, 
                                   hidden=[10,10], 
                                   stopping_rounds=0,  #disable early stopping
                                   seed=1,
                                   balance_classes = True)
dl_fit2.train(x=x, y=y, training_frame=train,validation_frame=valid)


# In[35]:


dl_fit3 = H2ODeepLearningEstimator(model_id='dl_fit3', 
                                   epochs=500, 
                                   hidden=[10,10],
                                   score_interval=1,          #used for early stopping
                                   stopping_rounds=50,         #used for early stopping
                                   stopping_metric='AUC',     #used for early stopping
                                   stopping_tolerance=0.0005, #used for early stopping
                                   seed=1,  
                                   balance_classes = True)
dl_fit3.train(x=x, y=y, training_frame=train, validation_frame=valid)


# In[36]:


dl_perf1 = dl_fit1.model_performance(test)
dl_perf2 = dl_fit2.model_performance(test)
dl_perf3 = dl_fit3.model_performance(test)


# In[37]:


# Retreive test set AUC
print (dl_perf1.gini())
print (dl_perf2.gini())
print( dl_perf3.gini())


# In[38]:


test_pred = gbm_fit4.predict(test_id) # test


# In[39]:


test_pred

