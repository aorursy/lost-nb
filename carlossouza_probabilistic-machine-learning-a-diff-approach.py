#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


# In[3]:


def chart(patient_id, ax):
    data = train[train['Patient'] == patient_id]
    x = data['Weeks']
    y = data['FVC']
    ax.set_title(patient_id)
    ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={'color':'red'})
    

f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart('ID00007637202177411956430', axes[0])
chart('ID00009637202177434476278', axes[1])
chart('ID00010637202177584971671', axes[2])


# In[4]:


# Kaggle, please add Pyro/PyTorch support!
import pymc3 as pm
import theano
import arviz as az
from sklearn import preprocessing


# In[5]:


# Very simple pre-processing: adding patient class
def patient_class(row):
    if row['Sex'] == 'Male':
        if row['SmokingStatus'] == 'Currently smokes':
            return 0
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 1
        elif row['SmokingStatus'] == 'Never smoked':
            return 2
    else:
        if row['SmokingStatus'] == 'Currently smokes':
            return 3
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 4
        elif row['SmokingStatus'] == 'Never smoked':
            return 5

train['Class'] = train.apply(patient_class, axis=1)


# In[6]:


# Very simple pre-processing: adding FVC and week baselines
aux = train[['Patient', 'Weeks']].groupby('Patient')    .min().reset_index()
aux = pd.merge(aux, train[['Patient', 'Weeks', 'FVC']], how='left', 
               on=['Patient', 'Weeks'])
aux = aux.groupby('Patient').mean().reset_index()
aux['Weeks'] = aux['Weeks'].astype(int)
aux['FVC'] = aux['FVC'].astype(int)
train = pd.merge(train, aux, how='left', on='Patient', suffixes=('', '_base'))


# In[7]:


# Very simple pre-processing: creating patient indexes
le = preprocessing.LabelEncoder()
train['PatientID'] = le.fit_transform(train['Patient'])

patients = train[['Patient', 'PatientID', 'Age', 'Class', 'Weeks_base', 'FVC_base']].drop_duplicates()
fvc_data = train[['Patient', 'PatientID', 'Weeks', 'FVC']]

patients.head()


# In[8]:


fvc_data.head()


# In[9]:


FVC_b = patients['FVC_base'].values
w_b = patients['Weeks_base'].values
age = patients['Age'].values
patient_class = patients['Class'].values

t = fvc_data['Weeks'].values
FVC_obs = fvc_data['FVC'].values
patient_id = fvc_data['PatientID'].values

with pm.Model() as hierarchical_model:
    # Hyperpriors for Alpha
    beta_int = pm.Normal('beta_int', 0, sigma=100)
    sigma_int = pm.HalfNormal('sigma_int', 100)
    
    # Alpha
    mu_alpha = FVC_b + beta_int * w_b
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_int, 
                      shape=train['Patient'].nunique())
    
    # Hyperpriors for Beta
    sigma_s = pm.HalfNormal('sigma_s', 100)
    alpha_s = pm.Normal('alpha_s', 0, sigma=100)
    beta_cs = pm.Normal('beta_cs', 0, sigma=100, shape=6)
    
    # Beta
    mu_beta = alpha_s + age * beta_cs[patient_class]
    beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_s,
                     shape=train['Patient'].nunique())
    
    # Model variance
    sigma = pm.HalfNormal('sigma', 200)
    
    # Model estimate
    FVC_est = alpha[patient_id] + beta[patient_id] * t
    
    # Data likelihood
    FVC_like = pm.Normal('FVC_like', mu=FVC_est,
                          sigma=sigma, observed=FVC_obs)


# In[10]:


# Inference button (TM)!
with hierarchical_model:
    trace = pm.sample(2000, tune=2000, target_accept=.9)


# In[11]:


with hierarchical_model:
    pm.traceplot(trace);


# In[12]:


def chart(patient_id, ax):
    data = train[train['Patient'] == patient_id]
    x = data['Weeks']
    y = data['FVC']
    ax.set_title(patient_id)
    ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={'color':'red'})
    
    x2 = np.arange(-12, 133, step=0.1)
    
    pid = patients[patients['Patient'] == patient_id]['PatientID'].values[0]
    for sample in range(100):
        alpha = trace['alpha'][sample, pid]
        beta = trace['beta'][sample, pid]
        sigma = trace['sigma'][sample]
        y2 = alpha + beta * x2
        ax.plot(x2, y2, linewidth=0.1, color='green')
        y2 = alpha + beta * x2 + sigma
        ax.plot(x2, y2, linewidth=0.1, color='yellow')
        y2 = alpha + beta * x2 - sigma
        ax.plot(x2, y2, linewidth=0.1, color='yellow')

f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart('ID00007637202177411956430', axes[0])
chart('ID00009637202177434476278', axes[1])
chart('ID00010637202177584971671', axes[2])


# In[13]:


# Very simple pre-processing: adding patient class
def patient_class(row):
    if row['Sex'] == 'Male':
        if row['SmokingStatus'] == 'Currently smokes':
            return 0
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 1
        elif row['SmokingStatus'] == 'Never smoked':
            return 2
    else:
        if row['SmokingStatus'] == 'Currently smokes':
            return 3
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 4
        elif row['SmokingStatus'] == 'Never smoked':
            return 5

test['Class'] = test.apply(patient_class, axis=1)
test = test.rename(columns={'FVC': 'FVC_base', 'Weeks': 'Weeks_base'})
test.head()


# In[14]:


# prepare submission dataset
submission = []
for i, patient in enumerate(test['Patient'].unique()):
    df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC'])
    df['Weeks'] = np.arange(-12, 134)
    df['Patient'] = patient
    df['PatientID'] = i
    df['FVC'] = 0
    submission.append(df)
    
submission = pd.concat(submission).reset_index(drop=True)
submission.head()


# In[15]:


FVC_b = test['FVC_base'].values
w_b = test['Weeks_base'].values
age = test['Age'].values
patient_class = test['Class'].values
t = submission['Weeks'].values
patient_id = submission['PatientID'].values
            
with pm.Model() as new_model:
    # Hyperpriors for Alpha
    beta_int = pm.Normal('beta_int', 
                         trace['beta_int'].mean(), 
                         sigma=trace['beta_int'].std())
    sigma_int = pm.TruncatedNormal('sigma_int', 
                                   trace['sigma_int'].mean(),
                                   sigma=trace['sigma_int'].std(),
                                   lower=0)
    
    # Alpha
    mu_alpha = FVC_b + beta_int * w_b
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_int, 
                      shape=test['Patient'].nunique())
    
    # Hyperpriors for Beta
    sigma_s = pm.TruncatedNormal('sigma_s', 
                                 trace['sigma_s'].mean(),
                                 sigma=trace['sigma_s'].std(),
                                 lower=0)
    alpha_s = pm.Normal('alpha_s', 
                        trace['alpha_s'].mean(), 
                        sigma=trace['alpha_s'].std())
    cov = np.zeros((6, 6))
    np.fill_diagonal(cov, trace['beta_cs'].var(axis=0))
    beta_cs = pm.MvNormal('beta_cs',
                          mu=trace['beta_cs'].mean(axis=0),
                          cov=cov,
                          shape=6)
    
    # Beta
    mu_beta = alpha_s + age * beta_cs[patient_class]
    beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_s,
                     shape=test['Patient'].nunique())
    
    # Model variance
    sigma = pm.TruncatedNormal('sigma', 
                               trace['sigma'].mean(),
                               sigma=trace['sigma'].std(),
                               lower=0)
    
    # Model estimate
    FVC_est = pm.Normal('FVC_est', mu=alpha[patient_id] + beta[patient_id] * t, 
                        sigma=sigma,
                        shape=submission.shape[0])


# In[16]:


with new_model:
    trace2 = pm.sample(2000, tune=2000, target_accept=.9)
    
trace2['FVC_est'].shape


# In[17]:


with new_model:
    pm.traceplot(trace2);


# In[18]:


preds = pd.DataFrame(data=trace2['FVC_est'].T)
submission = pd.merge(submission, preds, left_index=True, 
                      right_index=True)
submission['Patient_Week'] = submission['Patient'] + '_'     + submission['Weeks'].astype(str)
submission = submission.drop(columns=['FVC', 'PatientID'])

FVC = submission.iloc[:, :-1].mean(axis=1)
confidence = submission.iloc[:, :-1].std(axis=1)
submission['FVC'] = FVC
submission['Confidence'] = confidence
submission = submission[['Patient', 'Weeks', 'Patient_Week', 
                         'FVC', 'Confidence']]


# In[19]:


temp = pd.merge(train[['Patient', 'Weeks', 'FVC']], 
                submission.drop(columns=['Patient_Week']),
                on=['Patient', 'Weeks'], how='left', 
                suffixes=['', '_pred'])
temp = temp.dropna()
temp = temp.groupby('Patient')

# The metric only uses the last 3 measurements, the most uncertain
temp = temp.tail(3)


# In[20]:


sigma_clipped = temp['Confidence'].apply(lambda s: max(s, 70))
delta = temp.apply(lambda row: min([abs(row['FVC'] - row['FVC_pred']), 1000]), axis=1)
metric = -np.sqrt(2) * delta / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
metric.mean()


# In[21]:


submission = submission[['Patient_Week', 'FVC', 'Confidence']]
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




