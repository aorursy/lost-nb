#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import randint
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data_directory = os.path.join(os.getcwd(), '../input')
print(os.listdir(data_directory))


# In[3]:


get_ipython().system("cat '../input/readme.txt'")


# In[4]:


data_dictionnary = {}

operational_settings = ['op_setting_{}'.format(i + 1) for i in range (3)]
sensor_columns = ['sensor_{}'.format(i + 1) for i in range(27)]
features = operational_settings + sensor_columns
metadata = ['engine_no', 'time_in_cycles']
list_columns = metadata + features


list_file_train = [x for x in sorted(os.listdir(data_directory)) if 'train' in x]

for file_train in list_file_train:
    data_set_name = file_train.replace('train_', '').replace('.txt', '')
    file_test = 'test_' + data_set_name + '.txt'
    rul_test = 'RUL_' + data_set_name + '.txt'
    
    data_dictionnary[data_set_name] = {
        'df_train': pd.read_csv(os.path.join(data_directory, file_train), sep=' ', header=-1, names=list_columns),
        'df_test': pd.read_csv(os.path.join(data_directory, file_test), sep=' ', header=-1, names=list_columns),
        'RUL_test' :pd.read_csv(os.path.join(data_directory, rul_test), header=-1, names=['RUL']),
    }


# In[5]:


def add_rul(g):
    g['RUL'] = [max(g['time_in_cycles'])] * len(g)
    g['RUL'] = g['RUL'] - g['time_in_cycles']
    del g['engine_no']
    return g.reset_index()

for data_set in data_dictionnary:
    data_dictionnary[data_set]['df_train'] = data_dictionnary[data_set]['df_train']                        .groupby('engine_no').apply(add_rul).reset_index()
    del data_dictionnary[data_set]['df_train']['level_1']


# In[6]:


CHOSEN_DATASET = 'FD001'

df = data_dictionnary[CHOSEN_DATASET]['df_train'].copy()

df_eval = data_dictionnary[CHOSEN_DATASET]['df_test'].copy()


# In[7]:


dataset_description = df.describe()
dataset_description


# In[8]:


axes = dataset_description.T.plot.bar(subplots=True, figsize=(15,10))


# In[9]:


###############< ??? >###############
# What can you conclude from the graph above?


# In[10]:


df_plot = df.copy()[features]
df_corr = df_plot.corr(method='pearson')
fig, ax = plt.subplots(figsize=(15,15))
axes = sns.heatmap(df_corr, linewidths=.2, )


# In[11]:


###############< ??? >###############
# Can you plot a correlation matrix with another correlation coeficient?


# In[12]:


###############< ??? >###############
# What can append when you have correlated features?


# In[13]:


nan_column = df.columns[df.isna().any()].tolist()
const_columns = [c for c in df.columns if len(df[c].drop_duplicates()) <= 2]
print('Columns with all nan: \n' + str(nan_column) + '\n')
print('Columns with all const values: \n' + str(const_columns) + '\n')


# In[14]:


###############< ??? >###############
# Can you find all the couples that are strongly correlated ?


# In[15]:


df_plot = df.copy()
df_plot = df_plot.sort_values(metadata)
graph = sns.PairGrid(data=df_plot, x_vars="RUL", y_vars=features, hue="engine_no", height=4, aspect=6,)
graph = graph.map(plt.plot, alpha=0.5)
graph = graph.set(xlim=(df_plot['RUL'].max(),df_plot['RUL'].min()))
# graph = graph.add_legend()


# In[16]:


###############< ??? >###############
# What can you see from the graphs above?


# In[17]:


###############< ??? >###############
# Is is better to train on a smaller part?


# In[18]:


number_of_engine_no = len(df['engine_no'].drop_duplicates())

engine_no_val = range(50, 70)
engine_no_train = [x for x in range(number_of_engine_no) if x not in engine_no_val]


# In[19]:


selected_features = [x for x in features if x not in nan_column + const_columns]


# In[20]:


data_train = df[df['engine_no'].isin(engine_no_train)]
data_val = df[df['engine_no'].isin(engine_no_val)]

X_train, y_train = data_train[selected_features], data_train['RUL'] 
X_val, y_val = data_val[selected_features], data_val['RUL']

X_eval = df_eval[selected_features]


X_all, y_all = df[selected_features], df['RUL']


# In[21]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)


# In[22]:


print("Score on train data : " + str(rf_reg.score(X_train, y_train)))
print("Score on test data : " + str(rf_reg.score(X_val, y_val)))


# In[23]:


###############< ??? >###############
# Did you overfit?


# In[24]:


###############< ??? >###############
# Can you have the RMSE?


# In[25]:


###############< ??? >###############
# Try to improve you model.


# In[26]:


df_pred = data_train.copy()
df_pred['pred'] = rf_reg.predict(X_train)
df_pred['error'] = df_pred['pred'] - df_pred['RUL']


# In[27]:


df_plot = df_pred.copy()
df_plot = df_plot.sort_values(['engine_no', 'time_in_cycles'])
g = sns.PairGrid(data=df_plot, x_vars="RUL", y_vars=['RUL', 'pred', 'error'], hue="engine_no", height=6, aspect=6,)
g = g.map(plt.plot, alpha=0.5)
g = g.set(xlim=(df_plot['RUL'].max(),df_plot['RUL'].min()))


# In[28]:


df_eval['pred'] = rf_reg.predict(X_eval)

df_eval['result'] = df_eval['pred']
df_eval['engine_id'] = list(range(len(df_eval)))

df_eval[['engine_id','result']].to_csv('submission.csv', index=False)

