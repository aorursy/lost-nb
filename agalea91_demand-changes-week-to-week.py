#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


types = {'Semana':np.uint8, 'Cliente_ID':np.uint32,
         'Producto_ID':np.uint16, 'Demanda_uni_equil':np.uint32}

df = pd.read_csv('../input/train.csv', usecols=types.keys(),
                 dtype=types)


# In[3]:


# Histogram of target variable
df.Demanda_uni_equil.hist(bins=100)
plt.xlabel('Demand per week')
plt.ylabel('Number of clients');


# In[4]:


# Log scale histogram of target variable
df.Demanda_uni_equil.hist(bins=100, log=True)
plt.xlabel('Demand per week')
plt.ylabel('Number of clients');


# In[5]:


demand_sorted = df.Demanda_uni_equil.sort_values(ascending=True)


# In[6]:


print('plotting {0:.2f} % of data'.      format(100*(demand_sorted < 30).sum()/len(demand_sorted)))
demand_sorted[demand_sorted < 30].hist(bins=30)
plt.xlabel('Demand per week')
plt.ylabel('Number of clients');


# In[7]:


def demandVar(c_id, df, percent=False):
    ''' Get the amounts by which product demand changed
    week-to-week for a given set of client ids. Returned
    object is a pandas dataframe with NaN entries where
    the product was not ordered the week before or after.
    '''
    
    for week in range(4,10):
        try:
            
            vals_a = df[(df.Cliente_ID.values == c_id)&                        (df.Semana.values == week-1)].Demanda_uni_equil.values
            prod_a = df[(df.Cliente_ID.values == c_id)&                        (df.Semana.values == week-1)].Producto_ID.values
            dict_a = {p: v for p, v in zip(prod_a, vals_a)}
            
            vals_b = df[(df.Cliente_ID.values == c_id)&                        (df.Semana.values == (week))].Demanda_uni_equil.values
            prod_b = df[(df.Cliente_ID.values == c_id)&                        (df.Semana.values == week)].Producto_ID.values
            dict_b = {p: v for p, v in zip(prod_b, vals_b)}
            
            dict_merge = {}
            for key in np.unique(np.concatenate((prod_a, prod_b))):
                try:
                    if percent:
                        try:
                            # Calculate percent difference
                            dict_merge[key] = (dict_b[key].astype(int) - dict_a[key].astype(int))                                              /(dict_b[key].astype(int) + dict_a[key].astype(int))
                        except:
                            # If dividing by zero assign 0% change
                            dict_merge[key] = 0.0
                    else:
                        dict_merge[key] = dict_b[key].astype(int) - dict_a[key].astype(int)
                except:
                    # The product was not on the previous form
                    # or was removed from the current one
                    dict_merge[key] = np.nan
            
            if week==4:
                df_return = pd.DataFrame({'week_3-4': list(dict_merge.values())},
                                         index=list(dict_merge.keys()))
            else:
                df_new = pd.DataFrame({'week_'+str(week-1)+'-'+str(week): list(dict_merge.values())}, 
                                       index=list(dict_merge.keys()))
                df_return = pd.merge(df_return, df_new, how='outer', left_index=True, right_index=True)
                
        except:
            print('No week {}-{} data found'.                 format(week-1, week))

    return df_return


# In[8]:


c_ids = [df.Cliente_ID.values[int(i)] for i in np.linspace(0, len(df)-1, 5)]
c_ids


# In[9]:


var = demandVar(c_id=c_ids[0], df=df)
var


# In[10]:


var = demandVar(c_id=c_ids[1], df=df, percent=True)
var


# In[11]:


def get_vars(c_ids, percent=False):
    ''' Return a list of variations in the demand
    week-to-week on individual products for a set of clients. '''
    return_list = [[] for _ in range(len(c_ids))]
    for i, c_id in enumerate(c_ids):
        var = demandVar(c_id, df, percent)
        for col in var.columns:
            return_list[i] += list(var[col].dropna())
        
    return return_list


# In[12]:


var_list = get_vars(c_ids)


# In[13]:


colors = ['blue', 'red', 'green', 'turquoise', 'brown']

fig, ax = plt.subplots(1,2)
plt.suptitle('Change in demand on individual poducts for 5 clients')

for i in range(5):
    ax[0].hist(var_list[i], color=colors[i],
             normed=True, alpha=0.5, bins=20)
ax[0].set_ylim(0,0.2)
ax[0].set_xlabel('Change in demand')
ax[0].set_ylabel('Normed frequency')

for i in range(5):
    ax[1].hist(var_list[i], color=colors[i],
             normed=True, alpha=0.5, bins=20)
ax[1].set_ylim(0,1)


# In[14]:


var_list = get_vars(c_ids, percent=True)


# In[15]:


colors = ['blue', 'red', 'green', 'turquoise', 'brown']

fig, ax = plt.subplots(1,2)
plt.suptitle('Percent change in demand on individual poducts for 5 clients')

for i in range(5):
    ax[0].hist(var_list[i], color=colors[i],
             normed=True, alpha=0.5, bins=20)
ax[0].set_ylim(0,2)
ax[0].set_xlabel('Change in demand')
ax[0].set_ylabel('Normed frequency')

for i in range(5):
    ax[1].hist(var_list[i], color=colors[i],
               normed=True, alpha=0.5, bins=20)


# In[16]:


get_ipython().run_cell_magic('time', '', 'c_ids = [df.Cliente_ID.values[int(i)] for i in np.linspace(0, len(df)-1, 100)]\nvar_list, p_var_list = get_vars(c_ids), get_vars(c_ids, percent=True)')


# In[17]:


fig, ax = plt.subplots(1, 2)
plt.suptitle('Change in demand on individual poducts for 100 clients')

# Plot historgram for flattened list
ax[0].hist([x for row in var_list for x in row], color='red',
            normed=True, alpha=0.5, bins=100)
ax[0].set_xlim(-100,100)
ax[0].set_xlabel('Change in demand')
ax[0].set_ylabel('Normed frequency')

ax[1].hist([x for row in var_list for x in row], color='red',
            normed=True, alpha=0.5, bins=500)
ax[1].set_xlim(-10,10);


# In[18]:


fig, ax = plt.subplots()
plt.suptitle('Percent change in demand on individual poducts for 100 clients')

# Plot historgram for flattened list
ax.hist([x for row in p_var_list for x in row], color='red',
            normed=True, alpha=0.5, bins=100)
# ax[0].set_xlim(-100,100)
ax.set_xlabel('Change in demand')
ax.set_ylabel('Normed frequency');


# In[19]:




