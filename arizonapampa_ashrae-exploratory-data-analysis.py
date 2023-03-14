#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib
sns.set()
print("Setup Complete")


# In[2]:


import pathlib

data_dir = pathlib.Path('/kaggle/input/ashrae-energy-prediction')


# In[3]:


train_data = pd.read_csv(data_dir / 'train.csv', parse_dates=['timestamp']) #, date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))


# In[4]:


meter_meaning = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}


# In[5]:


train_data.dtypes


# In[6]:


train_data.head()


# In[7]:


print(f'Nb of buildings: {len(train_data.building_id.unique())}')


# In[8]:


for meter_val in train_data.meter.unique():
    print(meter_meaning[meter_val])
    print(train_data[train_data.meter == meter_val].meter_reading.describe())
    print()


# In[9]:


nb_building_per_meter = train_data[['building_id', 'meter']].drop_duplicates()                                                             .groupby(by='meter')                                                             .count()                                                             .reset_index()


# In[10]:


fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot([meter_meaning[x] for x in nb_building_per_meter['meter']], 
            nb_building_per_meter['building_id'],
            axes=axes)
axes.set_ylabel('Nb of buildings', fontsize=13)
axes.set_xlabel('')
axes.set_title('Nb of buildings recorded for each type of energy', fontsize=14)


# In[11]:


del nb_building_per_meter


# In[12]:


nb_building_per_building_id = train_data[['building_id', 'meter']].drop_duplicates()                                                                   .groupby(by='building_id')                                                                   .count()                                                                   .reset_index()                                                                   .groupby(by='meter')                                                                   .count()                                                                   .reset_index()


# In[13]:


nb_building_per_building_id


# In[14]:


fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot(nb_building_per_building_id['meter'], 
            nb_building_per_building_id['building_id'],
            ax=axes)
axes.set_ylabel('Nb of buildings', fontsize=13)
axes.set_xlabel('Nb of recorded meters', fontsize=13)
axes.set_title('Nb of buildings vs nb of recorded meters', fontsize=14)


# In[15]:


del nb_building_per_building_id


# In[16]:


nb_meters = train_data['meter'].nunique()
fig, axes = plt.subplots(nb_meters, 1, figsize=(10, 20), dpi=100)
for i, meter in enumerate(train_data['meter'].unique()):
    sns.kdeplot(data=train_data[train_data['meter'] == meter]['meter_reading'].dropna(),
                shade=True,
                label=meter_meaning[meter],
                ax=axes[i%nb_meters])
    axes[i%nb_meters].set_xlabel('')
    axes[i%nb_meters].set_title(f'Distribution of values for {meter}')


# In[17]:


b_metadata = pd.read_csv(data_dir / 'building_metadata.csv')


# In[18]:


b_metadata.dtypes


# In[19]:


b_metadata.head()


# In[20]:


b_metadata.describe()


# In[21]:


print(f'Nb of sites: {b_metadata.site_id.nunique()}')
print(f'Nb of buildings: {b_metadata.building_id.nunique()}')


# In[22]:


buildings_per_site_df = b_metadata.site_id.value_counts()
fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot(x=buildings_per_site_df.index,
            y=buildings_per_site_df.values)
axes.set_title('Nb of buildings per site')
del buildings_per_site_df


# In[23]:


buildings_per_floor_df = b_metadata.floor_count.value_counts()
fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot(x=buildings_per_floor_df.index,
            y=buildings_per_floor_df.values)
axes.set_title('Nb of buildings per floor count')
del buildings_per_floor_df


# In[24]:


buildings_per_year_built_df = b_metadata.year_built.value_counts()
fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.lineplot(x=buildings_per_year_built_df.index,
            y=buildings_per_year_built_df.values)
axes.set_title('Nb of buildings per year built')
del buildings_per_year_built_df


# In[25]:


buildings_per_primary_use_df = b_metadata.primary_use.value_counts()
fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot(x=buildings_per_primary_use_df.index,
            y=buildings_per_primary_use_df.values)
axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
axes.set_title('Nb of buildings per primary use built')
del buildings_per_primary_use_df


# In[26]:


fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.distplot(b_metadata.square_feet,
            hist=False,
            rug=True,
            ax=axes)
axes.set_title('Distribution of building surfaces')


# In[27]:


full_train_df = train_data.merge(b_metadata, 
                                 on='building_id',                                 
                                 how='inner')


# In[28]:


full_train_df.shape


# In[29]:


del train_data, b_metadata


# In[30]:


full_train_df.head()


# In[31]:


nb_sites = full_train_df.site_id.nunique()


# In[32]:


nb_meters = full_train_df.meter.nunique()


# In[33]:


mean_per_site_id = full_train_df.groupby(['site_id', 'meter'])['meter_reading'].mean().reset_index()

fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot(data=mean_per_site_id,
            x='site_id', 
            y='meter_reading',
            hue='meter')
axes.set_title('Mean meter reading per site id')
plt.show()
del mean_per_site_id


# In[34]:


mean_per_building_id = full_train_df[full_train_df.site_id == 13][['building_id', 'meter', 'meter_reading']]                                     .groupby(['building_id', 'meter'])                                     .mean()                                     .reset_index()


# In[35]:


mean_per_building_id.groupby(['building_id', 'meter'])['meter_reading'].max().sort_values(ascending=False).head(10)


# In[36]:


del mean_per_building_id


# In[37]:


mean_per_site_id_without_1099 = full_train_df[full_train_df.building_id != 1099].groupby(['site_id', 'meter'])['meter_reading'].mean().reset_index()
mean_per_site_id_without_1099['meter'] = mean_per_site_id_without_1099['meter'].map(lambda x: meter_meaning[x])

fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot(data=mean_per_site_id_without_1099,
            x='site_id', 
            y='meter_reading',
            hue='meter')
axes.set_title('Mean meter reading per site id after removing building 1099')
plt.show()
del mean_per_site_id_without_1099


# In[38]:


mean_per_primary_use = full_train_df.groupby(['primary_use', 'meter'])['meter_reading'].mean().reset_index()
mean_per_primary_use['meter'] = mean_per_primary_use['meter'].map(lambda x: meter_meaning[x])

fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
sns.barplot(data=mean_per_primary_use,
            x='primary_use', 
            y='meter_reading',
            hue='meter')
axes.set_title('Mean meter reading per primary_use')
axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
plt.show()
del mean_per_primary_use


# In[39]:


fig, axes = plt.subplots(nb_sites, nb_meters, figsize=(15, 60), dpi=100)
for i,site in enumerate(full_train_df.site_id.unique()):
    for j in range(0, nb_meters):
        sns.kdeplot(full_train_df[(full_train_df.site_id==site) & (full_train_df.meter==j)].meter_reading,
                    shade=True,
                    ax=axes[i][j])
        axes[i][j].set_title(f'{meter_meaning[j]} record for site {site}')
fig.tight_layout()
plt.show()


# In[ ]:




