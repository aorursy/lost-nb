#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import hvplot.pandas
import geoviews as gv
import holoviews as hv
hv.extension('bokeh')


# In[2]:


train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv', 
                    index_col='RowId')
test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv', 
                    index_col='RowId')
target_cols = [col for col in train.columns.tolist() if
                    col not in test.columns.tolist()]
target_df = train[(target_cols)]
train = train.drop(target_cols, axis=1)


# In[3]:


tt = pd.concat([train,test], keys=['train', 'test'], sort=False)

if tt.columns[-1:][0] == 'City': #Move city column to where it belongs
    ttcols = tt.columns.tolist() 
    ttcols_moved = ttcols[-1:] + ttcols[:-1] 
    tt = tt[ttcols_moved].reset_index(level=0)
display(tt.head(), target_df.head())
#del train, test


# In[4]:


# Optional function to change column name format from CamelCase to snake_case
def snakify(camel_list):
    snake_list = []
    for c in camel_list:
        underscored = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', c)
        refined = re.sub('([a-z0-9])([A-Z])', r'\1_\2', underscored).lower()
        snake_list.append(refined)
    return snake_list

# tt.columns = snakify(tt.columns)


# In[5]:


tt_bos = tt[tt.City == 'Boston'].drop_duplicates(['level_0','IntersectionId'])

points_bos = gv.Points(tt_bos, kdims=['Longitude', 'Latitude'],
                      vdims=['level_0']).opts(color='level_0', cmap=['dodgerblue', 
                      'darkorange'], width=500, height=450, alpha=0.5)

# points_bos_train = gv.Points(tt_bos[tt_bos.level_0=='train'], kdims=['Longitude', 'Latitude'], 
#                       vdims=['level_0']).opts(color='dodgerblue', width=500, height=450, 
#                       fill_alpha=0.1, line_width=1.5, size=3)

# points_bos_test = gv.Points(tt_bos[tt_bos.level_0=='test'], kdims=['Longitude', 'Latitude'], 
#                       vdims=['level_0']).opts(color='darkorange', width=500, height=450, 
#                       line_alpha=0.1, size=3)

tiles = gv.tile_sources.CartoLight()
display(points_bos * tiles)


# In[6]:


tt_phi = tt[tt.City == 'Philadelphia'].drop_duplicates(['level_0','IntersectionId'])
points_phi = gv.Points(tt_phi, kdims=['Longitude', 'Latitude'],
                      vdims=['level_0']).opts(color='level_0', cmap=['dodgerblue', 
                      'darkorange'], width=500, height=450, alpha=0.5)
tiles = gv.tile_sources.CartoLight()
display(tiles*points_phi)


# In[7]:


index_cols = ['EntryStreetName', 'ExitStreetName', 'EntryHeading', 'ExitHeading']
value_cols = ['TimeFromFirstStop_p' + str(i) for i in [20, 40, 50, 60 , 80]]

def get_times(city, iid, pathlist):
    intersect = train[(train.City == city) & (train.IntersectionId == iid)]
    targets = target_df.loc[intersect.index, :]
    intersect = intersect.join(targets)
    paths = intersect.groupby(index_cols)[value_cols].agg(['mean', 'std']).fillna(0)
    paths['Path'] = pathlist
    paths.columns = paths.columns.swaplevel()
    return paths.sort_index(axis=1)

pathlist_bos = ['E_left', 'E_right', 'NE_left', 'SW_right', 'NE_thru', 'SW_u', 'SW_thru']
bos = get_times('Boston', 2, pathlist_bos)

pathlist_phi = ['N_thru', 'N_right', 'E_left', 'E_thru']
phi = get_times('Philadelphia', 1824, pathlist_phi)

display(bos, phi)


# In[8]:


import hvplot.pandas
opts = {'invert_yaxis': False,
        'yticks': list(range(0,100,20)),
        'padding': 0.1,
        'width':450,
        'height': 300,
           }

# df = bos
# aggfunc='mean'
def make_plot(df, aggfunc):
    assert (aggfunc == 'mean') | (aggfunc == 'std')
    paths = df.set_index(('', 'Path')).loc[:, aggfunc].reset_index()
    paths.columns = [paths.columns[0][1]] + [c[-4:] for c in paths.columns[1:]]
    plot = hvplot.parallel_coordinates(paths, 'Path', **opts)
    if aggfunc == 'mean':
        return plot.options(ylabel='Mean Wait Time')
    else:
        return plot.options(ylabel='STD of Wait Times', show_legend=False)

land_cambridge = make_plot(bos, 'mean').options(title="Land & Cambridgeside") +    make_plot(bos, 'std')
fifth_cambria = make_plot(phi, 'mean').options(title="5th & Cambria") +    make_plot(phi, 'std')

display(land_cambridge, fifth_cambria)


# In[9]:


opts = {'cmap': 'Paired',
        'yticks': list(range(0,300,50)),
        'colorbar': False,
       'grid': True,
         }

land_ne = tt[(tt.IntersectionId == 2) &
             (tt.EntryStreetName == 'Land Boulevard') &
             (tt.ExitHeading == 'NE')
             ].join(target_df)
landplot = land_ne.hvplot.scatter('Hour', 'TimeFromFirstStop_p80', 
                                    c='Weekend', **opts)

cambria_e = tt[(tt.IntersectionId == 1824) &
             (tt.EntryStreetName == 'West Cambria Street') &
             (tt.ExitHeading == 'E')
             ].join(target_df)
cambplot = cambria_e.hvplot.scatter('Hour', 'TimeFromFirstStop_p80', 
                                    c='Weekend', **opts)

display(landplot.options(title='Land_NE_thru'), cambplot.options(title='Cambria_E_thru'))


# In[10]:


# alternate plot with seaborn to trigger viz output on the Notebooks page
import matplotlib.pyplot as plt
import seaborn as sns
cambria_e.plot(kind='scatter', x='Hour', y='TimeFromFirstStop_p20')

