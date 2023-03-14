#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from IPython.display import HTML 
import math
import datetime as dt
import gc
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('expand_frame_repr', True)
np.random.seed(2019)
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#from sklearn.linear_model import BayesianRidge
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
#to hide the code behind a toggle
#credit to $\href{https://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer}{harshil}$
#HTML('''<script>
#code_show=true; 
#function code_toggle() {
# if (code_show){
# $('div.input').hide();
# } else {
# $('div.input').show();
# }
# code_show = !code_show
#} 
#$( document ).ready(code_toggle);
#</script>
#<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')#




# Some miscellaneous functions
#Make size readable
def GetHumanReadable(size,precision=2):
    suffixes=['B','KB','MB','GB','TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1 #increment the index of the suffix
        size = size/1024.0 #apply the division
    return "%.*f%s"%(precision,size,suffixes[suffixIndex])




#Import*

df_metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
df_submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')
df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
df_train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
df_weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
df_weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
data_dict = {'df_metadata': df_metadata, 'df_submission': df_submission
             , 'df_test':df_test, 'df_train':df_train, 'df_weather_test':df_weather_test
             , 'df_weather_train':df_weather_train}
print('Imported', len(data_dict), 'files:')
print('Memory usage:')
for i in data_dict:
    print(i,':', GetHumanReadable(sys.getsizeof(data_dict[i])))
meter_type = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
meter_t = {'electricity':0, 'chilledwater':1, 'steam':2, 'hotwater':3}
meter_name = ['electricity', 'chilledwater', 'steam', 'hotwater']
primary_type = {'Education':0,'Entertainment/public assembly':1,'Food sales and service':2,'Healthcare':3,'Lodging/residential':4,'Manufacturing/industrial':5,'Office':6,'Other':7,'Parking':8,'Public services':9,'Religious worship':10,'Retail':11,'Services':12,'Technology/science':13,'Utility':14,'Warehouse/storage':15}




for i in data_dict:
    print(i,':', data_dict[i].shape[0],'rows' ,data_dict[i].shape[1],'columns')
    if data_dict[i].isnull().any().any():
        temp = data_dict[i].isnull().sum()
        print('Missing data :')
        print('Identifier  -  count')
        print(temp[temp>0])
    else:
        print('No missing data.')
    print('Preview : ')
    print(data_dict[i].head(3))
    print('---------------------------------------------------------------------------------------------------')
del temp




f,ax = plt.subplots(3,2, figsize = (12,18))
sns.countplot(x=df_metadata.site_id,ax= ax[0,0])
ax[0,0].set_title('Building frequency by site')
sns.countplot(y=df_metadata.primary_use,linewidth=5,ax= ax[1,0])
ax[1,0].set_title('Building frequency by usage')

sns.distplot(df_metadata.square_feet, kde = True, rug = True, ax = ax[0,1])
ax[0,1].set_title('Building size distribution')
sns.distplot(df_metadata.square_feet.apply(lambda x : math.log(x)), kde = True, rug = True, ax = ax[1,1])
ax[1,1].set_title('Building log size distribution')
sns.distplot(df_metadata.year_built.dropna(), kde = False, rug = True, bins = 25, ax = ax[2,1])
ax[2,1].set_title('Building frequency by year built')
sns.countplot(y = df_metadata.floor_count, ax =ax[2,0])
ax[2,0].set_title('Building frequency by floor count')
print('Done')




f,a = plt.subplots(1,3, figsize=(14,4))
temp1 = df_metadata['primary_use'].value_counts(normalize = True)
print('-The top 5 building usage take', round(sum(temp1[0:5])*100,2), '% of the total building usage.')
sns.barplot(x = ['Top 5 usage','Rest'], y = [sum(temp1[0:5]),sum(temp1[5:])], ax = a[0])
a[0].set_title('insert title')

del temp1

print('-The buildings with size less than 100K square feet account for' ,round(df_metadata[df_metadata.square_feet<100000].square_feet.count()*100/df_metadata.shape[0],2),'% of the total.')
sns.barplot(x = ['<100K', '100-400K', '400-600K', '>600K']
                 , y = [df_metadata[df_metadata.square_feet<100000].square_feet.count()/df_metadata.shape[0]
                        ,df_metadata[(df_metadata.square_feet>100000) & (df_metadata.square_feet<400000)].square_feet.count()/df_metadata.shape[0]
                        ,df_metadata[(df_metadata.square_feet>400000) & (df_metadata.square_feet<600000)].square_feet.count()/df_metadata.shape[0]
                        ,df_metadata[(df_metadata.square_feet>600000)].square_feet.count()/df_metadata.shape[0]], ax = a[1])
a[1].set_title('')

temp1 = df_metadata['floor_count'].value_counts(normalize = True).reset_index().floor_count
print('-The buildings with less than 5 floors account for', round(sum(temp1[0:5])*100,2), '% of the total.')
sns.barplot(x = ['<5 floor','>5 floor'], y = [sum(temp1[0:5]),sum(temp1[5:])], ax = a[2])
a[2].set_title('')

del temp1

print('')




Map_freq = pd.DataFrame(index = df_metadata.site_id.unique(),columns=df_metadata.primary_use.unique())

for (num,y) in enumerate(df_metadata.site_id.unique()):
    t = []
    for x in df_metadata.primary_use.unique():
        t.append(df_metadata[(df_metadata.site_id==y) & (df_metadata.primary_use==x)].count(0).values[0])
    Map_freq.iloc[num] = t
Map_freq = Map_freq[Map_freq.columns].astype(float)

a1 = plt.figure(figsize = (8,8))
a1 = sns.heatmap(Map_freq.transpose(),square=False, cmap = "gist_gray_r", annot = True)
a1.set_title('Building frequency by use and site')
#a1.set_xticklabels(a1.get_xticklabels(),rotation = 45)
del Map_freq




a1 = plt.figure()
a1 = sns.jointplot(data = df_metadata, y = 'square_feet', x = 'site_id', kind='kde')
a1.fig.suptitle('Kdeplot size/site_id')
a2 = plt.figure()
a2 = sns.jointplot(x = df_metadata.site_id, y = df_metadata.square_feet.apply(lambda x : math.log(x)), kind='kde')
a2.set_axis_labels(ylabel='log square feet', xlabel='site_id')
a2.fig.suptitle('Kdeplot logsize/site_id')
temp = df_metadata.groupby('site_id').sum().square_feet
f, a3 = plt.subplots(2,2, figsize = (12,12))
sns.barplot(y = temp.values, x = temp.index, ax = a3[0,0])
a3[0,0].set_title('Total size by site id')
sns.countplot(x = df_metadata.site_id, ax = a3[0,1])
a3[0,1].set_title('In comparison, building number by site id')
sns.violinplot(data = df_metadata, x = 'site_id', y = 'square_feet', ax = a3[1,0])
a3[1,0].set_title('Violinplot')
sns.boxplot(data = df_metadata, x = 'site_id', y = 'square_feet', ax = a3[1,1])
a3[1,1].set_title('Boxplot')
print('')




a1 = plt.figure()
a1 = sns.jointplot(x = 'site_id', y = 'year_built', data = df_metadata, color='k')
a1.fig.suptitle('Joint plot site/year built')
print('')




a1 = plt.figure()
a1 = sns.catplot(x = 'site_id', y = 'floor_count', data = df_metadata)
a1.fig.suptitle('Catplot floor count/site_id')
print('')




a1 = plt.figure()
a1 = sns.boxplot(data = df_metadata, y = 'primary_use', x = 'square_feet')
a1.set_title('Boxplot usage/size')
print('')




a1 = plt.figure()
a1 = sns.boxplot(data = df_metadata, x = 'year_built', y = 'primary_use')
a1.set_title('Boxplot usage/year built')
print('')




Map_freq = pd.DataFrame(index = df_metadata.primary_use.unique(),columns=df_metadata.floor_count.sort_values(na_position='first').unique())
n = df_metadata.shape[0]
for (num,y) in enumerate(df_metadata.primary_use.unique()):
    t = []
    t.append(df_metadata.groupby('primary_use').get_group(y).floor_count.isnull().sum())
    for x in df_metadata.floor_count.sort_values(na_position = 'first').unique()[1:]:
        t.append(df_metadata[(df_metadata.primary_use==y) & (df_metadata.floor_count==x)].count(0).values[0])
    Map_freq.iloc[num] = t
Map_freq = Map_freq[Map_freq.columns].astype(float)

a1 = plt.figure( figsize = (18,8))
a1 = sns.heatmap(Map_freq,square=False, cmap = "gist_gray_r", annot=True)
a1.set_title('Heat map floor/usage')
#a1.set_xticklabels(a1.get_xticklabels(),rotation = 45)




a1 = plt.figure()
a1 = (sns.jointplot( data = df_metadata, y = 'square_feet', x = 'year_built', kind='kde',color ="k").plot_joint(sns.scatterplot, color="r", size = 0.01, legend = False))
a1.fig.suptitle('Jointplot size/year built')




a1 = plt.figure()
a1 = sns.jointplot(y = 'square_feet', x = 'floor_count', data = df_metadata, color='k')
a1.fig.suptitle('Jointplot size/floor')




a1 = plt.figure()
a1 = sns.jointplot(x = 'year_built', y = 'floor_count', data = df_metadata, color = 'k')
a1.fig.suptitle('Joint plot floor/year built')




#def function
def reduce_mem_usage(df):
    start_mem = df.memory_usage(deep = True).sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)                    
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df




print('Reducing memory usage of \'df_train\' ')
df_train = reduce_mem_usage(df_train)
print('Reducing memory usage of \'df_metadata\' ')
df_metadata = reduce_mem_usage(df_metadata)
print('Reducing memory usage of \'df_weather_train\' ')
df_weather_train = reduce_mem_usage(df_weather_train)
print('Reducing memory usage of \'df_test\' ')
df_test = reduce_mem_usage(df_test)
print('Reducing memory usage of \'df_weather_test\' ')
df_weather_test = reduce_mem_usage(df_weather_test)




df_train = df_train.join(df_metadata.set_index('building_id'), on = 'building_id')
df_train['timestamp'] = df_train['timestamp'].astype(str) 
df_train.timestamp = pd.to_datetime(df_train.timestamp)
df_train.meter = df_train.meter.astype('category')
df_train.meter.cat.rename_categories(meter_type, inplace = True)
df_weather_train['timestamp'] = df_weather_train['timestamp'].astype(str) 
df_weather_train.timestamp = pd.to_datetime(df_weather_train.timestamp)
#df_train = pd.merge(df_weather_train, df_train, on = ['site_id', 'timestamp'])
print('df_train joined with df_metadata and df_weather_train. Resulting dataframe has a memory usage of :', GetHumanReadable(df_train.memory_usage().sum()))
df_test = df_test.join(df_metadata.set_index('building_id'), on = 'building_id')
df_test['timestamp'] = df_test['timestamp'].astype(str) 
df_test.timestamp = pd.to_datetime(df_test.timestamp)
df_test.meter = df_test.meter.astype('category')
df_test.meter.cat.rename_categories(meter_type, inplace = True)
df_weather_test['timestamp'] = df_weather_test['timestamp'].astype(str) 
df_weather_test.timestamp = pd.to_datetime(df_weather_test.timestamp)
#df_test = pd.merge(df_weather_test, df_test, on = ['site_id', 'timestamp'])
print('df_test joined with df_metadata and df_weather_test. Resulting dataframe has a memory usage of :', GetHumanReadable(df_test.memory_usage().sum()))




print('Train DataFrame:')
df_train.head()




f, ax1 = plt.subplots(2,1,figsize = (10,8))
sns.countplot(df_train.building_id, ax =ax1[0], saturation = 1)
ax1[0].set_title('Building reference count in training set')
ax1[0].set(xticks=[])
sns.countplot(df_test.building_id, ax = ax1[1], saturation = 1)
ax1[1].set_title('Building reference count in testing set')
ax1[1].set(xticks=[])
print('')




Meter_t = []
for building_id in range(1449):
    Meter_t.append(df_train[df_train.building_id == building_id].meter.unique())
df_metadata['electricity'] = [('electricity' in Meter_t[building_id]) for building_id in range(1449)]
df_metadata['chilledwater'] = [('chilledwater' in Meter_t[building_id]) for building_id in range(1449)]
df_metadata['steam'] = [('steam' in Meter_t[building_id]) for building_id in range(1449)]
df_metadata['hotwater'] = [('hotwater' in Meter_t[building_id]) for building_id in range(1449)]
del Meter_t
print('Done')




f, ax = plt.subplots(1,2, figsize = (14,6))
sns.countplot(x = df_metadata[['electricity', 'chilledwater', 'steam', 'hotwater']].sum(1), ax = ax[0])
ax[0].set_title('Buildings by number of equipped type of energy')
ax[0].set_xlabel('Number equipped')
sns.barplot(data = df_metadata[['electricity', 'chilledwater', 'steam', 'hotwater']].sum(0).reset_index(), x = 'index', y=0, ax =ax[1])
ax[1].set_title('Meter by number of buildings')
ax[1].set_xlabel('count')
ax[1].set_ylabel('meter type')
print('Done')




Total = df_metadata[['electricity', 'chilledwater', 'steam', 'hotwater']].sum().sum()
D = df_train.append(df_test).timestamp.value_counts(sort = False).apply(lambda x: Total-x)
D = D.reset_index()
D['year']= D['index'].dt.year
S1 = df_metadata[['electricity', 'chilledwater', 'steam', 'hotwater']].sum(1)


D2 = df_train.building_id.value_counts(sort = False).reset_index()
D2['count'] = D2.apply(lambda row : (S1[row['index']]*(365*24+24) - row['building_id'])/(S1[row['index']]*(365*24+24)),1)
D2['origin'] = 'train'
D3 = df_test.building_id.value_counts(sort = False).reset_index()
D3['count'] = D3.apply(lambda row : (S1[row['index']]*(364*24*2+24*2) - row['building_id'])/(S1[row['index']]*(364*24*2+24*2)),1)
D3['origin'] = 'test'

df_metadata['missing_train'] = D2['count']
D2 = D2.append(D3)
del D3




f,ax = plt.subplots(2,1,figsize = (16,16))
sns.lineplot(data = D, y = 'timestamp', x = 'index', hue = 'year',ax= ax[0])
ax[0].set_title('Number of missing timestamp record in train/test data (does not include timestamp with no data at all)')
ax[0].set_ylabel('Count')
ax[0].set_xlabel('Timestamp')
sns.lineplot(data = D2, y = 'count', x = 'index', hue = 'origin',ax= ax[1])
ax[1].set_title('Proportion of missing timestamp record in train/test data by building_id')
ax[1].set_ylabel('Count')
ax[1].set_xlabel('Building_id')
print('Done')

del D, D2




f, ax = plt.subplots(1,1, figsize = (16,6))
sns.scatterplot(x= df_metadata.building_id, y= df_metadata.electricity*1 + ~df_metadata.electricity*(-1), ax =ax, label='electricity')
sns.scatterplot(x= df_metadata.building_id, y= df_metadata.chilledwater*0.8+ ~df_metadata.chilledwater*(-0.8), ax =ax, label='chilledwater')
sns.scatterplot(x= df_metadata.building_id, y= df_metadata.steam*0.6+ ~df_metadata.steam*(-0.6), ax =ax, label='steam')
sns.scatterplot(x= df_metadata.building_id, y= df_metadata.hotwater*0.4+ ~df_metadata.hotwater*(-0.4), ax =ax, label='hotwater')
ax.set_ylabel('')
ax.set_xlabel('building_id')
ax.set_title('Presence(positive) or absence (negative) of meter type by building_id')
ax.legend()
print('Done')




S = df_metadata.groupby('site_id').sum()[['electricity', 'chilledwater', 'steam', 'hotwater']]
SS = df_metadata.groupby('primary_use').sum()[['electricity', 'chilledwater', 'steam', 'hotwater']]
SS = SS.reset_index()
#SS['primary_use'].cat.rename_categories(primary_type, inplace=True)
#SS.primary_use = SS.primary_use.astype(int)
SS = pd.melt(SS, id_vars = ['primary_use'], value_vars = ['electricity', 'chilledwater', 'steam', 'hotwater'], var_name='meter', value_name = 'count')
#SS.primary_use = SS.primary_use.astype('category')
#SS['primary_use']= SS.primary_use.cat.rename_categories({key:values for (key, values) in zip(primary_type.values(),primary_type.keys())})
S = S.reset_index()
S = pd.melt(S, id_vars = 'site_id', value_vars = ['electricity', 'chilledwater', 'steam', 'hotwater'], var_name='meter', value_name = 'count')
f, ax = plt.subplots(2,1,figsize = (18,12))
sns.barplot(data = S, x ='site_id', hue = 'meter', y = 'count', ax= ax[0])
ax[0].set_title('Meter type count by site_id')
sns.barplot(data= SS, x= 'primary_use', y='count', hue='meter', ax=ax[1])
ax[1].set_xticklabels(
    ax[1].get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'

)
ax[1].set_title('Meter type count by primary use')
del S, SS




TEMP = pd.melt(df_metadata, id_vars= 'building_id', value_vars=['electricity', 'chilledwater', 'steam', 'hotwater']).join(df_metadata[['building_id', 'square_feet']].set_index('building_id'), on = 'building_id')
f,ax = plt.subplots(2,2, figsize = (18,18))
sns.scatterplot(data = TEMP.loc[(TEMP['variable']=='electricity') & (TEMP['value']== True)], x = 'building_id', y = 'square_feet', ax= ax[0,0])
ax[0,0].set_title('Building with electricity')
sns.scatterplot(data = TEMP.loc[(TEMP['variable']=='chilledwater') & (TEMP['value']== True)], x = 'building_id', y = 'square_feet', ax= ax[0,1])
ax[0,1].set_title('Building with chilledwater')
sns.scatterplot(data = TEMP.loc[(TEMP['variable']=='steam') & (TEMP['value']== True)], x = 'building_id', y = 'square_feet', ax= ax[1,0])
ax[1,0].set_title('Building with steam')
sns.scatterplot(data = TEMP.loc[(TEMP['variable']=='hotwater') & (TEMP['value']== True)], x = 'building_id', y = 'square_feet', ax= ax[1,1])
ax[1,1].set_title('Building with hotwater')




def print_series(building_id, meter, ax1):
    
    if building_id in df_metadata.building_id:
        if meter in meter_name:
            if meter == 'electricity':
                typein = (df_metadata.iloc[building_id].electricity >0)
            elif meter == 'chilledwater':
                typein = (df_metadata.iloc[building_id].chilledwater >0)
            elif meter == 'steam':
                typein = (df_metadata.iloc[building_id].steam >0)
            else:
                typein = (df_metadata.iloc[building_id].hotwater >0)
            if typein:
                df_temp = df_train[(df_train.building_id == building_id) & (df_train.meter == meter)][['timestamp', 'meter_reading']].reset_index()
                sns.lineplot(data = df_temp, x = 'timestamp', y = 'meter_reading', ax=ax1)
                ax1.set_title('Building ID: '+str(building_id)+ ' : '+ str(meter))
                return ax1
            else:
                print('Error : Building not equipped with this meter type.')
        else:
            print('Error : meter type not recognized')
    else:
        print('Error : building id not in metadata.')
def prt_srs(building_id):
    print('Display building number' + str(building_id) + '.')
    print(df_metadata.loc[df_metadata['building_id'] == building_id])
    if building_id in df_metadata.building_id:
        meter = []
        if (df_metadata.iloc[building_id].electricity >0):
            meter.append('electricity')
        if (df_metadata.iloc[building_id].chilledwater >0):
            meter.append('chilledwater')
        if (df_metadata.iloc[building_id].steam >0):
            meter.append('steam')
        if (df_metadata.iloc[building_id].hotwater >0):
            meter.append('hotwater')
        if len(meter)>1:
            f, ax1 = plt.subplots(len(meter),1, figsize=(12,len(meter)*5))
            for (ind,m) in enumerate(meter):
                ax1[ind] = print_series(building_id, m, ax1[ind])
        else:
            f, ax1 = plt.subplots(1,1, figsize = (12,5))
            ax1 = print_series(building_id, meter[0], ax1)
    else:
        print('Error : building id not in metadata.')
    print('Done')




prt_srs(0)




prt_srs(200)




prt_srs(1232)




prt_srs(df_metadata.loc[df_metadata.missing_train == df_metadata.missing_train.max()].building_id.values[0])




f, ax = plt.subplots(1,2,figsize = (16,6))
sns.distplot(df_train['meter_reading'], ax =ax[0])
ax[0].set_title('Distribution of meter_reading')
sns.distplot(df_train['meter_reading'].apply(lambda x: np.log1p(x)), ax =ax[1])
ax[1].set_title('Distribution of log(meter_reading)')
ax[1].set_ylabel('log(meter_reading)')
print('')




f,ax = plt.subplots(1,1,figsize = (16,8))
sns.distplot(df_train.loc[df_train.meter=='electricity','meter_reading'].apply(np.log1p), label = 'electricity', color = (1,0,0), hist=False, ax=ax)
sns.distplot(df_train.loc[df_train.meter=='chilledwater','meter_reading'].apply(np.log1p), label = 'chilledwater', hist=False, color = (0.5,0.5,0), ax=ax)
sns.distplot(df_train.loc[df_train.meter=='steam','meter_reading'].apply(np.log1p), label = 'steam', color = (0,0.5,0.5), hist=False, ax=ax)
sns.distplot(df_train.loc[df_train.meter=='hotwater','meter_reading'].apply(np.log1p), label = 'hotwater', color = (0.33,0.33,0.34), hist=False, ax=ax)
ax.set_title('Distribution of log(meter_reading) by meter type')
ax.set_xlabel('log(meter_reading)')
print('')




#Adding time variable (df_train)
df_train['weekday'] = df_train.timestamp.dt.weekday_name.astype('category')
df_train['week'] = df_train.timestamp.dt.week.astype(np.int8)
df_train['hour'] = df_train.timestamp.dt.hour.astype(np.int8)
df_train['dayofmonth'] = df_train.timestamp.dt.day.astype(np.int8)
df_train['month'] = df_train.timestamp.dt.month.astype(np.int8)
df_train['dayofyear'] = df_train.timestamp.dt.dayofyear.astype(np.int16)
#Adding time variable (df_test)
df_test['weekday'] = df_test.timestamp.dt.weekday_name.astype('category')
df_test['week'] = df_test.timestamp.dt.week.astype(np.int8)
df_test['hour'] = df_test.timestamp.dt.hour.astype(np.int8)
df_test['dayofmonth'] = df_test.timestamp.dt.day.astype(np.int8)
df_test['month'] = df_test.timestamp.dt.month.astype(np.int8)
df_test['dayofyear'] = df_test.timestamp.dt.dayofyear.astype(np.int16)




df_maxmonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_minmonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_medianmonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_summonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_maxweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_minweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_medianweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_sumweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_maxhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_minhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_medianhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_sumhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_maxdayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_mindayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_mediandayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_sumdayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')




ax = plt.figure(figsize = (8,8))
ax = sns.lineplot(data = df_maxmonthly, y = 'meter_reading', x = 'month', hue = 'building_id', style = 'meter')
ax.set_title('Monthly max per meter/building_id')




print('Finding building_id')
print(df_maxmonthly.loc[df_maxmonthly.meter_reading == df_maxmonthly.meter_reading.max()])
print('Display series')
prt_srs(df_maxmonthly.loc[df_maxmonthly.meter_reading == df_maxmonthly.meter_reading.max()].building_id.values[0])




df_ex_train =df_train.loc[(df_train.building_id == 1099) & (df_train.meter == 'steam')]
df_ex_test = df_test.loc[(df_test.building_id == 1099) & (df_test.meter == 'steam')]
index_name = df_train.loc[(df_train.building_id == 1099) & (df_train.meter == 'steam')].index
df_train.drop(index_name, inplace = True)
index_name = df_test.loc[(df_test.building_id == 1099) & (df_test.meter == 'steam')].index
df_test.drop(index_name, inplace = True)
del index_name




print('Computing aggregate')
df_maxmonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_minmonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_medianmonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_summonthly = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'month']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_maxweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_minweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_medianweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_sumweekday = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'weekday']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_maxhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_minhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_medianhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_sumhour = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'hour']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_maxdayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).max().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_mindayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).min().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
#df_mediandayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).median().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
df_sumdayofyear = pd.DataFrame(data=df_train.groupby(['building_id', 'meter', 'dayofyear']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on ='building_id')
print('Done')




def print_aggregate( dataframe, groupby, time_type = 'month', graph_type = 'line', building_id = None, primary_use = None,site_id = None, aggregate = 'all',  meter = None, graph_name = '', ax = None):
    time_candidate = {'month', 'dayofyear', 'dayofmonth', 'week', 'weekday', 'hour'}
    groupby_candidate = {'building_id', 'site_id', 'meter', 'primary_use', 'square_feet', 'year_built', 'floor_count'}
    aggregate_candidate = {'all', 'min', 'max', 'sum', 'median'}
    graph_type_candidate = {'line'}
    if (time_type not in time_candidate) or (time_type not in dataframe.columns):
        print('Error : time_type error.')
        return
    if (groupby not in groupby_candidate) or (groupby not in dataframe.columns):
        print('Groupby error.')
        return
    if (aggregate not in aggregate_candidate):
        print('Aggregate type error.')
        return
    if graph_type not in graph_type_candidate:
        print('Graph type error.')
        return
    if groupby == 'building_id':
        if building_id is None:
            if ax is None:
                ax = plt.figure()
                ax = sns.lineplot(data = dataframe, y='meter_reading', x= time_type, hue = 'building_id', style = 'meter')
                ax.set_title(graph_name)
            else:
                sns.lineplot(data = dataframe, y = 'meter_reading', x = time_type, hue='building_id', style = 'meter', ax = ax)
                ax.set_title(graph_name)
        elif building_id in dataframe.building_id.unique():
            if ax is None:
                ax = plt.figure()
                ax = sns.lineplot(data = dataframe[dataframe.building_id == building_id].reset_index(), y = 'meter_reading', x = time_type, hue='meter')
                ax.set_title(graph_name)
            else: 
                sns.lineplot(data = dataframe[dataframe.building_id == building_id].reset_index(), y = 'meter_reading', x = time_type, hue='meter', ax = ax)
                ax.set_title(graph_name)
        else:
            print('Building_id error.')
    if groupby == 'site_id': 
        if aggregate == 'all':
            if (site_id is None) or (site_id not in df_metadata.site_id.unique()):
                print('Use building_id instead.')
            else:
                ax = plt.figure()
                ax = sns.lineplot(data = dataframe[dataframe.site_id== site_id].reset_index, y='meter_reading', x=time_type, hue = 'building_id', style = 'meter')
                ax.set_title(graph_name)
            return
        if aggregate == 'min':
            df_t = pd.DataFrame(dataframe.groupby(['site_id','meter', time_type]).min().meter_reading.dropna()).reset_index()
        if aggregate == 'max':
            df_t = pd.DataFrame(dataframe.groupby(['site_id','meter', time_type]).max().meter_reading.dropna()).reset_index()

        if aggregate == 'sum':
            df_t = pd.DataFrame(dataframe.groupby(['site_id','meter', time_type]).sum().meter_reading.dropna()).reset_index()

        if aggregate == 'median':
            df_t = pd.DataFrame(dataframe.groupby(['site_id','meter', time_type]).median().meter_reading.dropna()).reset_index()
        if ax is None:
            ax = plt.figure()
            ax = sns.lineplot(data=df_t, y = 'meter_reading', x = time_type, style = 'meter', hue = 'site_id')
            ax.set_title(graph_name)
        else:
            sns.lineplot(data=df_t, y = 'meter_reading', x = time_type, style = 'meter', hue = 'site_id',ax= ax)
            ax.set_title(graph_name)
    if groupby == 'meter':
        if aggregate == 'all':
            if (meter is None) or (meter not in ['electricity', 'chilledwater', 'steam', 'hotwater']):
                print('Use building_id instead.')
            else:
                ax = plt.figure()
                ax = sns.lineplot(data = dataframe[dataframe.meter == meter].reset_index(), y='meter_reading', x=time_type, hue='building_id')
                ax.set_title(graph_name)
            return
        if aggregate == 'min':
            df_t = pd.DataFrame(dataframe.groupby(['meter', time_type]).min().meter_reading.dropna()).reset_index()
            
        if aggregate == 'max':
            df_t = pd.DataFrame(dataframe.groupby(['meter', time_type]).max().meter_reading.dropna()).reset_index()
            
        if aggregate == 'sum':
            df_t = pd.DataFrame(dataframe.groupby(['meter', time_type]).sum().meter_reading.dropna()).reset_index()
            
        if aggregate == 'median':
            df_t = pd.DataFrame(dataframe.groupby(['meter', time_type]).median().meter_reading.dropna()).reset_index()
        if ax is None:
            ax = plt.figure()
            ax = sns.lineplot(data=df_t, y = 'meter_reading', x = time_type, hue = 'meter')
            ax.set_title(graph_name)
        else:
            sns.lineplot(data=df_t, y = 'meter_reading', x = time_type, hue = 'meter', ax= ax)
            ax.set_title(graph_name)
    if groupby == 'primary_use':
        if aggregate == 'all':
            if (primary_use is None) or (meter not in df_metadata.primary_use.unique()):
                print('Use building_id instead.')
            else:
                ax = plt.figure()
                ax = sns.lineplot(data = dataframe[dataframe.primary_use == primary_use].reset_index(), y='meter_reading', x=time_type, hue='building_id')
                ax.set_title(graph_name)
            return
        if aggregate == 'min':
            df_t = pd.DataFrame(dataframe.groupby(['primary_use','meter', time_type]).min().meter_reading.dropna()).reset_index()
        if aggregate == 'max':
            df_t = pd.DataFrame(dataframe.groupby(['primary_use','meter', time_type]).max().meter_reading.dropna()).reset_index()

        if aggregate == 'sum':
            df_t = pd.DataFrame(dataframe.groupby(['primary_use','meter', time_type]).sum().meter_reading.dropna()).reset_index()

        if aggregate == 'median':
            df_t = pd.DataFrame(dataframe.groupby(['primary_use','meter', time_type]).median().meter_reading.dropna()).reset_index()
        if ax is None:
            ax = plt.figure(figsize = (12,12))
            ax = sns.lineplot(data=df_t, y = 'meter_reading', x = time_type, style = 'meter', hue = 'primary_use')
            ax.set_title(graph_name)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

            # Put a legend to the right side
            ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        else:
            sns.lineplot(data=df_t, y = 'meter_reading', x = time_type, style = 'meter', hue = 'primary_use',ax= ax)
            ax.set_title(graph_name)

    
    if groupby == 'square_feet':
        df_t = pd.DataFrame(dataframe.groupby(['building_id', 'meter']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on = 'building_id')
        if ax is None:
            ax = plt.figure()
            ax = sns.scatterplot(data = df_t, x = 'square_feet', y = 'meter_reading', hue = 'meter')
            ax.set_title(graph_name)
        else:
            sns.scatterplot(data = df_t, x = 'square_feet', y = 'meter_reading', hue = 'meter', ax = ax)
            ax.set_title(graph_name)
    if groupby == 'year_built':
        df_t = pd.DataFrame(dataframe.groupby(['building_id', 'meter']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on = 'building_id')
        if ax is None:
            ax = plt.figure()
            ax = sns.scatterplot(data = df_t, x = 'year_built', y = 'meter_reading', hue = 'meter')
            ax.set_title(graph_name)
        else:
            sns.scatterplot(data = df_t, x = 'year_built', y = 'meter_reading', hue = 'meter', ax = ax)
            ax.set_title(graph_name)
    if groupby == 'floor_count':
        df_t = pd.DataFrame(dataframe.groupby(['building_id', 'meter']).sum().meter_reading.dropna()).reset_index().join(df_metadata.set_index('building_id'), on = 'building_id')
        if ax is None:
            ax = plt.figure()
            ax = sns.scatterplot(data = df_t, x = 'floor_count', y = 'meter_reading', hue = 'meter')
            ax.set_title(graph_name)
        else:
            sns.scatterplot(data = df_t, x = 'floor_count', y = 'meter_reading', hue = 'meter', ax = ax)
            ax.set_title(graph_name)




f, ax1 = plt.subplots(2,2, figsize=(20,20))
print_aggregate(df_maxmonthly, groupby='building_id',graph_name = 'Max monthly', ax = ax1[0,0])
print_aggregate(df_minmonthly, groupby='building_id',graph_name = 'Min monthly', ax = ax1[0,1])
print_aggregate(df_summonthly, groupby='building_id',graph_name = 'Sum monthly', ax = ax1[1,0])
print_aggregate(df_medianmonthly, groupby='building_id',graph_name = 'Median monthly', ax = ax1[1,1])




f, ax1 = plt.subplots(2,2, figsize=(20,20))
print_aggregate(df_maxweekday,time_type = 'weekday', groupby='building_id',graph_name = 'Max weekday', ax = ax1[0,0])
print_aggregate(df_minweekday,time_type = 'weekday', groupby='building_id',graph_name = 'Min weekday', ax = ax1[0,1])
print_aggregate(df_sumweekday,time_type = 'weekday', groupby='building_id',graph_name = 'Sum weekday', ax = ax1[1,0])
print_aggregate(df_medianweekday,time_type = 'weekday', groupby='building_id',graph_name = 'Median weekday', ax = ax1[1,1])




f, ax1 = plt.subplots(2,2, figsize=(20,20))
print_aggregate(df_maxhour,time_type = 'hour', groupby='building_id',graph_name = 'Max hourly', ax = ax1[0,0])
print_aggregate(df_minhour,time_type = 'hour', groupby='building_id',graph_name = 'Min hourly', ax = ax1[0,1])
print_aggregate(df_sumhour,time_type = 'hour', groupby='building_id',graph_name = 'Sum hourly', ax = ax1[1,0])
print_aggregate(df_medianhour,time_type = 'hour', groupby='building_id',graph_name = 'Median hourly', ax = ax1[1,1])




f, ax1 = plt.subplots(2,2,figsize = (20,20))
print_aggregate(df_summonthly, groupby = 'site_id',aggregate= 'sum',  ax = ax1[0,0], graph_name='Total consumption by site id')
print_aggregate(df_summonthly, groupby = 'floor_count', ax = ax1[0,1], graph_name='Total consumption by floor count')
print_aggregate(df_summonthly, groupby = 'year_built', ax = ax1[1,0], graph_name='Total consumption by year built')
print_aggregate(df_summonthly, groupby = 'square_feet', ax = ax1[1,1], graph_name='Total consumption by square_feet')




f, ax1 = plt.subplots(2,2,figsize = (20,20))
print_aggregate(df_sumweekday,time_type='weekday', groupby = 'meter', aggregate='sum', ax = ax1[0,0], graph_name = 'Total kwh by meter type')
print_aggregate(df_sumhour,time_type='hour', groupby = 'meter', aggregate='sum', ax = ax1[0,1], graph_name = 'Total kwh by meter type')
print_aggregate(df_summonthly, groupby = 'meter', aggregate='sum', ax = ax1[1,0], graph_name = 'Total kwh by meter type')
print_aggregate(df_sumdayofyear, time_type='dayofyear',groupby = 'meter', aggregate='sum', ax = ax1[1,1], graph_name = 'Total kwh by meter type')




#get extreme building_id index
KK = pd.DataFrame(df_summonthly.groupby(['building_id', 'meter']).sum().meter_reading.dropna()).reset_index()
extreme_index = KK.loc[KK.meter_reading>2e8].building_id.values
del KK
print('Without the extremes')
f, ax1 = plt.subplots(2,2,figsize = (20,20))
print_aggregate(df_sumweekday.loc[df_sumweekday.building_id.apply(lambda x: x not in extreme_index)],time_type='weekday', groupby = 'meter', aggregate='sum', ax = ax1[0,0], graph_name = 'Total kwh by meter type')
print_aggregate(df_sumhour.loc[df_sumhour.building_id.apply(lambda x: x not in extreme_index)],time_type='hour', groupby = 'meter', aggregate='sum', ax = ax1[0,1], graph_name = 'Total kwh by meter type')
print_aggregate(df_summonthly.loc[df_summonthly.building_id.apply(lambda x: x not in extreme_index)], groupby = 'meter', aggregate='sum', ax = ax1[1,0], graph_name = 'Total kwh by meter type')
print_aggregate(df_sumdayofyear.loc[df_sumdayofyear.building_id.apply(lambda x: x not in extreme_index)], time_type='dayofyear',groupby = 'meter', aggregate='sum', ax = ax1[1,1], graph_name = 'Total kwh by meter type')
#and remove them from the training data
ett = df_train.loc[(df_train.building_id.apply(lambda x: x in extreme_index)) & (df_train.meter.apply(lambda x: x in ['steam', 'chilledwater']) )].index
df_train.drop(ett)




del df_maxmonthly,df_minmonthly,df_medianmonthly,df_summonthly,df_maxweekday,df_minweekday,df_medianweekday,df_sumweekday,df_maxhour,df_minhour,df_medianhour,df_sumhour
#del df_maxdayofyear
#del df_mindayofyear
#del df_mediandayofyear
del df_sumdayofyear




df_train.isnull().sum()




print('Appending the testing weather data to the training weather data.')
DF_WT_Comb = df_weather_train.append(df_weather_test)
mis_col_n = {'air_temperature','cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed'} 
DF_WT_Comb['year'] = DF_WT_Comb.timestamp.dt.year
DF_WT_Comb['month'] = DF_WT_Comb.timestamp.dt.month
DF_WT_Comb['dayofyear'] = DF_WT_Comb.timestamp.dt.dayofyear
DF_WT_Comb['week'] = DF_WT_Comb.timestamp.dt.week
DF_WT_Comb = DF_WT_Comb.reset_index()
for n in mis_col_n:
    t_name = n+'miss'
    DF_WT_Comb[t_name] = DF_WT_Comb[n].isnull()
print('Done.')




ax = sns.FacetGrid(data = DF_WT_Comb, row = 'site_id', height = 4, aspect = 3)
ax = ax.map(sns.lineplot, 'timestamp',  'air_temperature', label = 'Air temp')
ax.set(ylim = (-30,50))
for axe in range(16):
    ax.axes[axe][0].vlines(x = DF_WT_Comb[(DF_WT_Comb.site_id == axe) & (DF_WT_Comb.air_temperaturemiss == True)].timestamp.values, color=(1,0,0), ymin= -30, ymax = 50, label = 'Missing values')
    ax.axes[axe][0].set_title('Air temperature vs time : site id number '+ str(axe))
    ax.axes[axe][0].legend(loc='best')




print('Ratio of values that are equal between site 7 and site 11:')
print((DF_WT_Comb.loc[DF_WT_Comb.site_id ==7].fillna(99999999).reset_index() == DF_WT_Comb.loc[DF_WT_Comb.site_id ==11].fillna(99999999).reset_index()).sum()/len(DF_WT_Comb.loc[DF_WT_Comb.site_id ==11]))




df_weather_train['air_temperature'].interpolate('cubic', inplace= True)
df_weather_test['air_temperature'].interpolate('cubic', inplace= True)
print(df_weather_test.air_temperature.isnull().sum())
print(df_weather_train.air_temperature.isnull().sum())




#cloud_diff = pd.Series()
#for i in range(16):
#    t = DF_WT_Comb[DF_WT_Comb.site_id == i]['cloud_coverage']
#    cloud_diff = cloud_diff.append(t)
#DF_WT_Comb['cloud_diff'] = cloud_diff.values
#del cloud_diff, t
ax = sns.FacetGrid(data = DF_WT_Comb, row = 'site_id', height = 4, aspect = 3)
ax = ax.map(sns.lineplot,'timestamp','cloud_coverage', label='meter_reading')

ax.set(xlim=(pd.Timestamp('2016-01-01 00:00:00'), pd.Timestamp('2018-12-31 23:00:00')))
ax.set(ylim = (-1,10))
for axe in range(16):
    ax.axes[axe][0].vlines(x = DF_WT_Comb[(DF_WT_Comb.site_id == axe) & (DF_WT_Comb.cloud_coveragemiss == True)].timestamp.values, color='red', ymin= -1, ymax = -0.5, linewidth = 0.1, label = 'missing values')
    ax.axes[axe][0].set_title('Cloud coverage vs time : site id number '+ str(axe))
    ax.axes[axe][0].legend(loc='best')




ax = sns.FacetGrid(data = DF_WT_Comb, col = 'site_id',col_wrap=4)
ax = ax.map(sns.distplot,'cloud_coverage', kde=False)




df_weather_train.drop('cloud_coverage', inplace = True, axis = 1)
df_weather_test.drop('cloud_coverage', inplace = True, axis = 1)




ax = sns.FacetGrid(data = DF_WT_Comb, row = 'site_id', height = 4, aspect = 3)
ax = ax.map(sns.lineplot, 'timestamp',  'dew_temperature', label = 'meter_reading')
ax.set(ylim = (-40,30))
for axe in range(16):
    ax.axes[axe][0].vlines(x = DF_WT_Comb[(DF_WT_Comb.site_id == axe) & (DF_WT_Comb.dew_temperaturemiss == True)].timestamp.values, color='red', ymin= -40, ymax = 30, label = 'missing values')
    ax.axes[axe][0].set_title('Dew temperature vs time : site id number '+ str(axe))
    ax.axes[axe][0].legend(loc='best')




df_weather_train['dew_temperature'].interpolate(method='cubic', inplace=True)
df_weather_test['dew_temperature'].interpolate(method='cubic', inplace=True)
print(df_weather_test.dew_temperature.isnull().sum())
print(df_weather_train.dew_temperature.isnull().sum())




ax = sns.FacetGrid(data = DF_WT_Comb, row = 'site_id', height = 5, aspect = 3)
#DF_WT_Comb['logprec'] = DF_WT_Comb['precip_depth_1_hr'].apply(np.log1p)
ax = ax.map(sns.lineplot, 'timestamp',  'precip_depth_1_hr', label = 'meter_reading')
#ax = ax.map(sns.distplot,  'logprec', label = 'meter_reading', hist=False)


for axe in range(16):
    ax.axes[axe][0].vlines(x = DF_WT_Comb[(DF_WT_Comb.site_id == axe) & (DF_WT_Comb.precip_depth_1_hrmiss == True)].timestamp.values, color='red', ymin= 0, ymax = 600, label = 'missing values')
    ax.axes[axe][0].set_title('Precip vs time : site id number '+ str(axe))
    ax.axes[axe][0].legend(loc='best')




ax = sns.FacetGrid(data = DF_WT_Comb, col = 'site_id', col_wrap = 4)
DF_WT_Comb['logprec'] = DF_WT_Comb['precip_depth_1_hr'].apply(np.log1p)
ax = ax.map(sns.distplot,  'logprec', hist=False)
ax.set(xlabel='log(precipitation)')




df_weather_train.drop('precip_depth_1_hr', inplace=True, axis=1)
df_weather_test.drop('precip_depth_1_hr', inplace=True, axis=1)




ax = sns.FacetGrid(data = DF_WT_Comb, row = 'site_id', height = 4, aspect = 3)
ax = ax.map(sns.lineplot, 'timestamp',  'sea_level_pressure', label='meter_reading')
for axe in range(16):
    ax.axes[axe][0].vlines(x = DF_WT_Comb[(DF_WT_Comb.site_id == axe) & (DF_WT_Comb.sea_level_pressuremiss == True)].timestamp.values, color='red', ymin= 980, ymax = 1040, label = 'missing values')
    ax.axes[axe][0].set_title('Sea level pressure vs time : site id number '+ str(axe))
    ax.axes[axe][0].legend(loc='best')




df_weather_train['sea_level_pressure'] = df_weather_train['sea_level_pressure'].fillna(DF_WT_Comb['sea_level_pressure'].median())
df_weather_test['sea_level_pressure'] = df_weather_test['sea_level_pressure'].fillna(DF_WT_Comb['sea_level_pressure'].median())
print(df_weather_test.sea_level_pressure.isnull().sum())
print(df_weather_train.sea_level_pressure.isnull().sum())




ax = sns.FacetGrid(data = DF_WT_Comb[DF_WT_Comb.year==2016], col = 'site_id', col_wrap=2, hue = 'month' ,subplot_kws=dict(projection='polar'),sharex=False, sharey=False, despine=False, height = 6, aspect = 1)
ax = (ax.map(sns.scatterplot,  'wind_direction','wind_speed').add_legend())
for axe in range(16):
    ax.axes[axe].set_title('Wind direction and speed for year 2016: site '+ str(axe))
ax.set(ylabel='')
ax.set(xlabel='')
print('')




def rose_plot(angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    #credit to https://stackoverflow.com/questions/22562364/circular-histogram-for-python
    #see also https://github.com/msmbuilder/msmexplorer/issues/98 for alternative
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    ax = plt.gca(projection='polar')
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)
    return ax

ax = sns.FacetGrid(data = DF_WT_Comb, col = 'site_id', col_wrap=2,subplot_kws=dict(projection='polar'),sharex=False, sharey=False, despine=False, height = 6, aspect = 1)
ax = ax.map(rose_plot,  'wind_direction', bins = 30)
for axe in range(16):
    ax.axes[axe].set_title('Circular histogram wind direction train+test: site '+ str(axe))
print('')




df_weather_train['wind_direction']=df_weather_train['wind_direction'].astype(float).interpolate(method='pad')
df_weather_test['wind_direction']=df_weather_test['wind_direction'].astype(float).interpolate(method='pad')
print(df_weather_train['wind_direction'].isnull().sum())
print(df_weather_test['wind_direction'].isnull().sum())




ax = sns.FacetGrid(data = DF_WT_Comb, row = 'site_id', height = 4, aspect = 3)
ax = ax.map(sns.lineplot, 'timestamp',  'wind_speed', label='meter_reading')
for axe in range(16):
    ax.axes[axe][0].vlines(x = DF_WT_Comb[(DF_WT_Comb.site_id == axe) & (DF_WT_Comb.wind_speedmiss == True)].timestamp.values, color='red', ymin= 0, ymax = 16, label = 'missing values')
    ax.axes[axe][0].set_title('Wind speed vs time : site id number '+ str(axe))
    ax.axes[axe][0].legend(loc='best')




df_weather_train['wind_speed'].interpolate(method='cubic', inplace= True)
df_weather_test['wind_speed'].interpolate(method='cubic', inplace= True)
print(df_weather_train['wind_speed'].isnull().sum())
print(df_weather_test['wind_speed'].isnull().sum())




#Merge weather and main df
df_train = df_train.merge(df_weather_train, on = ['site_id', 'timestamp'], how='left')
df_test = df_test.merge(df_weather_test, on = ['site_id', 'timestamp'], how='left')
gc.collect()




#Now for year built and floor_count
df_train['year_built'] = df_train['year_built'].astype(float).interpolate(method='pad').astype(np.int16)
print(df_train['year_built'].isnull().sum())
df_test['year_built'] = df_test['year_built'].astype(float).interpolate(method='pad').astype(np.int16)
print(df_test['year_built'].isnull().sum())




df_train['floor_count'] = df_train['floor_count'].fillna(0)
df_test['floor_count'] = df_test['floor_count'].fillna(0)
print(df_train['floor_count'].isnull().sum())
print(df_test['floor_count'].isnull().sum())




#Last conversion
df_train['site_id'] = df_train['site_id'].astype('category')
df_train['building_id'] = df_train['building_id'].astype('category')
df_train['floor_count'] = df_train['floor_count'].astype(np.uint8)
df_train['meter_reading'] = df_train['meter_reading'].apply(np.log1p)
df_train['square_feet'] = df_train['square_feet'].apply(np.log1p)
df_test['site_id'] = df_test['site_id'].astype('category')
df_test['building_id'] = df_test['building_id'].astype('category')
df_test['floor_count'] = df_test['floor_count'].astype(np.uint8) 
df_test['square_feet'] = df_test['square_feet'].apply(np.log1p) #remember to apply expm1
df_train['wind_direction'] = df_train['wind_direction'].astype(np.float16)
df_test['wind_direction'] = df_test['wind_direction'].astype(np.float16)
df_train['square_feet'] = df_train['square_feet'].astype(np.float32)
df_test['square_feet'] = df_test['square_feet'].astype(np.float32)




#remove extremes from training set or maybe not
#ett = df_train.loc[(df_train.building_id.apply(lambda x: x in extreme_index)) & (df_train.meter.apply(lambda x: x in ['steam', 'chilledwater']) )].index
#df_train.drop(ett, inplace= True)

