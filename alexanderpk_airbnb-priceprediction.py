#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Поставим стабильную версию CatBoost
get_ipython().system('pip uninstall catboost --yes')
get_ipython().system('pip install catboost==0.15.2')


# In[2]:


import os
import tqdm
import pickle
import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium 
import warnings

import hyperopt 
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor, DMatrix
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool, cv 


# In[3]:


import shap
shap.initjs()


# In[4]:


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sns.set()


# In[5]:


# Системные константы
EXT_DATA = '/kaggle/input/external-geodata'
PREPROCESS = '/kaggle/input/airbnb-preprocess'
DATA = '/kaggle/input/spsu-intensives'
CWD = os.getcwd()
VAL_SIZE = 0.1
DEVICE = 'CPU'

# Параметры обработчика данных
mmt_dict = {'test': os.path.join(DATA, 'test.csv'),
            'train': os.path.join(DATA, 'train.csv'),
            'districts': os.path.join(EXT_DATA, 'districts.csv'),
            'parks': os.path.join(EXT_DATA, 'parks.csv'),
            'sights': os.path.join(EXT_DATA, 'sights.csv')}


# In[6]:


os.chdir(PREPROCESS)
import make_master_table as mmt
os.chdir(CWD)


# In[7]:


# Корретировка ответов по тестовой выборке на выбросы:
def correct_test(df):
    newdf = df.copy(deep=True)
    zeros = [556, 748, 1628, 1951, 4048, 5926, 6106]
    nines = [557,1010,1288,1961,2114,2476,3564,4319,5358,6228,6541]
    large = [5450]
    
    newdf.loc[newdf['Id'].isin(zeros), 'Predicted'] = 0
    newdf.loc[newdf['Id'].isin(nines), 'Predicted'] = 9999
    newdf.loc[newdf['Id'].isin(large), 'Predicted'] = 12624
    
    return newdf


# In[8]:


df_test = pd.read_csv(os.path.join(DATA, 'test.csv'))
df_train = pd.read_csv(os.path.join(DATA, 'train.csv'))

unknown = list(set(df_test.select_dtypes(object)['property_type'].unique())-set(df_train['property_type'].unique()))
df_test[df_test['property_type'].isin(unknown)]


# In[9]:


df_train.groupby(['property_type'])['price'].median()


# In[10]:


df_train.info()


# In[11]:


df_test.info()


# In[12]:


df_train.describe()


# In[13]:


df_test.describe()


# In[14]:


df_train[df_train['price']==0]


# In[15]:


df_train[df_train['price']>0.5*df_train['price'].max()]


# In[16]:


df_train = df_train[df_train['price']!=0].copy(deep=True)
df_train = df_train[df_train['price']<9000].copy(deep=True)
df_train.reset_index(inplace=True, drop=True)


# In[17]:


fig = plt.figure(figsize=(6,5))
plt.hist(df_train['price'], bins=50)
plt.xlim((0, 1300))
plt.title('df_train: Price', fontsize=14)


# In[18]:


df_train['log_price'] = np.log(df_train['price'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax1, ax2 = axes[0], axes[1]

ax1.hist(df_train['price'], bins=50)
ax1.set_xlim((0, 1300))
ax1.set_title('df_train: Price', fontsize=14)

ax2.hist((df_train['log_price']))
ax2.set_title('df_train: Log Price', fontsize=14)

plt.show()


# In[19]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax1, ax2 = axes[0], axes[1]

sns.boxplot(y="price", data=df_train, ax=ax1)
ax1.set_title('df_train: Price', fontsize=14)

sns.boxplot(data=df_train['log_price'], ax=ax2)
ax2.set_title('df_train: Log Price', fontsize=14)

plt.show()


# In[20]:


# Строковые типы в таблице
for c in df_train.columns:
    if isinstance(df_train.iloc[0][c], str):
        print(f'Column: {c}')


# In[21]:


df_train['host_is_superhost'].value_counts()


# In[22]:


df_train['host_identity_verified'].value_counts()


# In[23]:


df_train['property_type'].value_counts()


# In[24]:


df_train['room_type'].value_counts()


# In[25]:


df_train['instant_bookable'].value_counts()


# In[26]:


df_train['cancellation_policy'].value_counts()


# In[27]:


# Теплокарта матрицы корреляций 
fig = plt.figure(figsize=(8,7))
ax = sns.heatmap(df_train.iloc[:,1:].corr(), vmin=0, vmax=1, cmap = 'YlGnBu')
plt.show()


# In[28]:


fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(12, 27))
subset = df_train.select_dtypes(include=['int64', 'float64',]).columns[2:-1]
for ax, feature in tqdm.tqdm_notebook(zip(axes.reshape(-1,1).tolist(), subset.tolist())):
    df_train.plot(feature,"log_price", subplots=True, kind="scatter", 
                  ax=ax, c='blue', alpha = 0.3)
plt.tight_layout()


# In[29]:


def get_orderlist(df, xs, j, k, y, sort):
    if sort:
        return df.groupby([xs[j,k]])[y].median().sort_values(ascending=False).index.tolist()
    else:
        pass
def boxplot_mat(df, x_columns, y, nrows, ncols, figsize, sort):
    xs = np.array(x_columns).reshape(nrows, ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    for j in range(0,nrows):
        for k in range(0,ncols):
            orderlist = get_orderlist(df, xs, j, k, y, sort)
            sns.boxplot(df[xs[j,k]],np.log(df[y]), ax=ax[j, k], order=orderlist)
    plt.show()


# In[30]:


boxplot_mat(df_train, 
            ['host_is_superhost', 'host_identity_verified',
             'property_type', 'room_type',
             'instant_bookable', 'cancellation_policy'], 
             'log_price', 
             2, 3, (22, 12), True)


# In[31]:


boxplot_mat(df_train, 
            ['bedrooms', 'accommodates',
             'bathrooms', 'beds',], 
             'log_price', 
             2, 2, (14, 10), False)


# In[32]:


fig = plt.figure(figsize=(16,10))
orderlist = df_train.groupby(['property_type'])['log_price'].median().sort_values(ascending=False).index.tolist()
sns.boxplot(df_train['property_type'],
            df_train['log_price'],
            order = orderlist)
plt.xticks(rotation=90)
plt.show()


# In[33]:


this_map = folium.Map(prefer_canvas=True)
std = df_train[df_train['price']!=0]['log_price'].std()
mean = df_train[df_train['price']!=0]['log_price'].mean()
legend_html =  """
        <div 
    <br>
        &nbsp; Mean + 3*std
        <i class="fa fa-circle fa-1x"
        style="color:#ff0000"></i>   
  <br>
        &nbsp; Mean -  3*std 
        <i class="fa fa-circle fa-1x"
        style="color:#008000"></i>   
  <br>  
      </div>
      """  
def return_color(log_price):
    if log_price > mean+3*std:
        return 'red'
    elif log_price < mean-3*std:
        return 'green'
    else:
        return 'blue'

def plot_dot(point):
    folium.CircleMarker(location=[point['latitude'], point['longitude']],
                        radius=1,
                        weight=point['log_price']/(5 if point['color'] =='blue' else 1),
                        color=point['color'],
                        fill=True,
                        alpha=0.3).add_to(this_map)

for i in tqdm.tqdm_notebook(range(df_train[df_train['price']!=0].shape[0])):
    point = df_train[df_train['price']!=0][['longitude','latitude', 'log_price']].iloc[i,:].to_dict()
    point['color'] = return_color(point['log_price'])
    plot_dot(point)

this_map.fit_bounds(this_map.get_bounds())
this_map.get_root().html.add_child(folium.Element(legend_html))

this_map.save('MelbournePlot_train.html')
this_map


# In[34]:


this_map = folium.Map(prefer_canvas=True)
for i in tqdm.tqdm_notebook(range(df_test.shape[0])):
    point = df_test[['longitude','latitude']].iloc[i,:].to_dict()
    point['log_price'] = 1
    point['color'] = 'blue'
    plot_dot(point)

this_map.fit_bounds(this_map.get_bounds())

this_map.save('MelbournePlot_test.html')
this_map


# In[35]:


# Расчет рассотяния между объектами на геоиде
def gps_distance(pos1, pos2, earth_rad=6350000.):
    # pos - [lat, long]
    # er - средний радиус земли, в метрах 
    coord_mtrx = np.array([pos1[0],pos2[0],pos1[1],pos2[1]])
    # Перевод координат в радианы
    coord_mtrx = (coord_mtrx*m.pi)/180    
    
    # Расчет на базе константного радиуса геоида 
    sin_delta_long = pow(m.sin( (coord_mtrx[3]-coord_mtrx[2])/2 ),2)
    sin_delta_lat = pow(m.sin( (coord_mtrx[1]-coord_mtrx[0])/2 ),2)
    under_sq = sin_delta_lat + m.cos(coord_mtrx[1])*m.cos(coord_mtrx[0])*sin_delta_long
    phi = 2*m.asin(m.sqrt(under_sq))
    dist = (earth_rad)*phi
    return dist


# In[36]:


# Дополнительынй признак 1: расстояние до центра Мельбурна по поисковой системе
def city_center_feature(df):
    # Центр Мельбурна согласно поисковому запросу OpenStreet Map: -37.8142176,144.9631608
    # https://nominatim.openstreetmap.org/details.php?place_id=258938382
    MELBOURNE_POS = [-37.8142176,144.9631608]
    distns = []
    for i in tqdm.tqdm_notebook(range(df.shape[0])):
        obj_pos = [df.iloc[i]['latitude'], df.iloc[i]['longitude']]
        distns.append(gps_distance(MELBOURNE_POS, obj_pos))
    df['center_dist'] = distns
    df['log_center_dist'] = np.log(df['center_dist'])
    return df
df_train = city_center_feature(df_train)


# In[37]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax1, ax2 = axes[0], axes[1]

ax1.hist(df_train['center_dist'])
ax1.set_title('df_train: Center distance', fontsize=14)

ax2.hist((df_train['log_center_dist']))
ax2.set_title('df_train: Log Center distance', fontsize=14)

plt.show()


# In[38]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax1, ax2 = axes[0], axes[1]

sns.boxplot(y="center_dist", data=df_train, ax=ax1)
ax1.set_title('df_train: Center distance', fontsize=14)

sns.boxplot(data=df_train['log_center_dist'], ax=ax2)
ax2.set_title('df_train: Log Center distance', fontsize=14)

plt.show()


# In[39]:


fig = plt.figure(figsize=(8,5))
sns.scatterplot(df_train['log_center_dist'], df_train['log_price'], alpha=0.5)
plt.show()


# In[40]:


df_districts = pd.read_csv(os.path.join(EXT_DATA, 'districts.csv'))
df_districts


# In[41]:


# Ближайший район к апартаментам
def close_district_feature(df, df_districts):
    close_district = []
    for i in tqdm.tqdm_notebook(range(df.shape[0])):
        obj_pos = [df.iloc[i]['latitude'], df.iloc[i]['longitude']]
        distns = dict()
        for j in range(df_districts.shape[0]):
            district, district_pos = df_districts.iloc[j]['District'],                                                       (df_districts.iloc[j]['latitude'],                                                        df_districts.iloc[j]['longitude'])
            distns[district] = gps_distance(district_pos, obj_pos)
        result = sorted(distns.items(), key=lambda x: x[1])[0]
        close_district.append(result[0])
        #df.loc[i, ['close_district']] = result[0]
    df['close_district'] = close_district
    return df
df_train = close_district_feature(df_train, df_districts)


# In[42]:


fig = plt.figure(figsize=(10,5))
orderlist = df_train.groupby(['close_district'])['log_price'].median().sort_values(ascending=False).index.tolist()
sns.boxplot(df_train['close_district'],
            df_train['log_price'],
            order = orderlist)
plt.xticks(rotation=90)
plt.show()


# In[43]:


# Парков в радиусе n км из списка 
df_parks = pd.read_csv(os.path.join(EXT_DATA, 'parks.csv'))
df_parks


# In[44]:


# Парков в радиусе n км из списка 
def parks_feature(df, df_parks, n_list):  
    for n in n_list:
        df[f'parks_{n}km'] = None
    for i in tqdm.tqdm_notebook(range(df.shape[0])):
        obj_pos = [df.iloc[i]['latitude'], df.iloc[i]['longitude']]
        distns = []
        for j in range(df_parks.shape[0]):
            park, park_pos = df_parks.iloc[j]['park'],                                                       (df_parks.iloc[j]['latitude'],                                                        df_parks.iloc[j]['longitude'])
            distns.append(gps_distance(park_pos, obj_pos))
            
        for n in n_list:
            parks_cnt = len(np.array(distns)[np.array(distns)<=[n*1000]])
            df.loc[i, [f'parks_{n}km']] = parks_cnt
    return df

df_train = parks_feature(df_train, df_parks,[2,5,10,15])


# In[45]:


df_train = df_train[~df_train['log_price'].isnull()].copy(deep=True)
df_train.info()


# In[46]:


df_train.describe()


# In[47]:


fig, axes = plt.subplots(2, 2, figsize=(9, 6))
axes = sum(axes.reshape(-1,1).tolist(),[])
for ax, field in zip(axes,['parks_2km','parks_5km',
                           'parks_10km','parks_15km']):
    ax.scatter(df_train[field], df_train['log_price'])
    ax.set_xlabel(field)
    ax.set_ylabel('log_price')
    ax.set_title(f'Scatter Plot: {field}', fontsize=14)
plt.tight_layout()
plt.show()


# In[48]:


# Некоторые важные места города по данным Airbnb и анализе карт с расположениями жилья
df_sights = pd.read_csv(os.path.join(EXT_DATA, 'sights.csv'))
df_sights


# In[49]:


# Расстояния до 20 точек города
def sights_feature(df, df_sights): 
    for j in tqdm.tqdm_notebook(range(df_sights.shape[0])):
        place, district_pos = df_sights.iloc[j]['place'],                                                    (df_sights.iloc[j]['latitude'],                                                    df_sights.iloc[j]['longitude'])
        # print(place)
        df[place] = None
        tmp = []
        for i in range(df.shape[0]):
            obj_pos = [df.iloc[i]['latitude'], df.iloc[i]['longitude']]
            tmp.append(gps_distance(district_pos, obj_pos))
        df[place] = tmp
    return df
df_train = sights_feature(df_train, df_sights)


# In[50]:


fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(12, 27))
subset = df_train.iloc[:,37:].columns.tolist()
for ax, feature in tqdm.tqdm_notebook(zip(axes.reshape(-1,1).tolist(), subset)):
    sns.scatterplot(np.log(df_train[feature]),df_train['log_price'],ax=ax[0])
plt.tight_layout()


# In[51]:


df_dummies = pd.get_dummies(df_train[df_train.select_dtypes('object').columns.tolist()])

for c in df_dummies.columns.tolist():
    df_train[c]=None
df_train[df_dummies.columns.tolist()] = df_dummies.values
for c in df_train.select_dtypes('object').columns.tolist():
    del df_train[c]


# In[52]:


# Запишем в словарь 3 базовых стратегии обработки пропусков
n = df_train.shape[0]
nullable = []
for c in df_train.columns.tolist():
    if df_train[~df_train[c].isnull()].shape[0]!=n:
        nullable.append(c)

nullable_dict = dict()
nullable_dict['zero'] = dict()
nullable_dict['mean'] = dict()
nullable_dict['median'] = dict()
for c in nullable:
    nullable_dict['zero'][c] = 0
    nullable_dict['mean'][c] = df_train[c].mean()
    nullable_dict['median'][c] = df_train[c].median()
    
# Сохраним словарь со стратегиями обработки
with open('fill_value_strategies.pkl', 'wb') as f:
    pickle.dump(nullable_dict, f)


# In[53]:


def fill_na(df, nullable_dict, strategy):
    fill_dict = nullable_dict[strategy]
    for c in list(fill_dict.keys()):
        df[c].fillna(fill_dict[c], inplace=True)
    return df

def preprocess_data(data, strategy, scaler_type='standard'):
    
    scalers = {'standard':StandardScaler(),
               'robust':RobustScaler(),
               'minmax':MinMaxScaler()}
    
    scaler = scalers[scaler_type]
    data2 = data.copy(deep=True)
    data2 = fill_na(data2, nullable_dict, strategy)
    cols_ = data2.columns.tolist()
    for c in ['id','log_price', 'price']:
        try:
            cols_.remove(c)
        except:
            pass
    scaler.fit(data2[cols_])
    data2[cols_] = scaler.transform(data2[cols_])
    
    y_train = data2['log_price']

    cols = data2.columns.tolist()
    for c in ['id','price', 'log_price']:
        try:
            cols.remove(c)
        except:
            pass
    return data2.loc[:,cols], y_train, cols, scaler


# In[54]:


mmt_instance = mmt.MasterTable(**mmt_dict)
df_train = mmt_instance.make_train(with_dummy=True)


# In[55]:


def cv_result(model, X, y, cv=5,scoring='neg_mean_squared_error'):
    nmse = cross_val_score(model, X, y, cv=cv, scoring=scoring) 
    rmse = [(-x)**0.5 for x in nmse]
    print(f'CrossVal RMSE: {np.mean(rmse)}')
    model.fit(X,y)
    print(f'Intercept:{model.intercept_}')
    stat = [X.columns.tolist(), model.coef_.tolist()]
    res = pd.DataFrame({'Coef':stat[1]}, index=stat[0])
    res.sort_values(by=['Coef'], inplace=True, ascending=False)
    return res


# In[56]:


df_train2, y_train, cols, _ = preprocess_data(df_train, 'zero')


# In[57]:


reg = LinearRegression()
cv_result(reg, df_train2.loc[:,cols[:21]], y_train)


# In[58]:


reg = LinearRegression()
cv_result(reg, df_train2.loc[:,cols], y_train)


# In[59]:


def regulizer_alpha(model, X, y, alphas, cv=5, scoring='neg_mean_squared_error', return_data=False):
    models = [model(a) for a in alphas]
    res = [cross_val_score(model, X, y, cv=cv, scoring=scoring) for model in models]
    mean_rmse = [np.mean((-x)**0.5) for x in res]
    
    fig = plt.figure(figsize = (6,4))
    plt.plot(alphas, mean_rmse, label ='Mean CV RMSE')
    plt.xlabel('Alpha coefficient')
    plt.ylabel('Mean RMSE')
    plt.legend()
    plt.title('Зависимость Mean CV RMSE от alpha.'+              f'\nОптимальное: {alphas[np.argmin(mean_rmse)]} | RMSE: {min(mean_rmse)}')
    print(f'RMSE: {min(mean_rmse)}| alpha: {alphas[np.argmin(mean_rmse)]}')
    plt.show()
    if return_data:
        return mean_rmse


# In[60]:


# Ridge: Начальные данные
regulizer_alpha(Ridge, df_train2.loc[:,cols[:21]], y_train, np.arange(0.05, 50, 0.5))


# In[61]:


# Ridge: Добавленные геопризнаки
regulizer_alpha(Ridge, df_train2.loc[:,cols], y_train, np.arange(0.05, 50, 0.5))


# In[62]:


# Lasso: Начальные данные
regulizer_alpha(Lasso, df_train2.loc[:,cols[:21]], y_train, np.arange(0.001, 20, 0.05))


# In[63]:


# Lasso: Добавленные геопризнаки
regulizer_alpha(Lasso, df_train2.loc[:,cols], y_train, np.arange(0.001, 20, 0.05))


# In[64]:


# Настроим Ridge немного точнее
# Ridge: Добавленные геопризнаки
regulizer_alpha(Ridge, df_train2.loc[:,cols], y_train, np.arange(1e-4, 2, 5e-3))


# In[65]:


df_train2, y_train,_ , _ = preprocess_data(df_train, 'mean')


# In[66]:


# Ridge: Начальные данные
regulizer_alpha(Ridge, df_train2.loc[:,cols[:21]], y_train, np.arange(0.05, 50, 0.5))


# In[67]:


# Ridge: Добавленные геопризнаки
regulizer_alpha(Ridge, df_train2.loc[:,cols], y_train, np.arange(0.05, 50, 0.5))


# In[68]:


# Lasso: Начальные данные
regulizer_alpha(Lasso, df_train2.loc[:,cols[:21]], y_train, np.arange(0.001, 20, 0.05))


# In[69]:


# Lasso: Добавленные геопризнаки
regulizer_alpha(Lasso, df_train2.loc[:,cols], y_train, np.arange(0.001, 20, 0.05))


# In[70]:


# Настроим Ridge немного точнее
# Ridge: Добавленные геопризнаки
regulizer_alpha(Ridge, df_train2.loc[:,cols], y_train, np.arange(1e-4, 2, 5e-3), return_data=False)


# In[71]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'median')


# In[72]:


# Ridge: Начальные данные
regulizer_alpha(Ridge, df_train2.loc[:,cols[:21]], y_train, np.arange(0.05, 50, 0.5))


# In[73]:


# Ridge: Добавленные геопризнаки
regulizer_alpha(Ridge, df_train2.loc[:,cols], y_train, np.arange(0.05, 50, 0.5))


# In[74]:


# Lasso: Начальные данные
regulizer_alpha(Lasso, df_train2.loc[:,cols[:21]], y_train, np.arange(0.001, 20, 0.05))


# In[75]:


# Lasso: Добавленные геопризнаки
regulizer_alpha(Lasso, df_train2.loc[:,cols], y_train, np.arange(0.001, 20, 0.05))


# In[76]:


# Настроим Ridge немного точнее
# Ridge: Добавленные геопризнаки
regulizer_alpha(Ridge, df_train2.loc[:,cols], y_train, np.arange(1e-4, 2, 5e-3))


# In[77]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'zero')


# In[78]:


ridge = Ridge(0.0001)
df_ridge = cv_result(ridge, df_train2.loc[:,cols], y_train)
df_ridge.head(20)


# In[79]:


lasso = Lasso(0.001)
df_lasso = cv_result(lasso, df_train2.loc[:,cols], y_train)
df_lasso.head(20)


# In[80]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'zero')


# In[81]:


def cvknn_result(X, y, **kwargs):
    rmses = []
    for k in tqdm.tqdm_notebook(kwargs['k']):
        model = KNeighborsRegressor(n_neighbors=k,
                                    metric=kwargs['metric'],
                                    metric_params=kwargs['param'])
        nmse = cross_val_score(model, X, y, cv=kwargs['cv'],                                scoring=kwargs['scoring']) 
        rmse = [(-x)**0.5 for x in nmse]
        rmses.append(np.mean(rmse))
                                                  
    fig, ax = plt.subplots()
    ax.plot(kwargs['k'], rmses)
    plt.title(f'Optimal neighbours: {1 + np.argmin(rmses)}'+              f'\nCV RMSE: {rmses[np.argmin(rmses)]}')
    ax.set_xlabel('Number of neighbours')
    ax.set_ylabel('CV RMSE')
    print(f'RMSE: {rmses[np.argmin(rmses)]} | k: {1 + np.argmin(rmses)}')
    plt.show()


# In[82]:


params_dict = {'k': np.arange(1, 31), 'metric':'minkowski',
               'param':{'p': 2}, 'cv':5,
               'scoring':'neg_mean_squared_error'}
cvknn_result(df_train2.loc[:,cols[:21]], y_train, **params_dict)


# In[83]:


params_dict = {'k': np.arange(1, 31), 'metric':'minkowski',
               'param':{'p': 2}, 'cv':5,
               'scoring':'neg_mean_squared_error'}
cvknn_result(df_train2.loc[:,cols], y_train, **params_dict)


# In[84]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'mean')


# In[85]:


params_dict = {'k': np.arange(1, 31), 'metric':'minkowski',
               'param':{'p': 2}, 'cv':5,
               'scoring':'neg_mean_squared_error'}
cvknn_result(df_train2.loc[:,cols[:21]], y_train, **params_dict)


# In[86]:


params_dict = {'k': np.arange(1, 31), 'metric':'minkowski',
               'param':{'p': 2}, 'cv':5,
               'scoring':'neg_mean_squared_error'}
cvknn_result(df_train2.loc[:,cols], y_train, **params_dict)


# In[87]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'median')


# In[88]:


params_dict = {'k': np.arange(1, 31), 'metric':'minkowski',
               'param':{'p': 2}, 'cv':5,
               'scoring':'neg_mean_squared_error'}
cvknn_result(df_train2.loc[:,cols[:21]], y_train, **params_dict)


# In[89]:


params_dict = {'k': np.arange(1, 31), 'metric':'minkowski',
               'param':{'p': 2}, 'cv':5,
               'scoring':'neg_mean_squared_error'}
cvknn_result(df_train2.loc[:,cols], y_train, **params_dict)


# In[90]:


def cv_random_forest(X, y, **kwargs):
    rmses = []
    trees = kwargs['trees']
    for k in tqdm.tqdm_notebook(trees):
        model = RandomForestRegressor(random_state=42,n_estimators=k)
        nmse = cross_val_score(model, X, y, cv=kwargs['cv'],                                scoring=kwargs['scoring']) 
        rmse = [(-x)**0.5 for x in nmse]
        rmses.append(np.mean(rmse))
                                                  
    fig, ax = plt.subplots()
    ax.plot(kwargs['trees'], rmses)
    
    plt.title(f'Optimal trees: {trees[np.argmin(rmses)]}'+              f'\nCV RMSE: {rmses[np.argmin(rmses)]}')
    ax.set_xlabel('Number of trees')
    ax.set_ylabel('CV RMSE')
    print(f'RMSE: {rmses[np.argmin(rmses)]} | Trees: {trees[np.argmin(rmses)]}')
    plt.show()


# In[91]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'zero')


# In[92]:


params_dict = {'trees': np.arange(5, 201, 20), 'cv':5,
               'scoring':'neg_mean_squared_error'}
cv_random_forest(df_train2.loc[:,cols[:21]], y_train, **params_dict)


# In[93]:


params_dict = {'trees': np.arange(5, 201, 20), 'cv':5,
               'scoring':'neg_mean_squared_error'}
cv_random_forest(df_train2.loc[:,cols], y_train, **params_dict)


# In[94]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'mean')


# In[95]:


params_dict = {'trees': np.arange(5, 201, 20), 'cv':5,
               'scoring':'neg_mean_squared_error'}
cv_random_forest(df_train2.loc[:,cols[:21]], y_train, **params_dict)


# In[96]:


params_dict = {'trees': np.arange(5, 201, 20), 'cv':5,
               'scoring':'neg_mean_squared_error'}
cv_random_forest(df_train2.loc[:,cols], y_train, **params_dict)


# In[97]:


df_train2, y_train, _, _ = preprocess_data(df_train, 'median')


# In[98]:


params_dict = {'trees': np.arange(5, 201, 20), 'cv':5,
               'scoring':'neg_mean_squared_error'}
cv_random_forest(df_train2.loc[:,cols[:21]], y_train, **params_dict)


# In[99]:


params_dict = {'trees': np.arange(5, 201, 20), 'cv':5,
               'scoring':'neg_mean_squared_error'}
cv_random_forest(df_train2.loc[:,cols], y_train, **params_dict)


# In[100]:


df_train2, y_train, cols, _ = preprocess_data(df_train, 'zero')


# In[101]:


def rf_randomized_cv_search(X, y, **kwargs):
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = kwargs['random_grid'], 
                                   n_iter = kwargs['iter'], cv = kwargs['cv'],
                                   verbose=kwargs['verbose'], random_state=42, n_jobs = -1)
    rf_random.fit(X, y)
    best_params = rf_random.best_estimator_

    # CV RMSE
    model = RandomForestRegressor(random_state=42, n_estimators=kwargs['n_estimators'],
                                  max_features=best_params.max_features,
                                  max_depth=best_params.max_depth,
                                  min_samples_split=best_params.min_samples_split,
                                  min_samples_leaf=best_params.min_samples_leaf,
                                  bootstrap=best_params.bootstrap)
    nmse = cross_val_score(model, X, y, cv=kwargs['cv'],                                scoring=kwargs['scoring']) 
    rmse = [(-x)**0.5 for x in nmse]
    print(f'RMSE: {np.mean(rmse)}')
    print(f'CV list: {rmse}')
    
    return best_params


# In[102]:


max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
STEPS = 15
random_grid = {
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
              }
params_dict = {'random_grid':random_grid,
               'cv':5, 'iter':STEPS,'verbose':2,
               'scoring':'neg_mean_squared_error', 'n_estimators':145}


# In[103]:


best = rf_randomized_cv_search(df_train2.loc[:,cols], y_train,**params_dict)


# In[104]:


rf_explainer = shap.TreeExplainer(best)
shap_values = rf_explainer.shap_values(df_train2.loc[:,cols], approximate=True)


# In[105]:


shap.summary_plot(shap_values, df_train2.loc[:,cols])


# In[106]:


shap.summary_plot(shap_values, df_train2.loc[:,cols], plot_type="bar")


# In[107]:


class RFHyperoptSearch:
    def __init__(self, X, y, cv):
        super(RFHyperoptSearch, self).__init__()
        self.X = X
        self.y = y
        self.cv = cv
        
    def rmse(self, y, y_pred):
        return np.sqrt(np.mean((y_pred - y)**2))
    
    def eval_func(self, params):
        
        for k in list(params.keys()):
            if k=='min_weight_fraction_leaf':
                pass
            params[k] = int(round(params[k],0))
        
        model = RandomForestRegressor(**params, random_state=42)
        scoring_rmse = make_scorer(self.rmse, greater_is_better=False)
        rmse = -cross_val_score(model, self.X, self.y, cv=self.cv,                                 scoring=scoring_rmse)
        return np.mean(rmse)


# In[108]:


params_space = {'max_depth': hyperopt.hp.uniform('max_depth', 100, 200),
                'n_estimators': hyperopt.hp.uniform('n_estimators', 140, 160),
                'min_samples_split': hyperopt.hp.uniform('min_samples_split', 2, 40),
                'min_samples_leaf': hyperopt.hp.uniform('min_samples_leaf', 2, 30)}

hyperopt_inst = RFHyperoptSearch(df_train2.loc[:,cols], y_train, 5)
trials = hyperopt.Trials()
best = hyperopt.fmin(
    hyperopt_inst.eval_func,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=10,
    trials=trials,
    rstate=np.random.RandomState(42)
)
print(best)


# In[109]:


best


# In[110]:


best['n_estimators'] = int(best['n_estimators'])
best['min_samples_leaf'] = int(best['min_samples_leaf'])
best['min_samples_split'] = int(best['min_samples_split'])
rf = RandomForestRegressor(**best, random_state=42)
rf.fit(df_train2.loc[:,cols], y_train)
rf_explainer = shap.TreeExplainer(rf)
shap_values = rf_explainer.shap_values(df_train2.loc[:,cols], approximate=True)


# In[111]:


shap.summary_plot(shap_values, df_train2.loc[:,cols])


# In[112]:


shap.summary_plot(shap_values, df_train2.loc[:,cols], plot_type="bar")


# In[113]:


mmt_instance = mmt.MasterTable(**mmt_dict)
df_train = mmt_instance.make_train(with_dummy=True)


# In[114]:


df_train2, y_train2, _ , boost_scaler = preprocess_data(df_train, 'zero')

X_train, X_val, y_train, y_val = train_test_split(df_train2, y_train2, test_size=VAL_SIZE,
                                                  random_state=42, shuffle=True)


# In[115]:


class HyperOpt:
    def __init__(self, **kwargs):
        super(HyperOpt, self).__init__()
        self.kwargs = kwargs
       
    def hyperopt_xgb_score(self, params):
        
        model = XGBRegressor(l2_leaf_reg=int(params['l2_leaf_reg']),
                             learning_rate=params['learning_rate'],
                             max_depth=int(params['max_depth']),
                             gamma = params['gamma'],
                             reg_alpha = params['reg_alpha'],
                             reg_lambda = params['reg_lambda'],
                             n_estimators=self.kwargs['n_estimators'],
                             objective='reg:squarederror',
                             verbosity=0,
                             random_seed=42,
                             task_type=DEVICE)
        fit_params={'early_stopping_rounds': self.kwargs['rounds'], 
                    'eval_metric': 'rmse',
                    'verbose': self.kwargs['verbose'],
                    'eval_set': [[self.kwargs['X_val'],  self.kwargs['y_val']]]}
        
        xgb_cv = cross_val_score(model, self.kwargs['X_train'], self.kwargs['y_train'], 
                                 cv = self.kwargs['cv'], 
                                 scoring = 'neg_mean_squared_error',
                                 fit_params = fit_params)
        best_rmse = np.mean([(-x)**0.5 for x in xgb_cv])
        print(f'Best RMSE: {best_rmse}', params)
        return best_rmse
    
    def hyperopt_catb_score(self, params):
        model = CatBoostRegressor(l2_leaf_reg=int(params['l2_leaf_reg']),
                                  learning_rate=params['learning_rate'],
                                  iterations=self.kwargs['iterations'],
                                  ignored_features = self.kwargs['ignored_features'],
                                  eval_metric='RMSE',
                                  random_seed=42,
                                  task_type=DEVICE,
                                  logging_level='Silent'
                                 )
    
        cv_data = cv(Pool(self.kwargs['X_train'], self.kwargs['y_train'], 
                          cat_features=self.kwargs['categorical_features_indices']),
                     model.get_params())
        best_rmse = np.min(cv_data['test-RMSE-mean'])
        return best_rmse
    
    def hyperopt_lgbm_score(self, params):
        model = LGBMRegressor(learning_rate=params['learning_rate'],
                              max_depth=int(params['max_depth']),
                              n_estimators=int(self.kwargs['n_estimators']),
                              subsample=params['subsample'],
                              reg_alpha = params['reg_alpha'],
                              reg_lambda = params['reg_lambda'],
                              silent = True,
                              metric='rmse',
                              random_state=42)
        
        fit_params={'early_stopping_rounds': self.kwargs['rounds'], 
                    'eval_metric': 'rmse',
                    'verbose': self.kwargs['verbose'],
                    'eval_set': [[self.kwargs['X_val'],  self.kwargs['y_val']]]}
        
        lgb_cv = cross_val_score(model, self.kwargs['X_train'], self.kwargs['y_train'], 
                                 cv = self.kwargs['cv'], 
                                 scoring = 'neg_mean_squared_error',
                                 fit_params = fit_params)
        
        best_rmse = np.mean([(-x)**0.5 for x in lgb_cv])
        print(f'Best RMSE: {best_rmse}', params)
        return best_rmse


# In[116]:


# Посмотрим на качество с дефолтными параметрами, 1500 деревьями и ранней остановкой
params = {'n_estimators':1500,
          'objective':'reg:squarederror',
          'random_seed':42,
          'verbosity':0,
          'task_type': DEVICE,
          'early_stopping_rounds':50}
gbm = XGBRegressor(**params)
gbm.fit(X_train,y_train, eval_set=[[X_val, y_val]])


# In[117]:


nlistgbm, implistgbm = [], []
for n, imp in zip(X_train.columns.tolist(), gbm.feature_importances_):
    nlistgbm.append(n)
    implistgbm.append(imp)
    
df_impgbm = pd.DataFrame({'Feature':nlistgbm, 'Importance': implistgbm})
df_impgbm.sort_values(by=['Importance'], ascending=False).head(20)


# In[118]:


min(gbm.evals_result_['validation_0']['rmse'])


# In[119]:


best = {'gamma': 0.15340366103115533,
 'l2_leaf_reg': 2.0,
 'learning_rate': 0.00853890793354474,
 'max_depth': 5.971733628773733,
 'reg_alpha': 1.8400184528746324,
 'reg_lambda': 1.0868061353806249}


# In[120]:


# Обучим итоговую модель с обновлением параметров
params = {'n_estimators':2200,
          'objective':'reg:squarederror',
          'random_seed':42,
          'verbosity':0,
          'task_type':DEVICE}

best['max_depth'] = round(best['max_depth'])
params.update(best)
print(params)

gbm = XGBRegressor(**params)
# Выделим небольшую часть данных для остановки
X_train, X_val, y_train, y_val = train_test_split(df_train2, y_train2,
                                                  test_size=VAL_SIZE, random_state=42,
                                                  shuffle=True)
gbm.fit(X_train,y_train, eval_set=[[X_val, y_val]])


# In[121]:


nlistgbm, implistgbm = [], []
for n, imp in zip(X_train.columns.tolist(), gbm.feature_importances_):
    nlistgbm.append(n)
    implistgbm.append(imp)
    
df_impgbm = pd.DataFrame({'Feature':nlistgbm, 'Importance': implistgbm})
df_impgbm.sort_values(by=['Importance'], ascending=False).head(20)


# In[122]:


xgb_explainer = shap.TreeExplainer(gbm)
shap_values = xgb_explainer.shap_values(X_val, approximate=True)


# In[123]:


shap.summary_plot(shap_values, X_val)


# In[124]:


shap.summary_plot(shap_values, X_val, plot_type="bar")


# In[125]:


test_to_submit = mmt_instance.make_test(with_dummy=True)


# In[126]:


def process_test_data(scaler, test_data):
    test_data.fillna(value=0, inplace=True)
    test_id = test_data['id'].copy(deep=True)
    try:
        del test_data['id']
    except:
        pass
    
    columns = test_data.columns.tolist()
    for c in test_data.select_dtypes('object').columns.tolist():
        columns.remove(c)
    test_data[columns] = scaler.transform(test_data[columns])
    return test_data, test_id

def predict_test(test_data, model):
    result = model.predict(test_data)
    # Для возврата реальной цены возведем логарифм в exp
    return np.exp(result) 


# In[127]:


test_to_submit_processed, test_to_submit_id = process_test_data(boost_scaler, test_to_submit)
pred = predict_test(test_to_submit_processed, gbm)

submit_result_xgb = pd.DataFrame({'Id':test_to_submit_id.values,
                                  'Predicted':pred})
submit_result_xgb = correct_test(submit_result_xgb)
submit_result_xgb.to_csv('xgb_submit_kaggle.csv', index=False)


# In[128]:


mmt_instance = mmt.MasterTable(**mmt_dict)
df_train = mmt_instance.make_train(with_dummy=False)


# In[129]:


def catboost_data_preparation(data):
    try:
        del data['id']
    except:
        pass
    data.fillna(value=0, inplace=True)
    data_cols = data.columns.tolist()
    categorical_features_indices = [data_cols.index(x) for x in data.select_dtypes('object').columns.tolist()]
    
    #for x in data.columns.tolist():
        #print(x, data[~data[x].isnull()].shape)
    # Стандартизация признаков
    scaler = StandardScaler()
    cols = data.columns.tolist()
    for c in data.select_dtypes('object').columns.tolist()+['price', 'log_price']:
        cols.remove(c)
    data[cols] = scaler.fit_transform(data[cols])
    return categorical_features_indices, cols, scaler, data
categorical_features_indices, catbs_cols, catb_scaler, data = catboost_data_preparation(df_train)


# In[130]:


categorical_features_indices


# In[131]:


X_train, X_val, y_train, y_val = train_test_split(df_train, df_train['log_price'],
                                                  test_size=VAL_SIZE, random_state=42,
                                                  shuffle=True)


# In[132]:


# Посмотрим на качество с дефолтными параметрами
params = {'iterations': 3000,
          'loss_function': 'RMSE',
          'eval_metric': 'RMSE',
          'ignored_features':['price', 'log_price'],
          'use_best_model': True,
          'logging_level': 'Silent',
          'task_type': DEVICE}
   
catb = CatBoostRegressor(**params)
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_val, y_val, cat_features=categorical_features_indices)
catb.fit(train_pool, eval_set=validate_pool)


# In[133]:


nlist, implist = [], []
for n, imp in zip(catb.feature_names_, catb.feature_importances_):
    nlist.append(n)
    implist.append(imp)
    
df_imp = pd.DataFrame({'Feature':nlist, 'Importance': implist})
df_imp.sort_values(by=['Importance'], ascending=False).head(20)


# In[134]:


catb.best_score_


# In[135]:


best = {'l2_leaf_reg': 6.0, 'learning_rate': 0.08390144719977513}


# In[136]:


# Итоговое обучение
params = {
    'iterations': 2200,
    'learning_rate': 0.003,
    'l2_leaf_reg': 2.0,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'ignored_features':['price', 'log_price'],
    'task_type': DEVICE}

params.update(best)
params.update({'use_best_model': True})
    
# Выделим небольшую часть данных для остановки
X_train, X_val, y_train, y_val = train_test_split(data, data['log_price'],test_size=VAL_SIZE, 
                                                  random_state=42, shuffle=True)
catb = CatBoostRegressor(**params)
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_val, y_val, cat_features=categorical_features_indices)
catb.fit(train_pool, eval_set=validate_pool)


# In[137]:


nlist, implist = [], []
for n, imp in zip(catb.feature_names_, catb.feature_importances_):
    nlist.append(n)
    implist.append(imp)
    
df_imp = pd.DataFrame({'Feature':nlist, 'Importance': implist})
df_imp.sort_values(by=['Importance'], ascending=False).head(20)


# In[138]:


catb_explainer = shap.TreeExplainer(catb)
shap_values = catb_explainer.shap_values(validate_pool)


# In[139]:


shap.summary_plot(shap_values, X_val)


# In[140]:


shap.summary_plot(shap_values, X_val, plot_type="bar")


# In[141]:


test_to_submit = mmt_instance.make_test(with_dummy=False)

test_to_submit_processed, test_to_submit_id = process_test_data(catb_scaler, test_to_submit)

# Для catboost необходимо наличие всех столбцов, аналогичных трейну. Добавим искусственно:
test_to_submit_processed['price'] = 0
test_to_submit_processed['log_price'] = 0
test_to_submit_processed = test_to_submit_processed[X_train.columns.tolist()]


# In[142]:


pred = predict_test(test_to_submit_processed, catb)

submit_result_catb = pd.DataFrame({'Id':test_to_submit_id.values,
                                   'Predicted':pred})
submit_result_catb = correct_test(submit_result_catb)
submit_result_catb.to_csv('catb_submit_kaggle.csv',index=False)


# In[143]:


df_train = mmt_instance.make_train(with_dummy=True)

df_train2, y_train2, _ , boost_scaler = preprocess_data(df_train, 'zero', 'standard')
X_train, X_val, y_train, y_val = train_test_split(df_train2, y_train2, test_size=VAL_SIZE,
                                                  random_state=42, shuffle=True)


# In[144]:


## Посмотрим на качество с дефолтными параметрами
params = {'n_estimators':1500,
          'random_seed':42,
          'metric':'rmse',
          'silent':True,
          'early_stopping_rounds':50
         }
lgbm = LGBMRegressor(**params)
lgbm.fit(X_train, y_train, eval_set=[[X_val, y_val]])


# In[145]:


nlistlgbm, implistlgbm = [], []
for n, imp in zip(X_train.columns.tolist(), lgbm.feature_importances_):
    nlistlgbm.append(n)
    implistlgbm.append(imp)
    
df_implgbm = pd.DataFrame({'Feature':nlistlgbm, 'Importance': implistlgbm})
df_implgbm.sort_values(by=['Importance'], ascending=False).head(20)


# In[146]:


lgbm.best_score_


# In[147]:


best = {'learning_rate': 0.00865486636310301, 
        'max_depth': 14.36591078571328, 
        'reg_alpha': 0.6068202197683499, 
        'reg_lambda': 1.766630554761724, 
        'subsample': 0.8373883555532842}


# In[148]:


# Обучим итоговую модель с обновлением параметров
params = {'n_estimators':1500,
          'random_seed':42,
          'metric':'rmse',
          'early_stopping_rounds':50
         }

best['max_depth'] = int(best['max_depth'])
params.update(best)
print(params)

lgbm = LGBMRegressor(**params)
# Выделим небольшую часть данных для остановки
X_train, X_val, y_train, y_val = train_test_split(df_train2, y_train2,
                                                  test_size=VAL_SIZE, random_state=42,
                                                  shuffle=True)
lgbm.fit(X_train,y_train, eval_set=[[X_val, y_val]])


# In[149]:


lgbm_explainer = shap.TreeExplainer(lgbm)
shap_values = lgbm_explainer.shap_values(X_val)


# In[150]:


shap.summary_plot(shap_values, X_val)


# In[151]:


shap.summary_plot(shap_values, X_val, plot_type="bar")


# In[152]:


test_to_submit = mmt_instance.make_test(with_dummy=True)

test_to_submit_processed, test_to_submit_id = process_test_data(boost_scaler, test_to_submit)
pred = predict_test(test_to_submit_processed, lgbm)

submit_result_lgbm = pd.DataFrame({'Id':test_to_submit_id.values,
                                  'Predicted':pred})

submit_result_lgbm = correct_test(submit_result_lgbm)
submit_result_lgbm.to_csv('lgbm_submit_kaggle.csv',index=False)

