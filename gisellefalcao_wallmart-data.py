#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


## Importe as bilbiotecas##
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import os
import seaborn as sns # visualization
from scipy import stats
from scipy.stats import norm 
import warnings 
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore') #ignore warnings

get_ipython().run_line_magic('matplotlib', 'inline')
import gc

import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[3]:


pwd


# In[4]:


tt=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip", parse_dates=["Date"])
tst=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip", parse_dates=["Date"])
lj=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
fat = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip", parse_dates=["Date"])


# In[5]:


#viasualiza previamente 5 linha do arquivo 'features.csv'
tt.head (5)


# In[6]:


#Converte date to object do arquivo 'features.csv'# 
fat['Date'] = pd.to_datetime(fat['Date']) #indica o titulo da coluna que quer converter 
fat.head(2)


# In[7]:


#viasualiza previamente 5 linha do arquivo 'test.csv'
tst.head (5)


# In[8]:


#Converte date to object do arquivo 'test.csv'# 
tst['Date'] = pd.to_datetime(tst['Date']) #indica o titulo da coluna que quer converter 
tst.head(2)


# In[9]:


#viasualiza previamente 5 linha do arquivo 'store.csv'
lj.head (5)


# In[10]:


#viasualiza previamente 5 linha do arquivo 'train.csv'
tt.head (5)


# In[11]:


#Converte date to object do arquivo 'train.csv'# 
tt['Date'] = pd.to_datetime(tt['Date']) #indica o titulo da coluna que quer converter 
tt.head(2)


# In[12]:


fat['IsHoliday'] . value_counts ()


# In[13]:


tt['IsHoliday'] . value_counts ()


# In[14]:


print("A estrutura dos dados de treinamento é:", tt.shape)
print("A estrutura dos dados de treinamento é:", tst.shape)
print("A proporção entre os dados de treinamento e os dados de teste é:", (round(tt.shape[0]*100/(tt.shape[0]+tst.shape[0])),100-round(tt.shape[0]*100/(tt.shape[0]+tst.shape[0]))))


# In[15]:


tt=tt.merge(lj, on='Store', how='left')
tt.head()


# In[16]:


tt['Ano']=tt['Date'].dt.year
tt['Mês']=tt['Date'].dt.month
tt['Semana']=tt['Date'].dt.week
tt['Dia']=tt['Date'].dt.day
tt['n_dias']=(tt['Date'].dt.date-tt['Date'].dt.date.min()).apply(lambda x:x.days)


# In[17]:


Ano=pd.Series(tt['Ano'].unique())
Semana=pd.Series(tt['Semana'].unique())
Mês=pd.Series(tt['Mês'].unique())
Dia=pd.Series(tt['Dia'].unique())
n_dias=pd.Series(tt['n_dias'].unique())


# In[18]:


tst['Ano']=tst['Date'].dt.year
tst['Mês']=tst['Date'].dt.month
tst['Semana']=tst['Date'].dt.week
tst['Dia']=tst['Date'].dt.day
tst['n_dias']=(tst['Date'].dt.date-tst['Date'].dt.date.min()).apply(lambda x:x.days)


# In[19]:


Ano=pd.Series(tst['Ano'].unique())
Semana=pd.Series(tst['Semana'].unique())
Mês=pd.Series(tst['Mês'].unique())
Dia=pd.Series(tst['Dia'].unique())
n_dias=pd.Series(tst['n_dias'].unique())


# In[20]:


print("O formato do conjunto de dados 'store.csv' é: ", lj.shape)
print("Os numeros das lojas são: ", lj['Store'].unique())
print("Os tipos de lojas são:", lj['Type'].unique())


# In[21]:


fat['Ano']=fat['Date'].dt.year
fat['Mês']=fat['Date'].dt.month
fat['Semana']=fat['Date'].dt.week
fat['Dia']=fat['Date'].dt.day
#tt['n_dias']=(tt['Date'].dt.date-tt['Date'].dt.date.min()).apply(lambda x:x.days)


# In[22]:


Ano=pd.Series(fat['Ano'].unique())
Semana=pd.Series(fat['Semana'].unique())
Mês=pd.Series(fat['Mês'].unique())
Dia=pd.Series(fat['Dia'].unique())
#n_dias=pd.Series(fat['n_dias'].unique())


# In[23]:


fat.head()


# In[24]:


fat_1 = fat.loc[(fat['Ano'] > 2012)
                |(fat['Mês'] > 10) & (fat['Ano']== 2012)]
tfat = fat.drop(fat_1.index)
tfat.head()


# In[25]:


tfat.shape


# In[26]:


fatext = ['Store','Date', 'Temperature','Fuel_Price','CPI','Unemployment','IsHoliday']
fat_tt = tfat.filter(items=fatext)
fat_tt.head()


# In[27]:


fatext = ['Store','Date', 'Temperature','Fuel_Price','CPI','Unemployment','IsHoliday']
fat_tst = fat_1.filter(items=fatext)
fat_tst.head()


# In[28]:


fat_tt.shape


# In[29]:


fatext = ['Store','Date', 'Temperature','Fuel_Price','CPI','Unemployment','IsHoliday']
fat_2 = tfat.filter(items=fatext)
fat_2.head()


# In[30]:


fat_tst.shape


# In[31]:


tfat = tt.groupby(['Store','Date']).agg({'Weekly_Sales': np.mean})
tfat.index_col=0
tfat.head()


# In[32]:


tst=tst.merge(lj, on='Store', how='left')
tst.head()


# In[33]:


tstfat = tst.groupby(['Store','Date']).agg({'Size':np.mean})
tstfat.index_col=0
tstfat.head()


# In[34]:


tstfat.shape


# In[35]:


tt_1 = tfat.reset_index(level=['Store', 'Date'])
tst_1 = tstfat.reset_index(level=['Store', 'Date'])
tt_1.head()


# In[36]:


tst_1.head()


# In[37]:


tt_1['Date'] = pd.to_datetime(tt_1['Date'])
tst_1['Date'] = pd.to_datetime(tst_1['Date'])


# In[38]:


New_fat=pd.merge(fat_tt, tt_1,
                      on=['Date', 'Store'])
New_fat.head()


# In[39]:


New_fat_tst=pd.merge(fat_tst, tst_1,
                      on=['Date', 'Store'])


# In[40]:


New_fat_tst.head()


# In[41]:


print(New_fat.describe()['Weekly_Sales'].round(2))


# In[42]:


print(lj.head())
grouped=lj.groupby('Type')
print(grouped.describe()['Size'].round(2))


# In[43]:


print(tt_1.head())
grouped=tt_1.groupby('Store')
print(grouped.describe()['Weekly_Sales'].round(2))


# In[44]:


data = pd.concat([lj['Type'], lj['Size']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))

plt.xlabel('Tipo')
plt.ylabel('Tamanho da loja')

fig = sns.boxplot(x='Type', y='Size', data=data)


# In[45]:


data = pd.concat([tt['Type'], tt['Weekly_Sales']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
plt.xlabel('Tipo')
plt.ylabel('Venda semanal')
fig = sns.boxplot(x='Type', y='Weekly_Sales', data=data, showfliers=False)


# In[46]:


plt.style.use('ggplot')

fig=plt.figure()
ax=fig.add_subplot(111)
plt.xlabel('Tamanho da loja')
plt.ylabel('Venda semanal')
ax.scatter(tt['Size'],tt['Weekly_Sales'], alpha=0.5)

plt.show()


# In[47]:


types=lj['Type'].unique()

plt.style.use('ggplot')

fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(111)

for t in types:
    x=tt.loc[tt['Type']==t, 'Size']
    y=tt.loc[tt['Type']==t, 'Weekly_Sales']
    
    ax.scatter(x,y,alpha=0.5, label=t)

ax.set_title('Gráfico de dispersão do tamanho da loja volume de venda e tipo de loja')
ax.set_xlabel('Tamanho')
ax.set_ylabel('Venda semanal')

ax.legend(loc='higher right',fontsize=12)
plt.style.use('classic')

plt.show()


# In[48]:


data = pd.concat([tt['Store'], tt['Weekly_Sales'], tt['Type']], axis=1)
f, ax = plt.subplots(figsize=(20, 8))
fig = sns.boxplot(x='Store', y='Weekly_Sales', data=data, showfliers=False, hue="Type")


# In[49]:


####### alterar no dataframe Store (str) to Store (int) 
# mod= 'Weekly_Sales ~ C(Store) + C(Type)'
# lm= ols(mod, data=data).fit()
# aov_tab = sm.stats.anova_lm(mod, type=2)
# print (aov_tab)


# In[50]:


New_fat.head()


# In[51]:


##############################     Verificar (gráfico ruim)
data = pd.concat([New_fat['Temperature'], New_fat['Weekly_Sales'], New_fat['IsHoliday']], axis=1)
f, ax = plt.subplots(figsize=(30, 8))
plt.title('Relação entre Desemprego e vendas') 
fig = sns.boxplot(x='Temperature', y='Weekly_Sales', data=data, showfliers=False, hue="IsHoliday")


# In[52]:


data = pd.concat([New_fat['Fuel_Price'], New_fat['Weekly_Sales'], New_fat['IsHoliday']], axis=1)
f, ax = plt.subplots(figsize=(25, 8))
plt.title('Relação entre Preço da gasolina e vendas') 
fig = sns.boxplot(x='Fuel_Price', y='Weekly_Sales', data=data, showfliers=False, hue="IsHoliday")


# In[53]:


data = pd.concat([New_fat['Unemployment'], New_fat['Weekly_Sales'], New_fat['IsHoliday']], axis=1)
f, ax = plt.subplots(figsize=(27, 8))
plt.title('Relação entre Desemprego e vendas') 
fig = sns.boxplot(x='Unemployment', y='Weekly_Sales', data=data, showfliers=False, hue="IsHoliday")


# In[54]:


def plota_bar_dupla_1():
   grupos = 45
   HolidayTrue = (tt['IsHoliday==True'])
   HolidayFalse = (tt['IsHoliday==False'])
   fig, ax = plt.subplots()
   indice = np.arange(grupos)
   bar_larg = 0.4
   transp = 0.7
   plt.bar(indice, HolidayTrue, bar_larg, alpha=transp, color=azul, label='Holiday')
   plt.bar(indice + bar_larg, HolidayFalse, bar_larg, alpha=transp, color=verde, label='No Holiday')

plt.xlabel('Weekly_Sales') 
plt.ylabel('Stores') 
plt.title('Vendas por lojas') 
#plt.xticks(indice + bar_larg) 
plt.legend() 
plt.tight_layout() 
plt.show()


# In[55]:


#Tentar colocar verdadeiro e falso para feriado ao lado e todas as lojas no eixo x
bins = np.linspace(tt.Weekly_Sales.min(), tt.Weekly_Sales.max(), 10)
g = sns.FacetGrid(tt, col="Store", hue="IsHoliday", palette="Set1", col_wrap=5)
g.map(plt.hist, 'Weekly_Sales', bins=bins, ec="k")

plt.title('Volume máximo e minimo das Vendas por lojas') 
g.axes[-1].legend()
plt.show()


# In[56]:


##Correlaçao
def plot_corr(corr):
    # Cortaremos a metade de cima pois é o espelho da metade de baixo
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True

    sns.heatmap(corr, mask=mask, cmap='RdBu', square=True, linewidths=.2)

# Calculando a correlação
corr = tt.corr() 
plt.title('Correlação entre os dados') 
plot_corr(corr)


# In[57]:


data = pd.concat([tt['Store'], tt['Weekly_Sales'], tt['IsHoliday']], axis=1)
f, ax = plt.subplots(figsize=(30, 8))
fig = sns.boxplot(x='Store', y='Weekly_Sales', data=data, showfliers=False, hue="IsHoliday")
plt.title('Relação entre as lojas, vendas semanais e feriados') 


# In[58]:


data = pd.concat([tt['Dept'], tt['Weekly_Sales'], tt['Type']], axis=1)
f, ax = plt.subplots(figsize=(25, 10))
plt.title('Relação entre as Departamento e vendas semanais') 
fig = sns.boxplot(x='Dept', y='Weekly_Sales', data=data, showfliers=False)


# In[59]:


data = pd.concat([tt['Dept'], tt['Weekly_Sales'], tt['Type']], axis=1)
f, ax = plt.subplots(figsize=(10, 50))
fig = sns.boxplot(y='Dept', x='Weekly_Sales', data=data, showfliers=False, hue="Type",orient="h") 


# In[60]:


data = pd.concat([tt['Dept'], tt['Weekly_Sales'], tt['IsHoliday']], axis=1)
f, ax = plt.subplots(figsize=(25, 10))
fig = sns.boxplot(x='Dept', y='Weekly_Sales', data=data, showfliers=False, hue="IsHoliday")


# In[61]:


plt.style.use('ggplot')
fig, axes = plt.subplots(1,2, figsize = (20,5))
fig.subplots_adjust(wspace=1, hspace=1)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

sales_holiday=tt[['IsHoliday','Weekly_Sales']]
target=[sales_holiday['Weekly_Sales'].loc[sales_holiday['IsHoliday']==True],sales_holiday['Weekly_Sales'].loc[sales_holiday['IsHoliday']==False]]
labels=['Holiday','Not Holiday']

#median
medianprop={'color':'#2196F3',
            'linewidth': 2,
            'linestyle':'-'}

# outliers

flierprop={'color' : '#EC407A',
          'marker' : 'o',
          'markerfacecolor': '#2196F3',
          'markeredgecolor':'white',
          'markersize' : 3,
          'linestyle' : 'None',
          'linewidth' : 0.1}



axes[0].boxplot(target,labels=labels, patch_artist = 'Patch',
                  showmeans=True,
                  flierprops=flierprop,
                  medianprops=medianprop)

plt.title('Volume de vendas') 


axes[1].boxplot(target,labels=labels, patch_artist = 'Patch',
                  showmeans=True,
                  flierprops=flierprop,
                  medianprops=medianprop)

axes[1].set_ylim(-6000,80000)

plt.show()


# In[62]:


print(tt[tt['IsHoliday']==True]['Weekly_Sales'].describe().round(1))
print(tt[tt['IsHoliday']==False]['Weekly_Sales'].describe().round(1))


# In[63]:


data = pd.concat([tt['Mês'], tt['Weekly_Sales']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Mês', y="Weekly_Sales", data=data, showfliers=False)
plt.title('Relação vendas semanais por mes') 


# In[64]:


data = pd.concat([tt['Mês'], tt['Weekly_Sales'],tt['IsHoliday']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Mês', y="Weekly_Sales", data=data, showfliers=False, hue='IsHoliday')
plt.title('Relação entre vendas semanais, meses e feriados') 


# In[65]:


data = pd.concat([tt['Mês'], tt['Weekly_Sales'],tt['Type']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Mês', y="Weekly_Sales", data=data, showfliers=False, hue='Type')
plt.title('Relação entre vendas semanais, meses e tipos de lojas')


# In[66]:


data = pd.concat([tt['Ano'], tt['Weekly_Sales']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Ano', y="Weekly_Sales", data=data, showfliers=False)
plt.title('Relação entre vendas semanais e Anos')


# In[67]:


data = pd.concat([tt['Semana'], tt['Weekly_Sales']], axis=1)
f, ax = plt.subplots(figsize=(20, 6))
fig = sns.boxplot(x='Semana', y="Weekly_Sales", data=data, showfliers=False)
plt.title('Relação entre vendas semanais, Semanas')


# In[68]:


f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(tt['Weekly_Sales'])
plt.title('Distribuição de vendas semanais')


# In[69]:


f, ax = plt.subplots(figsize=(8, 6))
grouped=tt.groupby(['Dept','Weekly_Sales','IsHoliday']).mean().round(0).reset_index()
sns.distplot(grouped)
plt.title('Distribuição de vendas semanais por departamentto no feriado')


# In[70]:


print("Skewness: ", tt['Weekly_Sales'].skew()) #skewness
print("Kurtosis: ", tt['Weekly_Sales'].kurt()) #kurtosis
tt['Weekly_Sales'].min()


# In[71]:


fig.add_subplot(1,2,1)
res = stats.probplot(tt.loc[tt['Weekly_Sales']>0,'Weekly_Sales'], plot=plt)

fig.add_subplot(1,2,2)
res = stats.probplot(np.log1p(tt.loc[tt['Weekly_Sales']>0,'Weekly_Sales']), plot=plt)


# In[72]:


tt.describe()['Weekly_Sales']


# In[73]:


tt_over_zero=tt[tt['Weekly_Sales']>0]
tt_below_zero=tt[tt['Weekly_Sales']<=0]
sales_over_zero = np.log1p(tt_over_zero['Weekly_Sales'])
#histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(sales_over_zero)


# In[74]:


print("Skewness: ", sales_over_zero.skew()) #skewness
print("Kurtosis: ", sales_over_zero.kurt()) #kurtosis


# In[75]:


grouped=tt.groupby(['Dept','Date']).mean().round(0).reset_index()
print(grouped.shape)
print(grouped.head())
data=grouped[['Dept','Date','Weekly_Sales']]


dept=tt['Dept'].unique()
dept.sort()
dept_1=dept[0:20]
dept_2=dept[20:40]
dept_3=dept[40:60]
dept_4=dept[60:]

fig, ax = plt.subplots(2,2,figsize=(20,10))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

for i in dept_1 :
    data_1=data[data['Dept']==i]
    ax[0,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')

for i in dept_2 :
    data_1=data[data['Dept']==i]
    ax[0,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')
    
for i in dept_3 :
    data_1=data[data['Dept']==i]
    ax[1,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')    

for i in dept_4 :
    data_1=data[data['Dept']==i]
    ax[1,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')        
    
ax[0,0].set_title('Mean sales record by department(0~19)')
ax[0,1].set_title('Mean sales record by department(20~39)')
ax[1,0].set_title('Mean sales record by department(40~59)')
ax[1,1].set_title('Mean sales record by department(60~)')


ax[0,0].set_ylabel('Mean sales')
ax[0,0].set_xlabel('Date')
ax[0,1].set_ylabel('Mean sales')
ax[0,1].set_xlabel('Date')
ax[1,0].set_ylabel('Mean sales')
ax[1,0].set_xlabel('Date')
ax[1,1].set_ylabel('Mean sales')
ax[1,1].set_xlabel('Date')


plt.show()


# In[76]:


grouped=tt.groupby(['Store','Date']).mean().round(0).reset_index()
grouped.shape
grouped.head()

data=grouped[['Store','Date','Weekly_Sales']]
type(data)


store=tt['Store'].unique()
store.sort()
store_1=store[0:5]
store_2=store[5:10]
store_3=store[10:15]
store_4=store[15:20]
store_5=store[20:25]
store_6=store[25:30]
store_7=store[30:35]
store_8=store[35:40]
store_9=store[40:]

fig, ax = plt.subplots(5,2,figsize=(20,15))

fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

for i in store_1 :
    data_1=data[data['Store']==i]
    ax[0,0].plot(data_1['Date'], data_1['Weekly_Sales'])
    
for i in store_2 :
    data_2=data[data['Store']==i]
    ax[0,1].plot(data_2['Date'], data_2['Weekly_Sales'])
    
for i in store_3 :
    data_3=data[data['Store']==i]
    ax[1,0].plot(data_3['Date'], data_3['Weekly_Sales'])

for i in store_4 :
    data_4=data[data['Store']==i]
    ax[1,1].plot(data_4['Date'], data_4['Weekly_Sales'])
    
for i in store_5 :
    data_5=data[data['Store']==i]
    ax[2,0].plot(data_5['Date'], data_5['Weekly_Sales'])  

for i in store_6 :
    data_6=data[data['Store']==i]
    ax[2,1].plot(data_6['Date'], data_6['Weekly_Sales'])  

for i in store_7 :
    data_7=data[data['Store']==i]
    ax[3,0].plot(data_7['Date'], data_7['Weekly_Sales'])      

for i in store_8 :
    data_8=data[data['Store']==i]
    ax[3,1].plot(data_8['Date'], data_8['Weekly_Sales'])     
    
for i in store_9 :
    data_9=data[data['Store']==i]
    ax[4,0].plot(data_9['Date'], data_9['Weekly_Sales'])     

    
ax[0,0].set_title('Mean sales record by store(0~4)')
ax[0,1].set_title('Mean sales record by store(5~9)')
ax[1,0].set_title('Mean sales record by store(10~14)')
ax[1,1].set_title('Mean sales record by store(15~19)')
ax[2,0].set_title('Mean sales record by store(20~24)')
ax[2,1].set_title('Mean sales record by store(25~29)')
ax[3,0].set_title('Mean sales record by store(30~34)')
ax[3,1].set_title('Mean sales record by store(35~39)')
ax[4,0].set_title('Mean sales record by store(40~)')



ax[0,0].set_ylabel('Mean sales')
ax[0,0].set_xlabel('Date')
ax[0,1].set_ylabel('Mean sales')
ax[0,1].set_xlabel('Date')
ax[1,0].set_ylabel('Mean sales')
ax[1,0].set_xlabel('Date')
ax[1,1].set_ylabel('Mean sales')
ax[1,1].set_xlabel('Date')
ax[2,0].set_ylabel('Mean sales')
ax[2,0].set_xlabel('Date')
ax[2,1].set_ylabel('Mean sales')
ax[2,1].set_xlabel('Date')
ax[3,0].set_ylabel('Mean sales')
ax[3,0].set_xlabel('Date')
ax[3,1].set_ylabel('Mean sales')
ax[3,1].set_xlabel('Date')
ax[4,0].set_ylabel('Mean sales')
ax[4,0].set_xlabel('Date')



plt.show()


# In[77]:


grouped=tt.groupby(['Store','Dept'])['Weekly_Sales'].max().reset_index()
grouped['Store']=grouped['Store'].astype(str)
grouped['Dept']=grouped['Dept'].astype(str)
grouped['Weekly_Sales']=grouped['Weekly_Sales'].astype(str)
grouped['key']=grouped['Weekly_Sales'] +'_'+ grouped['Store'] +'_'+ grouped['Dept']


tt['Store']=tt['Store'].astype(str)
tt['Dept']=tt['Dept'].astype(str)
tt['Weekly_Sales_2']=tt['Weekly_Sales'].astype(str)
tt['key']=tt['Weekly_Sales'].astype(str) +'_'+ tt['Store'].astype(str) +'_'+ tt['Dept'].astype(str)

train_2=pd.merge(tt, grouped['key'], how='inner', on='key' )
train_2['Date_2']=train_2['Mês'].astype(str) + '-' + train_2['Dia'].astype(str)

grouped_2=train_2.groupby(['Date_2','Store','Dept']).count().reset_index()
grouped_2.sort_values('Weekly_Sales',ascending=False,inplace=True)


# In[78]:


grouped_2['key_2']=grouped_2['Date_2'].astype(str) + grouped_2['Store'].astype(str) + grouped_2['Dept'].astype(str)
grouped_2['Count']=grouped_2['Weekly_Sales']
data=grouped_2[['key_2','Count']]

tt['Date_2']=tt['Mês'].astype(str) + '-' + tt['Dia'].astype(str)
tt['key_2']=tt['Date_2'].astype(str) + tt['Store'].astype(str) + tt['Dept'].astype(str)
train=pd.merge(tt, data, how='left', on='key_2' )
train.loc[train['Count'].isnull(),'Count']=0

#grouped_2['proportion']=grouped_2['Weekly_Sales']/sum(grouped_2['Store'])
#grouped_2['Count']=grouped_2['Weekly_Sales']
#data=grouped_2[['Date_2','Count']]
#print(data.head(100))

#train['Date_2']=train['Month'].astype(str) + '-' + train['Day'].astype(str)

#train=pd.merge(train, data, how='left', on='Date_2' )
#train.head(150)


# In[79]:


grouped.head()


# In[80]:


train_2.head()


# In[81]:


grouped_2.head()


# In[82]:


train_2.head()


# In[83]:


data = pd.concat([train['Count'], train['Weekly_Sales'], train['Store']], axis=1)
f, ax = plt.subplots(figsize=(5, 5))
fig=sns.boxplot(x='Count', y="Weekly_Sales", data=data, showfliers=False)


# In[84]:


from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# In[85]:


tt.data = tt[['Store','Dept','Weekly_Sales','Size','Ano','Mês','Semana','Dia','n_dias']]
tt.targed = tt[['IsHoliday']]
tst.data = tst[['Store', 'Dept', 'Type','Size','Ano','Mês','Semana','Dia','n_dias']]
tst.targed = tst[['IsHoliday']]
#Convertendo features (objct-str) para float
tt.data['Store']= tt.data['Store'].apply(lambda x:(float(str(x))))
tt.data['Dept']= tt.data['Dept'].apply(lambda x:(float(str(x))))
tt.targed = np.where(tt.targed['IsHoliday']==False, 0, 1)
X_train = tt.data.as_matrix()
Y_train = tt.targed ## Verificar se é necessário transformar tudo em matriz
X_test = tst.data
Y_test = tst.targed


# In[86]:


sns.pairplot(tt[['Store','Dept','IsHoliday','Weekly_Sales']], hue='IsHoliday')


# In[87]:


model = svm.SVC(kernel='poly')
model.fit(X_train, Y_train)


# In[88]:


print(X_train)


# In[89]:


from matplotlib import style
style.use("seaborn-colorblind")
tt.data.plot(x='Dept', y='Weekly_Sales', c=tt.targed, kind='scatter',colormap='Accent_r')


# In[90]:


type(X_train)


# In[91]:


## predict case
sns.lmplot('X_test', 'Y_test', data=X_train, hue='Types', palette='Set1' )


# In[92]:


def acuracia(model,X_train,Y_train):
    resultados = cross_val_predict(model,X_train,Y_train, cv=5)
    return metrics.accuracy_score(Y_Train,resultados)
acuracia


# In[ ]:




