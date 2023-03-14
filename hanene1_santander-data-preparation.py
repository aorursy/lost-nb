#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import date 
import missingno as msno
get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6) 


# In[ ]:


pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',5000)


# In[ ]:


#predefine some of the data type, for memory efficiency 
type_dict={'ncodpers':np.int32, 'ind_ahor_fin_ult1':np.uint8, 'ind_aval_fin_ult1':np.uint8, 
       'ind_cco_fin_ult1':np.uint8,'ind_cder_fin_ult1':np.uint8,
            'ind_cno_fin_ult1':np.uint8,'ind_ctju_fin_ult1':np.uint8,'ind_ctma_fin_ult1':np.uint8,
            'ind_ctop_fin_ult1':np.uint8,'ind_ctpp_fin_ult1':np.uint8,'ind_deco_fin_ult1':np.uint8,
            'ind_deme_fin_ult1':np.uint8,'ind_dela_fin_ult1':np.uint8,'ind_ecue_fin_ult1':np.uint8,
            'ind_fond_fin_ult1':np.uint8,'ind_hip_fin_ult1':np.uint8,'ind_plan_fin_ult1':np.uint8,
            'ind_pres_fin_ult1':np.uint8,'ind_reca_fin_ult1':np.uint8,'ind_tjcr_fin_ult1':np.uint8,
            'ind_valo_fin_ult1':np.uint8,'ind_viv_fin_ult1':np.uint8,
            'ind_recibo_ult1':np.uint8 }
df = pd.read_csv("../input/train_ver2.csv", nrows=5000000, dtype=type_dict,)
df_test = pd.read_csv("../input/test_ver2.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


datatypesDF = pd.DataFrame(df.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sns.barplot(data=datatypesDF,x="variableType",y="count",ax=ax,color="#34495e")
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")


# In[ ]:


df.isnull().sum()


# In[ ]:


missingValuesColumns= df.columns[df.isnull().any()].tolist()
msno.bar(df[missingValuesColumns],            figsize=(20,8),color="blue",fontsize=12,labels=True,)


# In[ ]:


msno.matrix(df[missingValuesColumns],width_ratios=(10,1),            figsize=(16,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)


# In[ ]:


msno.heatmap(df[missingValuesColumns],figsize=(20,20))


# In[ ]:


msno.dendrogram(df)


# In[ ]:


missing_values= df.isnull().sum()


# In[ ]:


# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing= missing_values.sum()
# percent of data that is missing
(total_missing/total_cells) * 100


# In[ ]:


df = df.drop(['conyuemp','ult_fec_cli_1t'], axis=1)


# In[ ]:


df['fecha_dato']= pd.to_datetime(df.fecha_dato, format='%Y-%m-%d')
df['fecha_alta']= pd.to_datetime(df.fecha_alta , format='%Y-%m-%d')
df['fecha_dato'].unique()


# In[ ]:


df['age']=pd.to_numeric(df.age,errors='coerce')
df['renta']= pd.to_numeric(df.renta, errors ='coerce')
df["antiguedad"]   = pd.to_numeric(df.antiguedad , errors="coerce") 
df["indrel_1mes"]   = pd.to_numeric(df.indrel_1mes , errors="coerce") 


# In[ ]:


df["Month"]= pd.DatetimeIndex(df["fecha_dato"]).month


# In[ ]:


# Add a new column of the total number of products per customer per month
df["tot_products"] = df.loc[:,"ind_ahor_fin_ult1":"ind_recibo_ult1"].sum(axis=1)
df["tot_products"]   = pd.to_numeric(df["tot_products"], errors="coerce") 


# In[ ]:


df['age'].hist(bins=50)
plt.title("Customers' Age Distribution")
plt.xlabel("Age(years)")
plt.ylabel("Number of customers") 


# In[ ]:


df.info()


# In[ ]:


df_tot= df.groupby(['age'])['tot_products'].agg('sum')


# In[ ]:


df_tot.sort_values(ascending=False).head(20)


# In[ ]:


#Number of customers in the train set
len(set(df.ncodpers.unique()))


# In[ ]:


#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(17, 12))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


#correlation renta matrix
k=10
cols= corrmat.nlargest(k, 'renta')['renta'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)


# In[ ]:


drop_column = ['ind_nuevo','indrel','indresi','indfall','tipodom','ind_empleado','pais_residencia','indrel_1mes','indext','fecha_alta','tiprel_1mes']
df.drop(drop_column, axis=1, inplace = True)


# In[ ]:


from tqdm import tqdm


# In[ ]:


prov_renta=df[['renta','nomprov']].groupby('nomprov').mean()
def correct_renta (x):
    if (df['renta'][x]==None): return prov_renta.loc[prov_renta.index==df['nomprov'][x]]['renta'][0]
    else: return df['renta'][x]

s=[]
for x in tqdm(range(len(df))):
    s.append(correct_renta(x))
df['renta']=s


# In[ ]:


missingValuesColumns= df.columns[df.isnull().any()].tolist()
msno.bar(df[missingValuesColumns],            figsize=(20,8),color="blue",fontsize=12,labels=True,)


# In[ ]:


seg_age=df[['age','segmento']].groupby('segmento').mean()
def correct_age (x):
    if (df['age'][x]==None): return seg_age.loc[seg_age.index==df['segmento'][x]]['age'][0]
    else: return df['age'][x]

s=[]
for x in tqdm(range(len(df))):
    s.append(correct_age(x))
df['age']=s


# In[ ]:


msno.matrix(df[missingValuesColumns],width_ratios=(10,1),            figsize=(17,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)


# In[ ]:


product_col = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
df[product_col]=df[product_col].fillna(0)


# In[ ]:


col=['age','ind_nuevo','indrel','tipodom','cod_prov','ind_actividad_cliente','renta', 'pais_residencia','sexo','ind_empleado','fecha_alta','indrel_1mes','tiprel_1mes','indresi','indext','canal_entrada','indfall','nomprov','segmento','antiguedad']
dict123={'age':40,'ind_nuevo':1,'indrel':1,'tipodom':1,'cod_prov':26,'ind_empleado':'N','ind_actividad_cliente':0,'renta':134254,'pais_residencia':'ES','sexo':'H','fecha_alta':'2014-07-28','indrel_1mes':1.0,'tiprel_1mes':'I','indresi':'S','indext':'N','canal_entrada':'KHE','indfall':'N','nomprov':'MADRID','segmento':'02 - PARTICULARES','antiguedad':8}
df[col]=df[col].fillna(dict123)


# In[ ]:


msno.matrix(df[missingValuesColumns],width_ratios=(10,1),            figsize=(17,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)


# In[ ]:


df_test["renta"]   = pd.to_numeric(df_test["renta"], errors="coerce")
unique_prov = df_test[df_test.cod_prov.notnull()].cod_prov.unique()
grouped = df_test.groupby("cod_prov")["renta"].median()

def impute_renta(df):      
    for cod in unique_prov:
        df.loc[df['cod_prov']==cod,['renta']] = df.loc[df['cod_prov']==cod,['renta']].fillna({'renta':grouped[cod]}).values
    df.renta.fillna(df_test["renta"].median(), inplace=True)
    
impute_renta(df)
impute_renta(df_test)


# In[ ]:


def drop_na(df):
 df.dropna(axis=0 , subset=['ind_actividad_cliente'], inplace= True)
drop_na(df)
    


# In[ ]:


train = pd.read_csv("../input/train_ver2.csv", usecols=["ncodpers"])


# In[ ]:


train_unique_customers = set(train.ncodpers.unique())
print("Number of customers: ", len(train_unique_customers))


# In[ ]:


num_occur = train.groupby('ncodpers').agg('size').value_counts()

plt.figure(figsize=(8,4))
sns.barplot(num_occur.index, num_occur.values, alpha=0.8)
plt.xlabel('Number of Occurrences of the customer', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.show()


# In[ ]:


del train_unique_customers


# In[ ]:


train = pd.read_csv('../input/train_ver2.csv', dtype='float16', 
                    usecols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 
                             'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                             'ind_viv_fin_ult1', 'ind_nomina_ult1',
                             'ind_nom_pens_ult1', 'ind_recibo_ult1'])


# In[ ]:


target_counts = train.astype('float64').sum(axis=0)
#print(target_counts)
plt.figure(figsize=(8,4))
sns.barplot(target_counts.index, target_counts.values, alpha=0.8)
plt.xlabel('Product Name', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


correlation=train[['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']].corr()>0.7


# In[ ]:


listes=[]
for i in correlation.columns:
    w=correlation.index[correlation[i] == True].tolist()
    tu=(i,w)
    listes.append(tu)


# In[ ]:


sns.heatmap(train[['ind_cno_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1']].corr())


# In[ ]:


train = pd.read_csv('../input/train_ver2.csv', usecols=['fecha_dato', 'fecha_alta'], parse_dates=['fecha_dato', 'fecha_alta'])
train['fecha_dato_yearmonth'] = train['fecha_dato'].apply(lambda x: (100*x.year) + x.month)
yearmonth = train['fecha_dato_yearmonth'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(yearmonth.index, yearmonth.values, alpha=0.8)
plt.xlabel('Year and month of observation', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


train = pd.read_csv('../input/train_ver2.csv', usecols=['age'])
train=train[train['age']!=' NA']
train[['age']] = train[['age']].apply(pd.to_numeric)


# In[ ]:


def age_cat(x):
    if int(x) <24:
        return('Youth')
    elif int(x) <55:
        return('adults')
    else:
        return('old')


# In[ ]:


train['age_cat']=train['age'].apply(lambda x:age_cat(x))


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='age_cat',data=train,palette='RdBu_r')


# In[ ]:


train['age'].hist(bins=50)
plt.title("Customers' Age Distribution")
plt.xlabel("Age(years)")
plt.ylabel("Number of customers") 


# In[ ]:


train = pd.read_csv('../input/train_ver2.csv', usecols=['canal_entrada'])


# In[ ]:


train["canal_entrada"].value_counts().plot(x=None, y=None, kind='pie') 


# In[ ]:


train = pd.read_csv('../input/train_ver2.csv', usecols=['nomprov','renta'])


# In[ ]:


train.renta.isnull().sum()


# In[ ]:


train.nomprov.unique()


# In[ ]:


train.loc[train.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"


# In[ ]:


train.loc[train.nomprov.isnull(),"nomprov"] = "UNKNOWN"


# In[ ]:


#train.loc[train.renta.notnull(),:].groupby("nomprov").agg([{"Sum":sum},{"Mean":mean}])
incomes = train.loc[train.renta.notnull(),:].groupby("nomprov").agg({"renta":{"MedianIncome":median}})
incomes.sort_values(by=("renta","MedianIncome"),inplace=True)
incomes.reset_index(inplace=True)
incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in train.nomprov.unique()],ordered=False)
incomes.head()


# In[ ]:




