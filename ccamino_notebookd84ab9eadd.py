#!/usr/bin/env python
# coding: utf-8

# In[45]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import  train_test_split
from sklearn import metrics
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6)


# In[46]:


# Calibrate the number of rows to not crash the kernel 
limit_rows   = 2000000
df           = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str}, nrows=limit_rows)
# Format to datetime data,maybe the month of the year is important to purchase
df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
# Not very sure just to keep with unique_ids 'cause principal id is fecha_dato + ncodpers and if we
# eliminate repeated ncodpers we lose some important information.
#We take unique_ids and unique fecha_datos just to play forward.
unique_ids   = pd.Series(df["ncodpers"].unique())
unique_fecha_dato = df["fecha_dato"].unique()
#df.count()
#unique_ids.count()
#limit_people = 1e4
# unique_id    = unique_ids.sample(n=limit_people)
# df           = df[df.ncodpers.isin(unique_id)]
df.head()


# In[47]:


df.isnull().any()


# In[48]:


with sns.plotting_context("notebook",font_scale=2.0):
    sns.set_style("darkgrid")
    sns.distplot(df["age"].dropna(),
                 bins=80,
                 kde=False,
                 color="tomato")
    sns.plt.title("Age Distribution")
    plt.ylabel("Count")


# In[49]:



df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 90,"age"] = df.loc[(df.age >= 30) & (df.age <= 90),"age"].mean(skipna=True)
df["age"].fillna(df["age"].mean(),inplace=True)
df["age"] = df["age"].astype(int)


# In[50]:


with sns.plotting_context("notebook",font_scale=2.0):
    sns.set_style("darkgrid")
    sns.distplot(df["age"].dropna(),
                 bins=80,
                 kde=False,
                 color="tomato")
    sns.plt.title("Age Distribution")
    plt.ylabel("Count")
    plt.xlim((15,100))


# In[51]:


df["ind_nuevo"].isnull().sum()


# In[52]:


months_active = df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()
months_active.max()


# In[53]:


df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1


# In[54]:


df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
np.sum(df["antiguedad"].isnull())


# In[55]:


df.loc[df["antiguedad"].isnull(),"ind_nuevo"].describe()


# In[56]:


df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()
df.loc[df.antiguedad <0, "antiguedad"] = 0


# In[57]:


dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
df["fecha_alta"].describe()


# In[58]:


pd.Series([i for i in df.indrel]).value_counts()


# In[59]:


df.loc[df.indrel.isnull(),"indrel"] = 1


# In[60]:


df.drop(["tipodom"],axis=1,inplace=True)


# In[61]:


df["nomprov"].isnull().sum()


# In[62]:


df["cod_prov"].isnull().sum()


# In[63]:


unique_cod_prov = df["cod_prov"].unique()
unique_nomprov = df["nomprov"].unique()
df.loc[df.cod_prov.isnull(),"cod_prov"] = 0


# In[64]:


np.sum(df["ind_actividad_cliente"].isnull())
df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df["ind_actividad_cliente"].median()


# In[65]:


df.renta.isnull().sum()


# In[66]:


incomes = df.loc[df.renta.notnull(),:].groupby("nomprov").agg({"renta":{"MedianIncome":median}})
incomes.sort_values(by=("renta","MedianIncome"),inplace=True)
incomes.reset_index(inplace=True)
incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in df.nomprov.unique()],ordered=False)
incomes.head()


# In[67]:


with sns.axes_style({
        "axes.facecolor":   "blue",
        "axes.grid"     :    False,
        "figure.facecolor": "white"}):
    h = sns.factorplot(data=incomes,
                   x="nomprov",
                   y=("renta","MedianIncome"),
                   order=(i for i in incomes.nomprov),
                   size=6,
                   aspect=1.5,
                   scale=1.0,
                   color="#ffc400",
                   linestyles="None")
plt.xticks(rotation=90)
plt.tick_params(labelsize=10,labelcolor="black")#
plt.ylabel("Median Income",size=18,color="black")
plt.xlabel("City",size=18,color="black")
plt.title("Income Distribution by City",size=20,color="black")
plt.ylim(0,180000)
plt.yticks(range(0,180000,40000))


# In[68]:


grouped        = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
new_incomes    = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
new_incomes    = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
df.sort_values("nomprov",inplace=True)
df             = df.reset_index()
new_incomes    = new_incomes.reset_index()


# In[69]:


df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()
df.sort_values(by="fecha_dato",inplace=True)


# In[70]:


df.drop(["nomprov"],axis=1,inplace=True)


# In[71]:


df.ind_nomina_ult1.isnull().sum()


# In[72]:


df.ind_nom_pens_ult1.isnull().sum()


# In[73]:


df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0


# In[74]:


string_data = df.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
for col in missing_columns:
    print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))
del string_data


# In[75]:


df.loc[df.indfall.isnull(),"indfall"] = "N"
df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
df.tiprel_1mes = df.tiprel_1mes.astype("category")

# As suggested by @StephenSmith
map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

df.indrel_1mes.fillna("P",inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df.indrel_1mes = df.indrel_1mes.astype("category")


unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    df.loc[df[col].isnull(),col] = "UNKNOWN"


# In[76]:


df.isnull().any()


# In[77]:


feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in feature_cols:
    df[col] = df[col].astype(int)
    


# In[78]:



usecols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
       
df_train = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str}, nrows=limit_rows)

df_test = pd.read_csv("../input/test_ver2.csv",dtype={"sexo":str,
         "ind_nuevo":str,"ult_fec_cli_1t":str,"indext":str}, nrows=limit_rows)

# pd.read_csv('../input/train.csv', usecols=usecols)


df_train = df_train.drop_duplicates(['ncodpers'], keep='last')
df_test = df_test.drop_duplicates(['ncodpers'], keep='last')
df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)


# In[79]:


print(len(df_train['renta']))
print(len(df_train))


# In[80]:





# In[80]:




models = {}
id_preds = defaultdict(list)
ids = df_train['ncodpers'].values

#Delete in final model (Done before)
feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in feature_cols:
    df[col] = df[col].astype(int)
# -------------------------------------------------------------------------------------------#    
for c in df_train.filter(regex="ind_+.*ult.*").columns:
    if c != 'ncodpers':
        print(c)
        y_train = df_train[c]
        x_train = df_train[['renta','ncodpers']]#.drop([c, 'ncodpers'], 1)
        
        # For v2 Gradient Boosting
        #y_test = df_test[c]
        #x_test = df_test.drop([c, 'ncodpers'], 1)
        #model = XGBClassifier()
        #model.fit(x_train,y_train)
        #p_train = model.predict(x_test)[:,1] 
 
        
        
        #models[c] = model
        #for id, p in zip(ids, p_train):
        #    id_preds[id].append(p)
        #    
        #print(metrics.accuracy_score(y_test, p_train))
        #----------------------------------------------#
        
        # For v1 Logistic Regression
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        p_train = clf.predict_proba(x_train)[:,1]
        
        models[c] = clf
        for id, p in zip(ids, p_train):
            id_preds[id].append(p)
            
        print(roc_auc_score(y_train, p_train))
        #----------------------------------------------#


# In[81]:



# for each ncodpers bring back every active (1) product in a dict ncodper: list of active products
already_active = {}
for row in df_train.values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(df_train.columns[1:], row) if c[1] > 0]
    already_active[id] = active

# returns the names of the products in order of probabilities    
# id_preds is a list of id + probability vector of length 23 (predict value of #25 to #48 columns)
# and preds just keep the value of probability if the feature is not "already active"
#train_preds = {}
#for id, p in id_preds.items():
    # Here be dragons
    #preds = [i[0] for i in sorted([i for i in zip(df_train.columns[1:], p) if i[0] not in already_active[id]], key=lambda i:i [1], reverse=True)[:7]]
    #train_preds[id] = preds
   


# In[82]:


null


# In[88]:


df_test[df_test['renta']=='         NA']=0
df_test['renta'].describe()


# In[91]:


colsfinales = [ 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

test_preds = []
for row in df_test.values:
    df_test['added_products']=''
    for c in colsfinales:
        x_test = df_test[['renta','ncodpers']]
        
        p_train = models[c].predict_proba(x_test)[:,1]
        if(p_train[1]>0.5):
            df_test['added_products']=df_test['added_products']+' '+c


pd.DataFrame({'added_products': df_test['added_products'], 'ncodpers': df_test['ncodpers']}).to_csv(filename, index=False)


# In[84]:




