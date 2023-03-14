#!/usr/bin/env python
# coding: utf-8

# In[1]:


### This Notebook contains many models and predictions.
### The best performing submission was Gradient Boosting Algorithm which was my highest in leaders board.
### Best solution not submitted as i ran out of submission.
### This notebook creates my best sol using Gradient Boosting
### For csv of other models predictions remove # from makecsv method below model block.


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[3]:


df_tr = pd.read_csv('/kaggle/input/predict-the-housing-price/train.csv')
df_ts = pd.read_csv('/kaggle/input/predict-the-housing-price/Test.csv')
print("Train:",df_tr.shape)
print("Test:",df_ts.shape)


# In[4]:


df_tr.head(5)


# In[5]:


print('Train Dataset Infomarion')
print ("Rows     : " ,df_tr.shape[0])
print ("Columns  : " ,df_tr.shape[1])
print ("\nFeatures : \n" ,df_tr.columns.tolist())
print ("\nMissing values :  ",df_tr.isnull().sum().values.sum())
print ("\nUnique values :  \n",df_tr.nunique())


# In[6]:


df_ts.head(5)


# In[7]:


print('Test Dataset Infomarion')
print ("Rows     : " ,df_ts.shape[0])
print ("Columns  : " ,df_ts.shape[1])
print ("\nFeatures : \n" ,df_ts.columns.tolist())
print ("\nMissing values :  ",df_ts.isnull().sum().values.sum())
print ("\nUnique values :  \n",df_ts.nunique())


# In[8]:


### Train Null values
sns.heatmap(df_tr.isnull())


# In[9]:


# Train null list
nullist = []
nullist = df_tr.isnull().sum()
#nullist.loc[nullist != 0]
nul = pd.DataFrame(nullist.loc[nullist != 0])
nul


# In[10]:


# Numeric Nulls in Train
cols_tr = df_tr.columns
num_cols_tr= df_tr._get_numeric_data().columns
cat_cols_tr = list(set(cols_tr) - set(num_cols_tr))

sns.heatmap(df_tr[num_cols_tr].isnull())


# In[11]:


## Categorical nulls in Train
sns.heatmap(df_tr[cat_cols_tr].isnull())


# In[12]:


# Test null list
nullist1 = []
nullist1 = df_ts.isnull().sum()
#nullist.loc[nullist != 0]
nul1 = pd.DataFrame(nullist1.loc[nullist1 != 0])
nul1


# In[13]:


### Test Null values
sns.heatmap(df_ts.isnull())


# In[14]:


# Numeric Nulls in Test
cols = df_ts.columns
num_cols = df_ts._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))

sns.heatmap(df_ts[num_cols].isnull())


# In[15]:


### Categorical null cols in test
sns.heatmap(df_ts[cat_cols].isnull())


# In[16]:


### Droping cols with too many nulls
drop_columns = ['FireplaceQu','PoolQC','Fence','MiscFeature','BsmtUnfSF']
df_tr.drop(drop_columns, axis = 1, inplace = True)
df_ts.drop(drop_columns, axis = 1, inplace = True)


# In[17]:


cols = df_tr.columns
num_cols = df_tr._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[18]:


fill_col = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'GarageType','GarageFinish','GarageCond']
for i in fill_col:
    print(i,"values :\n",df_tr[i].value_counts())
    print("_____________________")


# In[19]:


### Categorical data
for i in cat_cols:
    print(i,"values :\n",df_tr[i].value_counts())
    print("_____________________")


# In[20]:


## Filling No where Nan in Categorical data
for col in df_tr[fill_col]:
    df_tr[col] = df_tr[col].fillna('None')
for col in df_ts[fill_col]:
    df_ts[col] = df_ts[col].fillna('None')


# In[21]:


colfil = ['BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars', 
            'GarageArea']
for coll in colfil:
    df_ts[coll].fillna(df_ts[coll].median(), inplace = True)


# In[22]:


num_cols = df_tr._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[23]:


df_tr['LotFrontage'].describe()


# In[24]:


(df_tr['LotFrontage'].plot.box()) 


# In[25]:


sns.violinplot(df_tr['LotFrontage'])


# In[26]:


### replace null with median as there are many outliers
df_tr['LotFrontage'].fillna(value=df_tr['LotFrontage'].median(),inplace=True)
df_ts['LotFrontage'].fillna(value=df_ts['LotFrontage'].median(),inplace=True)


# In[27]:


df_tr.GarageYrBlt.describe()


# In[28]:


(df_tr['GarageYrBlt'].plot.box()) 


# In[29]:


sns.violinplot(df_tr['GarageYrBlt'])


# In[30]:


### replace null with mean as there are many outliers
df_tr['GarageYrBlt'].fillna(value=df_tr['GarageYrBlt'].mean(),inplace=True)
df_ts['GarageYrBlt'].fillna(value=df_ts['GarageYrBlt'].mean(),inplace=True)


# In[31]:


df_tr['MasVnrArea'].describe()


# In[32]:


(df_tr['MasVnrArea'].plot.box()) 


# In[33]:


### replace null with median as there are many outliers
df_tr['MasVnrArea'].fillna(value=df_tr['MasVnrArea'].median(),inplace=True)
df_ts['MasVnrArea'].fillna(value=df_ts['MasVnrArea'].median(),inplace=True)


# In[34]:


#sns.heatmap(df_tr.isnull())
df_tr.isnull().sum()


# In[35]:


df_tr.columns


# In[36]:


### Creating some Featrues 
both_col = [df_tr, df_ts]
for col in both_col:
    col['YrBltAndRemod'] = col['YearBuilt'] + col['YearRemodAdd']
    col['TotalSF'] = col['TotalBsmtSF'] + col['1stFlrSF'] + col['2ndFlrSF']
    col['Total_sqr_footage'] = (col['BsmtFinSF1'] + col['BsmtFinSF2'] +
                                 col['1stFlrSF'] + col['2ndFlrSF'])

    col['Total_Bathrooms'] = (col['FullBath'] + (0.5 * col['HalfBath']) +
                               col['BsmtFullBath'] + (0.5 *col['BsmtHalfBath']))

    col['Total_porch_sf'] = (col['OpenPorchSF'] + col['3SsnPorch'] +
                              col['EnclosedPorch'] + col['ScreenPorch'] +
                              col['WoodDeckSF'])


# In[37]:


## Binary some feature
both_col = [df_tr, df_ts]
for col in both_col:
    col['haspool'] = col['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    col['has2ndfloor'] = col['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    col['hasgarage'] = col['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    col['hasbsmt'] = col['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    col['hasfireplace'] = col['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[38]:


plt.subplots(figsize=(30,30))
sns.heatmap(df_tr.corr(),cmap="GnBu",vmax=0.9, square=True)


# In[39]:


### droping some columns
drop_col = ['Exterior2nd','GarageYrBlt','Condition2','RoofMatl','Electrical','HouseStyle','Exterior1st',
            'Heating','GarageQual','Utilities','MSZoning','Functional','KitchenQual']
df_tr.drop(drop_col, axis = 1,inplace = True)
df_ts.drop(drop_col, axis = 1,inplace = True)


# In[40]:


df_tr


# In[41]:


df_ts


# In[42]:


cols = df_tr.columns
num_cols = df_tr._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[43]:


sns.heatmap(df_tr.isnull())


# In[44]:


sns.heatmap(df_ts.isnull())


# In[45]:


df_ts[cat_cols]


# In[46]:


df_tr[cat_cols]


# In[47]:


### value counts in categorical data in train
for i in df_tr[cat_cols]:
    print(i,":",len(df_tr[i].unique()))


# In[48]:


### value counts in categorical data in test
for i in df_ts[cat_cols]:
    print(i,":",len(df_ts[i].unique()))


# In[49]:


### LabelEncoding of categorical data


# In[50]:


from sklearn.preprocessing import LabelEncoder


# In[51]:


dftr = df_tr[cat_cols].apply(LabelEncoder().fit_transform)


# In[52]:


dfts = df_ts[cat_cols].apply(LabelEncoder().fit_transform)


# In[53]:


df_tr_final = df_tr[num_cols].join(dftr)


# In[54]:


num_cols = df_ts._get_numeric_data().columns
df_ts_final = df_ts[num_cols].join(dfts)


# In[55]:


df_tr_final


# In[56]:


df_ts_final


# In[ ]:





# In[57]:


ids = df_ts['Id']
df_tr_final.drop('Id',axis=1,inplace=True)
df_ts_final.drop('Id',axis=1,inplace=True)


# In[58]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import statsmodels.api as sm


# In[59]:


### SLR on all columns
for i in df_tr_final.columns:
    X = df_tr_final[[i]]#.values.reshape(1,-1)
    y = df_tr_final[['SalePrice']]#.values.reshape(1,-1)

    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=101)
    LR = LinearRegression()
    LR.fit(X_train,y_train)
    y_pred = LR.predict(X_test)
    print(i,"gives R2 score",r2_score(y_pred,y_test))
    print(i,'gives MSE is:',mean_squared_error(y_test, y_pred))
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    print(i,'gives RMSE is:',rms)
    print("------------------------------------------")
    #print('Coefficient is',LR.coef_[0][0])
    #print('intercept is',LR.intercept_[0])


# In[60]:


X = df_tr_final.drop('SalePrice',axis=1)#.values.reshape(1,-1)
y = df_tr_final['SalePrice']#.values.reshape(1,-1)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=101)


# In[61]:


### Using Rfe
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#X_train1 = scaler.fit_transform(X_train)
#y_train1 = scaler.fit_transform(y_train)
rfe = RFE(LR, 10)
rfe.fit(X_train,y_train)


# In[62]:


#rfe.support_


# In[63]:


X_train.columns[rfe.support_]


# In[64]:


cols = X_train.columns[rfe.support_]


# In[65]:


LR.fit(X_train[cols],y_train)


# In[66]:


y_pred = LR.predict(X_test[cols])
print("gives R2 score",r2_score(y_pred,y_test))
print('gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('gives RMSE is:',rms)
print("-----------------------------")


# In[67]:


y_pred = LR.predict(df_ts_final[cols])


# In[68]:


#For creating Output CSV file
def makecsv(y_pred,subno): ### input file name in ""
    subdf = pd.DataFrame()
    subdf['Id'] = df_ts['Id']
    subdf['SalePrice'] = y_pred
    subdf.to_csv(subno, index=False)


# In[69]:


# makecsv(y_pred,"rfesol.csv")


# In[70]:


import scipy.stats as stats


# In[71]:


stats.ttest_1samp(a=df_tr['OverallQual'],popmean=df_tr['SalePrice'].mean())


# In[72]:


model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# In[73]:


X = df_tr_final.drop('SalePrice',axis=1)#.values.reshape(1,-1)
y = df_tr_final['SalePrice']#.values.reshape(1,-1)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=101)


# In[74]:


### For using rfe selected features
#X_train = X_train[cols]
#X_test = X_test[cols]


# In[75]:


LR.fit(X_train,y_train)


# In[76]:


### Multiple Linear regression fo all
y_pred = LR.predict(X_test)
print("Multiple Linear regression gives R2 score",r2_score(y_pred,y_test))
print('Multiple Linear regression gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('Multiple Linear regression gives RMSE is:',rms)
print("-------------------------------------------")


# In[77]:


## Testing on Test Dataset
y_pred = LR.predict(df_ts_final)


# In[78]:


#makecsv(y_pred,"MLsol.csv")


# In[79]:


from sklearn.ensemble import RandomForestRegressor


# In[80]:


rf = RandomForestRegressor(n_estimators = 300, random_state = 0)
rf.fit(X_train,y_train)


# In[81]:


y_pred = rf.predict(X_test)


# In[82]:


print('all gives R2 score',r2_score(y_pred,y_test))
print('all gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('all gives RMSE is:',rms)
print("-----------------------------------------")


# In[83]:


## Testing on Test Dataset
y_pred = rf.predict(df_ts_final)


# In[84]:


#makecsv(y_pred,"Rfsol.csv")


# In[85]:


import xgboost as xgb


# In[86]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =42, nthread = -1)


# In[87]:


model_xgb.fit(X_train,y_train)


# In[88]:


y_pred = model_xgb.predict(X_test)
print('XGB score:',model_xgb.score(X_train,y_train))
print('all gives R2 score',r2_score(y_pred,y_test))
print('all gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('all gives RMSE is:',rms)
print("-----------------------------------------")


# In[89]:


## Testing on Test Dataset
y_pred = model_xgb.predict(df_ts_final)


# In[90]:


#makecsv(y_pred,"xgbsol.csv")


# In[91]:


from sklearn import ensemble


# In[92]:


GBoost = ensemble.GradientBoostingRegressor(n_estimators = 3000, max_depth = 5,max_features='sqrt',
                                            min_samples_split = 10,learning_rate = 0.005,loss = 'huber',
                                            min_samples_leaf=15,random_state =10)
GBoost.fit(X_train, y_train)


# In[93]:


y_pred = GBoost.predict(X_test)
print('GBosst score:',GBoost.score(X_train,y_train))
print('all gives R2 score',r2_score(y_pred,y_test))
print('all gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('all gives RMSE is:',rms)
print("-----------------------------------------")


# In[94]:


#Testing on Test Dataset
y_pred = GBoost.predict(df_ts_final)


# In[95]:


makecsv(y_pred,"gbsol.csv")

