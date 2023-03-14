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


from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.xkcd()


# In[ ]:


print(os.listdir("../input/")) #展示所有文件的地址


# In[ ]:


train_applic=pd.read_csv("../input/application_train.csv")
test_applic=pd.read_csv("../input/application_test.csv")


# In[ ]:


train_applic.head()


# In[ ]:


test_applic.head()


# In[ ]:


train_applic.shape


# In[ ]:


test_applic.shape


# In[ ]:


train_applic["TARGET"].value_counts() #将目标数量统计处理


# In[ ]:


train_applic["TARGET"].plot.hist() #hist是将目标的频率统计出来


# In[ ]:


train_applic["TARGET"].astype(int).plot.hist() #hist是将目标的频率统计出来


# In[ ]:


train_applic["TARGET"].astype(int).value_counts()


# In[ ]:


#定义函数，用来查找数据是否存在缺失值
def find_missing_value(df):
    missing_value=df.isnull().sum() #计缺失值个数
    missing_value_percent=missing_value*100/len(df) #计算百分比
#     missing_value_percent=df.isnull().sum()*100/len(df) #计算百分比
    #将缺失值以及百分比放到一个相同的表格中用来统计
    missing_value_table=pd.concat([missing_value,missing_value_percent],axis=1)  
    missing_value_table=missing_value_table[missing_value_table.iloc[:,1]!=0].sort_values(by=[1],ascending=False)
    

    return missing_value_table


# In[ ]:


missing_values=find_missing_value(train_applic)
missing_values.head(20)


# In[ ]:


train_applic.dtypes.value_counts() #查找有多少种类


# In[ ]:


train_applic.select_dtypes("object").apply(pd.Series.nunique,axis=0)
#计算objects中每列有多少类。nunique返回不同值


# In[ ]:


'''
pandas.Series.nunique() return number of unique elements in the object
pandas.Series.unique() return unique values of Series object
'''

olecon=LabelEncoder()
label_count=0

for col in train_applic:
    if train_applic[col].dtype=="object":
        if len(list(train_applic[col].unique()))<=2:
            olecon.fit(train_applic[col])
            train_applic[col]=olecon.transform(train_applic[col])
            test_applic[col]=olecon.transform(test_applic[col])
            label_count +=1
print(label_count)
        


# In[ ]:


# one-hot-label 
train_applic=pd.get_dummies(train_applic)
test_applic=pd.get_dummies(test_applic)

# 使用get-dummies 将类型变量变成数值变量


# In[ ]:


train_applic.head(5)


# In[ ]:


train_applic.dtypes.value_counts()


# In[ ]:


train_applic.shape


# In[ ]:


test_applic.shape


# In[ ]:


train_labels=train_applic["TARGET"]


# In[ ]:


train_applic,test_applic=train_applic.align(test_applic,join="inner",axis=1)


# In[ ]:


train_applic.shape


# In[ ]:


train_applic.head(2)


# In[ ]:


test_applic.shape


# In[ ]:


test_applic.head(2)


# In[ ]:


train_applic["TARGET"]=train_labels


# In[ ]:


train_applic.head(2)


# In[ ]:


print(train_applic['DAYS_BIRTH'].head(),train_applic['DAYS_EMPLOYED'].head())  #生日是负数


# In[ ]:


print(train_applic['DAYS_ID_PUBLISH'].head(),train_applic['DAYS_REGISTRATION'].head())  #生日是负数


# In[ ]:


(train_applic["DAYS_BIRTH"]/-365).describe()


# In[ ]:


train_applic["DAYS_EMPLOYED"].head()


# In[ ]:


train_applic["DAYS_EMPLOYED"].describe() #最大超过100年


# In[ ]:


train_applic['DAYS_EMPLOYED'].plot.hist(title="工作时间")
plt.xlabel("days emplyment")


# In[ ]:


max_days_emp=train_applic[train_applic["DAYS_EMPLOYED"]==365243]
exclu_max_days_emp=train_applic[train_applic["DAYS_EMPLOYED"]!=365243]
exclu_max_days_emp["DAYS_EMPLOYED"].plot.hist()
#去掉最大值似乎好很多，同时最大值也是异常值的表现


# In[ ]:


train_applic['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)


# In[ ]:


train_applic['DAYS_EMPLOYED'].plot.hist()


# In[ ]:


test_applic['DAYS_EMPLOYED'].plot.hist()


# In[ ]:


test_applic['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)


# In[ ]:


test_applic['DAYS_EMPLOYED'].plot.hist()


# In[ ]:


train_applic.shape


# In[ ]:


test_applic.shape


# In[ ]:


# train_applic.isnull().sum().sort_values(ascending=False)


# In[ ]:


# test_applic.isnull().sum().sort_values(ascending=False)


# In[ ]:


correlations=train_applic.corr()['TARGET'].sort_values(ascending=False)


# In[ ]:


correlations.head(15)


# In[ ]:


train_applic['DAYS_BIRTH']=abs(train_applic['DAYS_BIRTH'])


# In[ ]:


train_applic['DAYS_BIRTH'].corr(train_applic['TARGET'])


# In[ ]:


train_applic['TARGET'].corr(train_applic['DAYS_BIRTH'])


# In[ ]:


train_applic[['TARGET','DAYS_BIRTH']].head()


# In[ ]:


extra_data=train_applic[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
extra_data_corrs=extra_data.corr()


# In[ ]:


extra_data_corrs  #相关性矩阵


# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(extra_data_corrs)
plt.title("correlation heatmap")


# In[ ]:


poly_features_train=train_applic[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','TARGET']];
poly_features_test=test_applic[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]


# In[ ]:


poly_features_train.head()


# In[ ]:


poly_features_test.head()


# In[ ]:


from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='median')
poly_traget=poly_features_train['TARGET']
poly_features_train=poly_features_train.drop(columns=['TARGET'])

poly_features_train=imputer.fit_transform(poly_features_train)
poly_features_test=imputer.fit_transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=2)


# In[ ]:


poly_transformer.fit(poly_features_train)
poly_features_train=poly_transformer.transform(poly_features_train)
poly_features_test=poly_transformer.transform(poly_features_test)


# In[ ]:


poly_features_train.shape


# In[ ]:


poly_features_test.shape


# In[ ]:





# In[ ]:





# In[ ]:


poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'])[:16]


# In[ ]:


poly_features_train=pd.DataFrame(poly_features_train,columns=poly_transformer.get_feature_names(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))
poly_features_train['TARGET']=poly_traget

poly_corrs=poly_features_train.corr()['TARGET'].sort_values()


# In[ ]:


poly_corrs


# In[ ]:


poly_features_test=pd.DataFrame(poly_features_test,columns=poly_transformer.get_feature_names(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))


# In[ ]:


poly_features_train.shape


# In[ ]:


poly_features_test.shape


# In[ ]:


poly_features_train['SK_ID_CURR']=train_applic['SK_ID_CURR']
app_poly_features_train=train_applic.merge(poly_features_train,on="SK_ID_CURR",how='left')


# In[ ]:


poly_features_test['SK_ID_CURR']=test_applic['SK_ID_CURR']
app_poly_features_test=test_applic.merge(poly_features_test,on="SK_ID_CURR",how='left')


# In[ ]:


app_poly_features_train.shape


# In[ ]:


app_poly_features_test.shape


# In[ ]:


app_poly_features_train,app_poly_features_test=app_poly_features_train.align(app_poly_features_test,join='inner',axis=1)


# In[ ]:


app_poly_features_train.shape


# In[ ]:


app_poly_features_test.shape


# In[ ]:


app_poly_features_train.head()


# In[ ]:


app_poly_features_test.head()


# In[ ]:


train_applic_poly=app_poly_features_train
test_applic_poly=app_poly_features_test


# In[ ]:


train_applic_poly['TARGET']=poly_traget


# In[ ]:


train_applic_poly.head()


# In[ ]:





# In[ ]:


# train_applic_poly.isnull().sum()


# In[ ]:


# train_applic_poly.dtypes.value_counts()


# In[ ]:


# test_applic_poly.shape


# In[ ]:


# test_applic_poly.dtypes.value_counts()


# In[ ]:


# train_applic_poly.isnull().sum().sort_values(ascending=False)


# In[ ]:


train_applic_poly.head()


# In[ ]:


# train_applic_poly=imputer.fit_transform(train_applic_poly)
# test_applic_poly=imputer.fit_transform(test_applic_poly)


# In[ ]:


# train_applic_poly.head()


# In[ ]:


# test_applic_poly.shape


# In[ ]:


train_applic_poly.fillna(train_applic_poly.median(),inplace=True)


# In[ ]:


train_applic_poly.head()


# In[ ]:


test_applic_poly.fillna(test_applic_poly.median(),inplace=True)


# In[ ]:


train_applic_poly.shape


# In[ ]:


test_applic_poly.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler,Imputer


# In[ ]:


# train=train_applic_poly.copy()
# test=test_applic_poly.copy()


# In[ ]:


'''
归一化：
1.只对特征进行归一化
2.不对目标进行归一化
'''
taeget=train_applic_poly['TARGET']
train=train_applic_poly.drop(columns=['TARGET'])
test=test_applic_poly.copy()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


scaler=MinMaxScaler(feature_range=(0,1)) #归一化
scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)


# In[ ]:


train.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(C=0.0001)
log_reg.fit(train,train_labels)  #二分类时候真实label没有放进去


# In[ ]:


# log_reg_pred=log_reg.predict_proba(test)


# In[ ]:


log_reg_pred=log_reg.predict_proba(test)[:,1]

#train test 都变成了narry


# In[ ]:


submit=test_applic_poly[['SK_ID_CURR']]
submit['TARGET']=log_reg_pred


# In[ ]:


submit.to_csv('log_reg_baseline.csv',index=False)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier(n_estimators=100,random_state=50,verbose=1,n_jobs=-1)


# In[ ]:


random_forest.fit(train,train_labels)


# In[ ]:





# In[ ]:


train_labels


# In[ ]:


important_feature_values=random_forest.feature_importances_
features=list(train_applic_poly.drop(columns=['TARGET']))
feature_importances=pd.DataFrame({'feature':features,'importance':important_feature_values})

predictions=random_forest.predict_proba(test)[:,1]


# In[ ]:


submit=test_applic_poly[['SK_ID_CURR']]
submit['TARGET']=predictions
submit.to_csv('random_forest_baseline.csv',index=False)


# In[ ]:


feature_importances.sort_values(by='importance',ascending=False)


# In[ ]:


feature_importances.plot.bar()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




