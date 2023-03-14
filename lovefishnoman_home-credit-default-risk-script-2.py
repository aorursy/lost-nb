#!/usr/bin/env python
# coding: utf-8



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




from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.xkcd()




print(os.listdir("../input/")) #展示所有文件的地址




train_applic=pd.read_csv("../input/application_train.csv")
test_applic=pd.read_csv("../input/application_test.csv")




train_applic.head()




test_applic.head()




train_applic.shape




test_applic.shape




train_applic["TARGET"].value_counts() #将目标数量统计处理




train_applic["TARGET"].plot.hist() #hist是将目标的频率统计出来




train_applic["TARGET"].astype(int).plot.hist() #hist是将目标的频率统计出来




train_applic["TARGET"].astype(int).value_counts()




#定义函数，用来查找数据是否存在缺失值
def find_missing_value(df):
    missing_value=df.isnull().sum() #计缺失值个数
    missing_value_percent=missing_value*100/len(df) #计算百分比
#     missing_value_percent=df.isnull().sum()*100/len(df) #计算百分比
    #将缺失值以及百分比放到一个相同的表格中用来统计
    missing_value_table=pd.concat([missing_value,missing_value_percent],axis=1)  
    missing_value_table=missing_value_table[missing_value_table.iloc[:,1]!=0].sort_values(by=[1],ascending=False)
    

    return missing_value_table




missing_values=find_missing_value(train_applic)
missing_values.head(20)




train_applic.dtypes.value_counts() #查找有多少种类




train_applic.select_dtypes("object").apply(pd.Series.nunique,axis=0)
#计算objects中每列有多少类。nunique返回不同值




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
        




# one-hot-label 
train_applic=pd.get_dummies(train_applic)
test_applic=pd.get_dummies(test_applic)

# 使用get-dummies 将类型变量变成数值变量




train_applic.head(5)




train_applic.dtypes.value_counts()




train_applic.shape




test_applic.shape




train_labels=train_applic["TARGET"]




train_applic,test_applic=train_applic.align(test_applic,join="inner",axis=1)




train_applic.shape




train_applic.head(2)




test_applic.shape




test_applic.head(2)




train_applic["TARGET"]=train_labels




train_applic.head(2)




print(train_applic['DAYS_BIRTH'].head(),train_applic['DAYS_EMPLOYED'].head())  #生日是负数




print(train_applic['DAYS_ID_PUBLISH'].head(),train_applic['DAYS_REGISTRATION'].head())  #生日是负数




(train_applic["DAYS_BIRTH"]/-365).describe()




train_applic["DAYS_EMPLOYED"].head()




train_applic["DAYS_EMPLOYED"].describe() #最大超过100年




train_applic['DAYS_EMPLOYED'].plot.hist(title="工作时间")
plt.xlabel("days emplyment")




max_days_emp=train_applic[train_applic["DAYS_EMPLOYED"]==365243]
exclu_max_days_emp=train_applic[train_applic["DAYS_EMPLOYED"]!=365243]
exclu_max_days_emp["DAYS_EMPLOYED"].plot.hist()
#去掉最大值似乎好很多，同时最大值也是异常值的表现




train_applic['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)




train_applic['DAYS_EMPLOYED'].plot.hist()




test_applic['DAYS_EMPLOYED'].plot.hist()




test_applic['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)




test_applic['DAYS_EMPLOYED'].plot.hist()




train_applic.shape




test_applic.shape




# train_applic.isnull().sum().sort_values(ascending=False)




# test_applic.isnull().sum().sort_values(ascending=False)




correlations=train_applic.corr()['TARGET'].sort_values(ascending=False)




correlations.head(15)




train_applic['DAYS_BIRTH']=abs(train_applic['DAYS_BIRTH'])




train_applic['DAYS_BIRTH'].corr(train_applic['TARGET'])




train_applic['TARGET'].corr(train_applic['DAYS_BIRTH'])




train_applic[['TARGET','DAYS_BIRTH']].head()




extra_data=train_applic[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
extra_data_corrs=extra_data.corr()




extra_data_corrs  #相关性矩阵




plt.figure(figsize=(8,6))
sns.heatmap(extra_data_corrs)
plt.title("correlation heatmap")




poly_features_train=train_applic[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','TARGET']];
poly_features_test=test_applic[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]




poly_features_train.head()




poly_features_test.head()




from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='median')
poly_traget=poly_features_train['TARGET']
poly_features_train=poly_features_train.drop(columns=['TARGET'])

poly_features_train=imputer.fit_transform(poly_features_train)
poly_features_test=imputer.fit_transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=2)




poly_transformer.fit(poly_features_train)
poly_features_train=poly_transformer.transform(poly_features_train)
poly_features_test=poly_transformer.transform(poly_features_test)




poly_features_train.shape




poly_features_test.shape














poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'])[:16]




poly_features_train=pd.DataFrame(poly_features_train,columns=poly_transformer.get_feature_names(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))
poly_features_train['TARGET']=poly_traget

poly_corrs=poly_features_train.corr()['TARGET'].sort_values()




poly_corrs




poly_features_test=pd.DataFrame(poly_features_test,columns=poly_transformer.get_feature_names(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))




poly_features_train.shape




poly_features_test.shape




poly_features_train['SK_ID_CURR']=train_applic['SK_ID_CURR']
app_poly_features_train=train_applic.merge(poly_features_train,on="SK_ID_CURR",how='left')




poly_features_test['SK_ID_CURR']=test_applic['SK_ID_CURR']
app_poly_features_test=test_applic.merge(poly_features_test,on="SK_ID_CURR",how='left')




app_poly_features_train.shape




app_poly_features_test.shape




app_poly_features_train,app_poly_features_test=app_poly_features_train.align(app_poly_features_test,join='inner',axis=1)




app_poly_features_train.shape




app_poly_features_test.shape




app_poly_features_train.head()




app_poly_features_test.head()




train_applic_poly=app_poly_features_train
test_applic_poly=app_poly_features_test




train_applic_poly['TARGET']=poly_traget




train_applic_poly.head()









# train_applic_poly.isnull().sum()




# train_applic_poly.dtypes.value_counts()




# test_applic_poly.shape




# test_applic_poly.dtypes.value_counts()




# train_applic_poly.isnull().sum().sort_values(ascending=False)




train_applic_poly.head()




# train_applic_poly=imputer.fit_transform(train_applic_poly)
# test_applic_poly=imputer.fit_transform(test_applic_poly)




# train_applic_poly.head()




# test_applic_poly.shape




train_applic_poly.fillna(train_applic_poly.median(),inplace=True)




train_applic_poly.head()




test_applic_poly.fillna(test_applic_poly.median(),inplace=True)




train_applic_poly.shape




test_applic_poly.shape




from sklearn.preprocessing import MinMaxScaler,Imputer




# train=train_applic_poly.copy()
# test=test_applic_poly.copy()




'''
归一化：
1.只对特征进行归一化
2.不对目标进行归一化
'''
taeget=train_applic_poly['TARGET']
train=train_applic_poly.drop(columns=['TARGET'])
test=test_applic_poly.copy()




train.shape




test.shape




scaler=MinMaxScaler(feature_range=(0,1)) #归一化
scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)




train.shape




from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(C=0.0001)
log_reg.fit(train,train_labels)  #二分类时候真实label没有放进去




# log_reg_pred=log_reg.predict_proba(test)




log_reg_pred=log_reg.predict_proba(test)[:,1]

#train test 都变成了narry




submit=test_applic_poly[['SK_ID_CURR']]
submit['TARGET']=log_reg_pred




submit.to_csv('log_reg_baseline.csv',index=False)




from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier(n_estimators=100,random_state=50,verbose=1,n_jobs=-1)




random_forest.fit(train,train_labels)









train_labels




important_feature_values=random_forest.feature_importances_
features=list(train_applic_poly.drop(columns=['TARGET']))
feature_importances=pd.DataFrame({'feature':features,'importance':important_feature_values})

predictions=random_forest.predict_proba(test)[:,1]




submit=test_applic_poly[['SK_ID_CURR']]
submit['TARGET']=predictions
submit.to_csv('random_forest_baseline.csv',index=False)




feature_importances.sort_values(by='importance',ascending=False)




feature_importances.plot.bar()































