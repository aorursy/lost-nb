#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn. preprocessing import MinMaxScaler
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import plotly.graph_objects as go
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor




train_df = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', dtype={'Id':str})            .dropna().reset_index(drop=True) # Load train dataFrame
loading_df = pd.read_csv('../input/trends-assessment-prediction/loading.csv', dtype={'Id':str})




# Lets merge train df with loading_df
train_df = train_df.merge(loading_df, on='Id', how='left')
train_df.head() 




train_df.shape




X_train = train_df.iloc[:,6:]  #Train feature
target = train_df.iloc[:,1:6]   #Target feature
X_train.head()




x_train,x_val,y_train,y_val = train_test_split(X_train,target,test_size=0.33,shuffle=True) #Lets split the data




scaler = MinMaxScaler()
scaler.fit(x_train)  ##Fit on train set
x_train = scaler.transform(x_train)  ##transform train set
x_val = scaler.transform(x_val)   ##transform validaion set




x_train = pd.DataFrame(x_train,columns=X_train.columns)  ##Convert numpy into dataframe
x_val = pd.DataFrame(x_val,columns=X_train.columns)




loss_wt = [.3, .175, .175, .175, .175]  ##weight for each target variable in calculatting loss (Given)




def k_loss(weight,y_pred,y_true):   ##Lets define the loss function
    s = np.sum(np.abs(y_pred-y_true))/np.sum(y_true)
    return weight*s




#model =  LGBMRegressor(random_state=17)  #Lets define model for feature selection
#model = xgb.XGBRegressor(tree_method= 'gpu_hist') ##
model = DecisionTreeRegressor() #You can use your own model for feature selection




col = target.columns




loss_dict ={} #To keep the history of loss
best_feat_col ={} #Dictionary to store best feature for each target variable

for i in range(5): #Iterate for every target feature
    print("Selecting best feature subset for " +str(col[i])+".....")
    min_loss=1000
    store =[]
    best_feature=[]
    #best_feature
    for x in range(1,27): ##Iterate for x best subset among 26 feature
        sfs = SFS(model,k_features=x,forward=True,floating=False,scoring = 'neg_mean_squared_error',cv = 0) #For forward selection set forward to True
        sfs.fit(x_train, y_train.iloc[:,i])
        col_n = list((sfs.k_feature_names_))
        model.fit(x_train[col_n],y_train.iloc[:,i])
        loss = k_loss(loss_wt[i],y_val.iloc[:,i],model.predict(x_val[col_n]))
        if(loss<min_loss):
            min_loss=loss
            best_feature = col_n
        store.append(loss)
    best_feat_col[col[i]]= list(best_feature)
    loss_dict[col[i]]=store
            
    




fig = go.Figure(data=go.Scatter(x=list(range(1,27)), y=loss_dict['age']))
fig.update_layout(title='Count of best features vs Loss for "AGE"',
                   xaxis_title='Count of features',
                   yaxis_title='Loss')
fig.show()




fig = go.Figure(data=go.Scatter(x=list(range(1,27)), y=loss_dict['domain1_var1']))
fig.update_layout(title='Count of best features vs Loss for "domain1_var1"',
                   xaxis_title='Count of features',
                   yaxis_title='Loss')
fig.show()




fig = go.Figure(data=go.Scatter(x=list(range(1,27)), y=loss_dict['domain1_var2']))
fig.update_layout(title='Count of best features vs Loss for "domain1_var2"',
                   xaxis_title='Count of features',
                   yaxis_title='Loss')
fig.show()




fig = go.Figure(data=go.Scatter(x=list(range(1,27)), y=loss_dict['domain2_var1']))
fig.update_layout(title='Count of best features vs Loss for "domain2_var1"',
                   xaxis_title='Count of features',
                   yaxis_title='Loss')
fig.show()




fig = go.Figure(data=go.Scatter(x=list(range(1,27)), y=loss_dict['domain2_var2']))
fig.update_layout(title='Count of best features vs Loss for "domain2_var2"',
                   xaxis_title='Count of features',
                   yaxis_title='Loss')
fig.show()




best_feat_col['age']




best_feat_col['domain1_var1']




best_feat_col['domain1_var2']  ##Interesting result




best_feat_col['domain2_var1']




best_feat_col['domain2_var2']






