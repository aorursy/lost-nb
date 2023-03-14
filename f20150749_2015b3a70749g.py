#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv('../input/Regression\ Evaluative\ Lab/train.csv')
test = pd.read_csv('../input/Regression\ Evaluative\ Lab/test.csv')
label= train['AveragePrice']
train.drop(columns=['AveragePrice'], inplace=True)
train = train.append(test)
p = train.columns.tolist()
p.remove('id')
p.remove('type')
p.remove('year')
#p.remove('AveragePrice')
for f in p:
   # train[f] = np.log(train[f]+0.001)
#train['AveragePrice'] = np.log(train["AveragePrice"])


train.info()




train.head()




train = pd.get_dummies(columns=["year"],data=train)




# Compute the correlation matrix
corr = train.corr(method="kendall")
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 100, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=1, vmin = -1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

plt.show()




train.drop(columns=['Total Bags'], inplace=True)




from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  
#Instantiate the scaler
scaled_data = scaler.fit_transform(train)
#scaled_X_val = scaler.transform(X_val)




from sklearn.model_selection import train_test_split
from math import sqrt

scaled_X_train,scaled_X_val,y_train,y_val = train_test_split(scaled_data[:10000], label, test_size = 0.33, random_state = 42)




from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor




rfr = RandomForestRegressor(random_state= 42, n_jobs=4)
parameters = {'n_estimators':[80,100,140, 170,250],'max_features':[2,3,4,5,8,9]}
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
scorer = make_scorer(r2_score)
grid_obj = GridSearchCV(rfr,parameters,scoring=scorer, cv= 5, n_jobs=6)
grid_fit = grid_obj.fit(scaled_X_train,y_train)
best_clf = grid_fit.best_estimator_ 
y_pred = best_clf.predict(scaled_X_val)
acc_op = r2_score(y_true = y_val, y_pred = y_pred)*100
print("Accuracy score on optimized model:{}".format(acc_op))




scaled_X_train = scaled_data[:10000,:]
scaled_X_val = scaled_data[10000:,:]
y_train = label.tolist()
rfr = RandomForestRegressor(random_state= 42, n_jobs = 6)
parameters = {'n_estimators':[60,80,100,120, 140, 170, 250],'max_features':[2,3,4,5,6,8,9]}
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
scorer = make_scorer(r2_score)
grid_obj = GridSearchCV(rfr,parameters,scoring=scorer, cv= 5, n_jobs=6)
grid_fit = grid_obj.fit(scaled_X_train,y_train)
best_clf = grid_fit.best_estimator_
y_predt = best_clf.predict(scaled_X_val)
p = pd.Series(data=y_predt)
test = pd.read_csv('test.csv')
newd = pd.concat([test['id'],p], axis = 1)
newd.rename(index=str, columns={0:'AveragePrice'}, inplace = True)
newd.to_csv("submc.csv", index=False)




from sklearn.metrics import mean_squared_error
from math import sqrt
print('The mean absolute error on the validation data of the stacked model is {}'.format(round((mean_squared_error(y_val,y_pred)),7)))

