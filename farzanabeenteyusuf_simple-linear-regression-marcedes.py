#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')




train_df.head()




test_df.head()




train_y = train_df['y']
train_x = train_df.drop(["y"], axis=1)
train_x= pd.get_dummies(train_x)
test_x = pd.get_dummies(test_df)
train_x.head()




from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=10)





regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)




print("Intercept: %.2f" %regr.intercept_)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_val) - y_val) ** 2))
# Explained variance score: 1 is perfect prediction
print('R-squared score: %.2f' % regr.score(X_val, y_val))




output = pd.DataFrame({'y': regr.predict(test_x)})
output['ID'] = test_x['ID']
output = output.set_index('ID')
output.to_csv('sub.csv')






