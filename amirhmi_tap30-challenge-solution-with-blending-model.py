#!/usr/bin/env python
# coding: utf-8



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




with open(r'/kaggle/input/data.txt', 'r') as file:
    dataset = file.readlines()
dataset = [data.strip().split(' ') for data in dataset]
dataset[:4]




time_period = int(dataset[0][0])
m, n = int(dataset[1][0]), int(dataset[1][1])
dataset = dataset[2:]




dict_train_features ,dict_test_features = [], []

for t in range(time_period):
    for i in range(m):
        for j in range(n):
            label = dataset[8*t + i][j]
            instance_features = {'m': i, 'n': j, 'hour': t%24, 'day': t//24, 'requests': int(label)}
            if label == '-1':
                dict_test_features.append(instance_features)
            else:
                dict_train_features.append(instance_features)




import pandas as pd

tap30 = pd.DataFrame(dict_train_features)
test_set = pd.DataFrame(dict_test_features).drop(['requests'], axis=1)




tap30.head()




tap30.info()




mean_requests = pd.DataFrame({'requests mean' : tap30.groupby(['m', 'n'])['requests'].mean()}).reset_index()
mean_requests.head()




import matplotlib.pyplot as plt

mean_requests.plot.scatter(x='n', y='m', c='requests mean',
                           colormap='jet', figsize=(12, 8))
plt.show()




mean_requests = pd.DataFrame({'requests mean' : tap30.groupby(['hour'])['requests'].mean()}).reset_index()

mean_requests.plot(figsize=(12, 8))
plt.show()




tap30['hour2'] = tap30['hour']**2
test_set['hour2'] = test_set['hour']**2




import seaborn as sns

corr = tap30.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()




from sklearn.model_selection import train_test_split

tap30_labels = tap30['requests']
X_train, X_test, y_train, y_test = train_test_split(tap30.drop(['requests'], axis=1),
                                                   tap30_labels, test_size=0.01, random_state=42)




from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()




from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [200, 400, 500],
    'max_depth' : [14, 16, 20],
}

cv_rf = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3)
cv_rf.fit(X_train, y_train)
cv_rf.best_params_




final_model = cv_rf.best_estimator_
final_model.fit(X_train, y_train)




from sklearn.metrics import mean_squared_error
import numpy as np

print("Train RMSE : ", np.sqrt(mean_squared_error(y_train, final_model.predict(X_train))))
print("Test RMSE : ", np.sqrt(mean_squared_error(y_test, final_model.predict(X_test))))




from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=15, n_estimators=200)
gbrt.fit(X_train, y_train)

errors = [np.sqrt(mean_squared_error(y_test, y_pred))
          for y_pred in gbrt.staged_predict(X_test)]

plt.plot(np.arange(200), errors)
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=10,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)




print("Train RMSE : ", np.sqrt(mean_squared_error(y_train, gbrt_best.predict(X_train))))
print("Test RMSE : ", np.sqrt(mean_squared_error(y_test, gbrt_best.predict(X_test))))




from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_reg = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=12), n_estimators=1000,
        learning_rate=0.1)
ada_reg.fit(X_train, y_train)




print("Train RMSE : ", np.sqrt(mean_squared_error(y_train, ada_reg.predict(X_train))))
print("Test RMSE : ", np.sqrt(mean_squared_error(y_test, ada_reg.predict(X_test))))




def blend_model(data):
    return (0.25 * ada_reg.predict(X_test) + 
          0.25 * gbrt_best.predict(X_test) + 
          0.5 * final_model.predict(X_test))




y_pred = blend_model(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))




y_pred = (0.25 * ada_reg.predict(test_set) + 
          0.25 * gbrt_best.predict(test_set) + 
          0.5 * final_model.predict(test_set))

test_prepared = test_set
test_prepared['hour'] = 24 * test_prepared['day'] + test_prepared['hour']
test_prepared = test_prepared.drop(['day', 'hour2'], axis=1)
test_prepared['id'] = test_prepared['hour'].map(str) + ':'+ test_prepared['m'].map(str) +                         ':' + test_prepared['n'].map(str)
test_prepared['demand'] = y_pred
test_prepared[['id','demand']].to_csv('result.csv',index=False)

