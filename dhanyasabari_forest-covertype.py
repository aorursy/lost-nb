#!/usr/bin/env python
# coding: utf-8



#This Python 3 environment comes with many helpful analytics libraries installed
#It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
#For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


#Input data files are available in the "../input/" directory.
#For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Any results you write to the current directory are saved as output.




df_train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
df_test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')




df_train.head()




#df_test1.head()




df_train.dtypes




pd.set_option('display.max_columns', None)
df_train.describe()




df_train = df_train.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)
df_test = df_test.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)




df_train = df_train.iloc[:,1:]
df_test = df_test.iloc[:,1:]




size = 10
corrmat = df_train.iloc[:, :size].corr()
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(corrmat, vmax = 0.8, square = True)




data = df_train.iloc[:, :size]
cols = data.columns
#Running pearson coefficient for all combinations
data_corr = data.corr()
threshold = 0.5
corr_list = []




data_corr




#sorting the highly correlated values
for i in range(0, size):
    for j in range(i+1, size):
        if data_corr.iloc[i, j] >= threshold and data_corr.iloc[i, j]<1        or data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j]<=-threshold:
            corr_list.append([data_corr.iloc[i,j],i,j])




#Sorting values
s_corr_list = sorted(corr_list, key = lambda x: -abs(x[0]))

#print the higher values
for v, i, j in s_corr_list:
    print("%s and %s = %.2f" % (cols[i], cols[j], v))




df_train.iloc[:, :10].skew()




for v, i, j in s_corr_list:
    sns.pairplot(data = df_train, hue = 'Cover_Type', size = 6, x_vars = cols[i], y_vars = cols[j])
    plt.show()




# A violin plot is a hybrid of a box plot and a kernel density plot, which shows peaks in the data.
cols = df_train.columns
size = len(cols) - 1 # We don't need the target attribute
# x-axis has target attributes to distinguish between classes
x = cols[size]
y = cols[0:size]

for i in range(0, size):
    sns.violinplot(data=df_train, x=x, y=y[i])
    plt.show()




df_train.Wilderness_Area2.value_counts()




### Group one-hot encoded variables of a category into one single variable
cols = df_train.columns
r,c = df_train.shape

# Create a new dataframe with r rows, one column for each encoded category, and target in the end
new_data = pd.DataFrame(index= np.arange(0,r), columns=['Wilderness_Area', 'Soil_Type', 'Cover_Type'])

# Make an entry in data for each r for category_id, target_value
for i in range(0,r):
    p = 0;
    q = 0;
    # Category1_range
    for j in range(10,14):
        if (df_train.iloc[i,j] == 1):
            p = j-9 # category_class
            break
    # Category2_range
    for k in range(14,54):
        if (df_train.iloc[i,k] == 1):
            q = k-13 # category_class
            break
            # Make an entry in data for each r
    new_data.iloc[i] = [p,q,df_train.iloc[i, c-1]]
    
# plot for category1
sns.countplot(x = 'Wilderness_Area', hue = 'Cover_Type', data = new_data)
plt.show()

# Plot for category2
plt.rc("figure", figsize = (25,10))
sns.countplot(x='Soil_Type', hue = 'Cover_Type', data= new_data)
plt.show()




# Checking the value count for different soil_types
for i in range(10, df_train.shape[1]-1):
    j = df_train.columns[i]
    print (df_train[j].value_counts())




df_train = df_train.drop(['Soil_Type8', 'Soil_Type25'], axis=1)
df_test = df_test.drop(['Soil_Type8', 'Soil_Type25'], axis=1)
df_train1 = df_train # To be used for algos like SVM where we need normalization and StandardScaler
df_test1 = df_test # To be used under normalization and StandardScaler




# Checking for data transformation (take only non-categorical values)
df_train.iloc[:,:10].skew()




from scipy import stats
plt.figure(figsize =(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Hydrology'], plot=plt)




df_train1['Horizontal_Distance_To_Hydrology'] = np.sqrt(df_train1['Horizontal_Distance_To_Hydrology'])




plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Hydrology'], plot=plt)




#Vertical_Distance_To_Hydrology
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Vertical_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Vertical_Distance_To_Hydrology'], plot=plt)




#Horizontal_Distance_To_Roadways
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Roadways'], fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Roadways'], plot=plt)




df_train1['Horizontal_Distance_To_Roadways'] = np.sqrt(df_train1['Horizontal_Distance_To_Roadways'])




# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Roadways'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Roadways'], plot=plt)




plt.figure(figsize=(8, 6))
sns.distplot(df_train1['Hillshade_9am'], fit=stats.norm)
plt.figure(figsize=(8, 6))
res = stats.probplot(df_train1['Hillshade_9am'], plot = plt)




df_train['Hillshade_9am'] = np.square(df_train1['Hillshade_9am'])




plt.figure(figsize = (8,6))
sns.distplot(df_train['Hillshade_9am'], fit = stats.norm)
fig = plt.figure(figsize = (8,6))
res = stats.probplot(df_train1['Hillshade_9am'], plot = plt)




# Hillshade_Noon
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_Noon'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_Noon'],plot=plt)




df_train1['Hillshade_Noon'] = np.square(df_train1['Hillshade_Noon'])




# Plot again after square transformation
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_Noon'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_Noon'],plot=plt)




# Horizontal_Distance_To_Fire_Points
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Fire_Points'], fit=stats.norm)
plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Fire_Points'],plot=plt)




df_train1['Horizontal_Distance_To_Fire_Points'] = np.sqrt(df_train1['Horizontal_Distance_To_Fire_Points'])




# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Fire_Points'], fit=stats.norm)
plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Fire_Points'],plot=plt)




# To be used in case of algorithms like SVM
df_test1[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Fire_Points'        ,'Horizontal_Distance_To_Roadways']] = np.sqrt(df_test1[['Horizontal_Distance_To_Hydrology',        'Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Roadways']])




# To be used in case of algorithms like SVM
df_test1[['Hillshade_9am','Hillshade_Noon']] = np.square(df_test1[['Hillshade_9am','Hillshade_Noon']])




from sklearn.preprocessing import StandardScaler




#non categorical variables only

Size = 10
X_train_temp = df_train.iloc[:,:Size]
X_test_temp = df_test.iloc[:,:Size]
X_train_temp1 = df_train1.iloc[:,:Size]
X_test_temp1 = df_test1.iloc[:,:Size]

X_train_temp1 = StandardScaler().fit_transform(X_train_temp1)
X_test_temp1 = StandardScaler().fit_transform(X_test_temp1)




df_train1.iloc[:,:]




r,c = df_train.shape
X_train = np.concatenate((X_train_temp,df_train.iloc[:,Size:c-1]),axis=1)
X_train1 = np.concatenate((X_train_temp1, df_train1.iloc[:,Size:c-1]), axis=1) # to be used for SVM
y_train = df_train.Cover_Type.values




from sklearn import svm
from sklearn.model_selection import train_test_split
#In the new version these are in the model_selection module. Use this: from sklearn.model_selection import learning_curve, GridSearchCV.
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV




x_data, x_test_data, y_data, y_test_data = train_test_split(X_train1,y_train,test_size=0.2, random_state=123)
svm_para = [{'kernel':['rbf'],'C': [1,10,100,100]}]




classifier = GridSearchCV(svm.SVC(),svm_para,cv=3,verbose=2)
classifier.fit(x_data,y_data)
classifier.best_params_
#classifier.best_score_




# Parameters optimized using the code in above cell
#C_opt = 10 # reasonable option
#clf = svm.SVC(C=C_opt,kernel='rbf')
#clf.fit(X_train1,y_train)
classifier.fit(X_train1,y_train)
classifier.score(X_train1,y_train)




classifier.best_score_




classifier.cv_results_




#y_pred = classifier.predict(df_test1)




df_Test1 = pd.read_csv('../input/forest-cover-type-prediction/test.csv')




#solution = pd.DataFrame({'Id':df_Test1.Id, 'Cover_Type':y_pred}, columns = ['Id','Cover_Type'])
#solution.to_csv('SVMcover_sol.csv', index=False)




from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
x_data, x_test_data, y_data, y_test_data = train_test_split(X_train,y_train,test_size= 0.3, random_state=0)
etc_para = [{'n_estimators': [20, 30, 100], 'max_depth':[5, 10, 15], 'max_features': [0.1, 0.2, 0.3]}]
#default number of features is sqrt(n)
#default number of min_samples_leaf is 1




ETC = GridSearchCV(ExtraTreesClassifier(),param_grid=etc_para, cv=10, n_jobs=-1)
ETC.fit(x_data, y_data)
ETC.best_params_
ETC.best_score_




print ('Best accuracy obtained: {}'.format(ETC.best_score_))
print ('Parameters:')
for key, value in ETC.best_params_.items():
    print('\t{}:{}'.format(key,value))




# Classification Report
Y_pred = ETC.predict(x_test_data)
target = ['class1', 'class2','class3','class4','class5','class6','class7' ]
print (classification_report(y_test_data, Y_pred, target_names=target))




from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(model,title, X, y,n_jobs = 1, ylim = None, cv = None,train_sizes = np.linspace(0.1, 1, 5)):
    
    # Figrue parameters
    plt.figure(figsize=(10,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    
    train_sizes, train_score, test_score = learning_curve(model, X, y, cv = cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # Calculate mean and std
    train_score_mean = np.mean(train_score, axis=1)
    train_score_std = np.std(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    test_score_std = np.std(test_score, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_score_mean - train_score_std, train_score_mean + train_score_std,                    alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes, test_score_mean - test_score_std, test_score_mean + test_score_std,                    alpha = 0.1, color = 'g')
    
    plt.plot(train_sizes, train_score_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_score_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc = "best")
    return plt




# 'max_features': 0.3, 'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf: 1'
etc = ExtraTreesClassifier(bootstrap=True, oob_score=True, n_estimators=100, max_depth=10, max_features=0.3,                            min_samples_leaf=1)

etc.fit(X_train, y_train)
# yy_pred = etc.predict(X_test)
etc.score(X_train, y_train)




r,c = df_test.shape
X_test = np.concatenate((X_test_temp, df_test.iloc[:,Size:c]), axis = 1)
yy_pred = etc.predict(X_test)
solution = pd.DataFrame({'Id':df_Test1.Id, 'Cover_Type':yy_pred}, columns = ['Id','Cover_Type'])
solution.to_csv('ETCcover_sol.csv', index=False)




# Plotting learning curve
title = 'Learning Curve (ExtraTreeClassifier)'
# cross validation with 50 iterations to have a smoother curve
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
model = etc
plot_learning_curve(model,title,X_train, y_train, n_jobs=-1,ylim=None,cv=cv)
plt.show()

