#!/usr/bin/env python
# coding: utf-8



import os                             # File management
import pandas as pd                   # Dataframe manipulation
import numpy as np                    # Mathematics operation
import string                         # String manipulation
import seaborn as sns                 # Data visualization with seaborn
import matplotlib.pyplot as plt       # Data visualization with matplotlib

# Outlier detection
import statsmodels.api as sm
import statsmodels.formula.api as smapi

# Cross validation and data partitioning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Grid-search
from sklearn.model_selection import GridSearchCV

# Evaluation metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_error

# Principal Component Analysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Data modelling
from sklearn.linear_model import LinearRegression  # Linear regression
from sklearn.linear_model import HuberRegressor    # Robust regression with Huber weight
from sklearn.tree import DecisionTreeRegressor     # Decision Tree Regression

# Ignore warnings
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)




for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




# Importing the data
train_data = pd.read_csv('/kaggle/input/diamonds-price/diamonds_train.csv')
test_data = pd.read_csv('/kaggle/input/diamonds-price/diamonds_test.csv')




train_data.info()




print('Dimension of training data:\n{}'.format(train_data.shape[0]),
      'rows and {}'.format(train_data.shape[1]),'columns')
train_data.isna().sum()




print('Dimension of testing data:\n{}'.format(test_data.shape[0]),
      'rows and {}'.format(test_data.shape[1]),'columns')
print('The response variable is price')
test_data.isna().sum()




train_data.head()




# Summary statistics of numerical variable
train_data.drop('id',axis=1).describe()




train_data[(train_data['x'] == 0) | (train_data['y'] == 0) | (train_data['z'] == 0)]




# Unusual data removing
train_data_clean = train_data[((train_data['x'] != 0) & (train_data['y'] != 0)) & (train_data['z'] != 0)]
train_data_clean.reset_index(drop=True)




print('Dimension of training data after unusual observation removing:\n{}'.format(train_data_clean.shape[0]),
      'rows and {}'.format(train_data_clean.shape[1]),'columns')




volume_diamond = (train_data_clean['x']*train_data_clean['y']*train_data_clean['z'])/3
lw_prop = train_data_clean['x']/train_data_clean['y']
train_data_clean = pd.concat([train_data_clean,
                              pd.Series(volume_diamond,name='volume'),
                              pd.Series(lw_prop,name='lw_prop')],axis=1)




# Summary statistics of numerical variable
for i in train_data_clean.select_dtypes('object').columns:
    print(train_data_clean.iloc[:][i].value_counts(),'\n')




corrmat = train_data_clean.corr() # For calculating correlation
f,ax = plt.subplots(figsize=(10,9))
sns.heatmap(round(corrmat,3),ax=ax,cmap='YlOrBr',linewidths = 0.1,annot=True)




# Listing variables that are numeric
numerical_var = list(train_data_clean.columns[train_data_clean.dtypes!=object])
numerical_var.remove('id')
numerical_var




fig,axes = plt.subplots(3,3,sharey=False) # Prepare the grids
fig.set_size_inches(12,9)                 # Set size of these grids
ax = axes.ravel()

# Create plots with looping
for i in range(len(numerical_var)):
    sns.scatterplot(x=numerical_var[i],y='price',data=train_data_clean,ax=ax[i])
    ax[i].set_xlabel(numerical_var[i],fontsize=12)
    ax[i].set_ylabel('Diamond Price',fontsize=12)

plt.tight_layout()
plt.figure()




cut_cats = ['Fair','Good','Very Good','Premium','Ideal']
data_describe_cut = pd.DataFrame()
for i in cut_cats:
    cat_describe = pd.Series(train_data_clean[train_data_clean['cut'] == i]['price'].describe(),name=i)
    data_describe_cut = pd.concat([data_describe_cut,cat_describe],axis=1,sort=False)
data_describe_cut




# Choose the size
plt.figure(figsize=(9,6))
# Create plot
fig = sns.boxplot(y='price',x='cut',hue='cut',data=train_data_clean,palette='husl',order=cut_cats,width=0.5,
                  dodge=False)
fig.get_legend().remove()                      # Remove legend
plt.xlabel('Diamond Cut')                      # Horizontal labels
plt.ylabel('Price')                            # Vertical labels
plt.title('Barplot of Diamond Price in Cuts')  # Title of figure
plt.show(fig)




col_cats = ['D','E','F','G','H','I','J']
data_describe_col = pd.DataFrame()
for i in col_cats:
    cat_describe = pd.Series(train_data_clean[train_data_clean['color'] == i]['price'].describe(),name=i)
    data_describe_col = pd.concat([data_describe_col,cat_describe],axis=1,sort=False)
data_describe_col




# Choose the size
plt.figure(figsize=(9,6))
# Create plot
fig = sns.boxplot(y='price',x='color',hue='color',data=train_data_clean,palette='husl',order=col_cats,width=0.5,
                  dodge=False)
fig.get_legend().remove()                        # Remove legend
plt.xlabel('Diamond color')                      # Horizontal labels
plt.ylabel('Price')                              # Vertical labels
plt.title('Barplot of Diamond Price in Colors')  # Title of figure
plt.show(fig)




clarity_cats = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
data_describe_clarity = pd.DataFrame()
for i in clarity_cats:
    cat_describe = pd.Series(train_data_clean[train_data_clean['clarity'] == i]['price'].describe(),name=i)
    data_describe_clarity = pd.concat([data_describe_clarity,cat_describe],axis=1,sort=False)
data_describe_clarity




# Choose the size
plt.figure(figsize=(9,6))
# Create plot
fig = sns.boxplot(y='price',x='clarity',hue='clarity',data=train_data_clean,palette='husl',order=clarity_cats,
                  width=0.5,dodge=False)
fig.get_legend().remove()                           # Remove legend
plt.xlabel('Diamond clarity')                       # Horizontal labels
plt.ylabel('Price')                                 # Vertical labels
plt.title('Barplot of Diamond Price in Clarity')    # Title of figure
plt.show(fig)




train_data_clean.drop(['depth','table','lw_prop'],axis=1,inplace=True)
train_data_clean.head()




# Fitting
regression = smapi.ols(formula='price~volume',data=train_data_clean).fit()
print(regression.summary())




# Bonferroni outlier test
test = regression.outlier_test()
print('Bad data points (bonf(p) < 0.05):')
test[test['bonf(p)'] < 0.05]




# Create list for any value which is not in index
indices = list(test[test['bonf(p)'] < 0.05].index)
not_in_indices = [x for x in train_data_clean.index if x not in indices]
# Outliers removing
train_data_clean = train_data_clean.loc[not_in_indices]




# Listing variables those are numeric
numerical_var = list(train_data_clean.columns[train_data_clean.dtypes!=object])
numerical_var.remove('id')
numerical_var




fig,axes = plt.subplots(3,2,sharey=False) # Prepare the grids
fig.set_size_inches(12,9)                 # Set size of these grids
ax = axes.ravel()

# Create plots with looping
for i in range(len(numerical_var)):
    sns.scatterplot(x=numerical_var[i],y='price',data=train_data_clean,ax=ax[i])
    ax[i].set_xlabel(numerical_var[i],fontsize=12)
    ax[i].set_ylabel('Diamond Price',fontsize=12)

plt.tight_layout()
plt.figure()




# Correlation after outliers removing
train_data_clean[numerical_var][:].corr()




train_data_transform = train_data_clean.copy()
# Data transformation
train_data_transform['price'] = np.log(train_data_transform['price'])
train_data_transform['carat'] = np.log(train_data_transform['carat'])
train_data_transform['volume'] = np.log(train_data_transform['volume'])




fig,axes = plt.subplots(3,2,sharey=False) # Prepare the grids
fig.set_size_inches(12,9)                 # Set size of these grids
ax = axes.ravel()

# Create plots with looping
for i in range(len(numerical_var)):
    sns.scatterplot(x=numerical_var[i],y='price',data=train_data_transform,ax=ax[i])
    ax[i].set_xlabel(numerical_var[i],fontsize=12)
    ax[i].set_ylabel('Diamond Price',fontsize=12)

plt.tight_layout()
plt.figure()




# Correlation after outliers removing
train_data_transform[numerical_var][:].corr()




train_data_transform.head()




# Create column of 'volume'
volume_diamond = (test_data['x']*test_data['y']*test_data['z'])/3
test_data = pd.concat([test_data,pd.Series(volume_diamond,name='volume')],axis=1)
test_data.head()




# The anomaly is also found in testing data
test_data[test_data['volume'] == 0]




# Handle anomalies data
test_data.loc[test_data['volume'] == 0,'volume'] = test_data['volume'].mean()




# Drop any columns of 'depth' and 'table'
test_data.drop(['depth','table'],axis=1,inplace=True)
test_data.head()




# Data transformation
test_data['carat'] = np.log(test_data['carat'])
test_data['volume'] = np.log(test_data['volume'])
test_data.head()




object_var = list(train_data_transform.loc[:,train_data_transform.dtypes == np.object].columns)
num_var = [x for x in list(train_data_transform.columns) if x not in object_var]
print('Categorical variables : ',object_var)
print('Numerical variables   : ',num_var)




# Create dummies
df_train_onehot = pd.DataFrame()
for i in object_var:
    df_train_onehot = pd.concat([df_train_onehot,pd.get_dummies(train_data_transform[i],prefix=i,drop_first=True)],
                                axis = 1)
# Data concatenating
df_train_final = pd.concat([train_data_transform.loc[:,num_var],df_train_onehot],axis=1)




df_train_final.head()




num_var.remove('price')




# Create dummies
df_test_onehot = pd.DataFrame()
for i in object_var:
    df_test_onehot = pd.concat([df_test_onehot,pd.get_dummies(test_data[i],prefix=i,drop_first=True)],axis = 1)
# Data concatenating

df_test_final = pd.concat([test_data.loc[:,num_var],df_test_onehot],axis=1)




df_test_final.head()




# Data partitioning - training data into training and validation
df_train_final = df_train_final.reset_index(drop=True)
X = df_train_final[df_train_final.columns[~df_train_final.columns.isin(['id','price'])]]
y = df_train_final['price']
# Training = 80% and validation = 20%
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
print('Dimension of training   :',X_train.shape)
print('Dimension of validation :',X_val.shape)




# Predictors of testing data
X_test = df_test_final[df_test_final.columns[~df_test_final.columns.isin(['id'])]]




X_train.columns




X_test.columns




RMSE = []
MAE = []
lm_model = LinearRegression()
cv = KFold(n_splits=10,random_state=42,shuffle=True)
i = 0
for train_index, test_index in cv.split(X_train):
    i += 1
    X_training,X_testing,y_training,y_testing = X_train.iloc[train_index],X_train.iloc[test_index],                                                y_train.iloc[train_index],y_train.iloc[test_index]
    # Fitting the model
    lm_model.fit(X_training,y_training)
    # Predicting the price
    y_pred = lm_model.predict(X_testing)
    # Evaluation
    RMSE_val = np.sqrt(mean_squared_error(np.exp(y_pred),np.exp(y_testing)))
    MAE_val = mean_absolute_error(np.exp(y_pred),np.exp(y_testing))
    RMSE.append(RMSE_val)
    MAE.append(MAE_val)
    print(f'RMSE in CV - {i}: {round(RMSE_val,6)}',f'and MAE: {round(MAE_val,6)}')
print('Average of RMSE: {}'.format(sum(RMSE)/len(RMSE)))
print('Average of MAE: {}'.format(sum(MAE)/len(MAE)))




print('Intercept:',lm_model.intercept_)
pd.DataFrame({'Variable':X.columns,'Intercept':lm_model.coef_})




# Fitting the model
lm_model.fit(X_train,y_train)




# Predict training data and validation data
y_pred_train = lm_model.predict(X_train)
y_pred_val = lm_model.predict(X_val)




# Model evaluation - training data
RMSE_train_linreg_no_pca = np.sqrt(mean_squared_error(np.exp(y_pred_train),np.exp(y_train)))
MAE_train_linreg_no_pca = mean_absolute_error(np.exp(y_pred_train),np.exp(y_train))
corr_train_linreg_no_pca = np.corrcoef(np.exp(y_pred_train),np.exp(y_train))[0,1]
# Model evaluation - testing data
RMSE_test_linreg_no_pca = np.sqrt(mean_squared_error(np.exp(y_pred_val),np.exp(y_val)))
MAE_test_linreg_no_pca = mean_absolute_error(np.exp(y_pred_val),np.exp(y_val))
corr_test_linreg_no_pca = np.corrcoef(np.exp(y_pred_val),np.exp(y_val))[0,1]

print('RMSE for training data                  :{}'.format(RMSE_train_linreg_no_pca))
print('RMSE for testing data                   :{}'.format(RMSE_test_linreg_no_pca))
print('MAE for training data                   :{}'.format(MAE_train_linreg_no_pca))
print('MAE for testing data                    :{}'.format(MAE_test_linreg_no_pca))
print('Pearson Correlation for training data   :{}'.format(corr_train_linreg_no_pca))
print('Pearson Correlation for testing data    :{}'.format(corr_test_linreg_no_pca))




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_train),y=np.exp(y_pred_train))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Linear Regression - Training No PCA')




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_val),y=np.exp(y_pred_val))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Linear Regression - Testing No PCA')




# Predict testing data
y_pred_test = lm_model.predict(X_test)
y_pred_test = np.exp(y_pred_test)




pred_linreg_no_pca = pd.DataFrame({'id':df_test_final.id,'price':y_pred_test})
pred_linreg_no_pca.head()




tree_model = DecisionTreeRegressor(random_state = 0)
tree_model.fit(X_train,y_train)




# Model evaluation - training data
RMSE_train_dc_baseline = np.sqrt(mean_squared_error(np.exp(tree_model.predict(X_train)),np.exp(y_train)))
MAE_train_dc_baseline = mean_absolute_error(np.exp(tree_model.predict(X_train)),np.exp(y_train))
corr_train_dc_baseline = np.corrcoef(np.exp(tree_model.predict(X_train)),np.exp(y_train))[0,1]
# Model evaluation - testing data
RMSE_test_dc_baseline = np.sqrt(mean_squared_error(np.exp(tree_model.predict(X_val)),np.exp(y_val)))
MAE_test_dc_baseline = mean_absolute_error(np.exp(tree_model.predict(X_val)),np.exp(y_val))
corr_test_dc_baseline = np.corrcoef(np.exp(tree_model.predict(X_val)),np.exp(y_val))[0,1]

print('RMSE for training data                  :{}'.format(RMSE_train_dc_baseline))
print('RMSE for testing data                   :{}'.format(RMSE_test_dc_baseline))
print('MAE for training data                   :{}'.format(MAE_train_dc_baseline))
print('MAE for testing data                    :{}'.format(MAE_test_dc_baseline))
print('Pearson Correlation for training data   :{}'.format(corr_train_dc_baseline))
print('Pearson Correlation for testing data    :{}'.format(corr_test_dc_baseline))




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_train),y=np.exp(tree_model.predict(X_train)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree Baseline - Training No PCA')




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_val),y=np.exp(tree_model.predict(X_val)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree Baseline - Testing No PCA')




# 10-fold cross validation
kfold = KFold(n_splits=10)
# List that contains metric evaluation of RMSE
train_score = []
val_score = []
# Choose max_depth using looping
depth_range = range(5,50)
for depth in depth_range:
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X_train,y_train)
    # Save the score of RMSE
    train_score.append(np.sqrt(mean_squared_error(np.exp(tree.predict(X_train)),np.exp(y_train))))
    val_score.append(np.sqrt(mean_squared_error(np.exp(tree.predict(X_val)),np.exp(y_val))))




# Show evaluation scores of looping
plt.figure(figsize=(9,6))
# Show scores in one window
plt.plot(depth_range,train_score,'o-')
plt.plot(depth_range,val_score,'o-')
# Show the legends
plt.legend(['Training RMSE Score','Validation RMSE Score'])
# Determine label for horizontal axis
plt.xlabel('max_depth',fontsize=12)
# Determine label for vertical axis
plt.ylabel('RMSE',fontsize=12)
# Show the plot
plt.show()




# Hyperparameters
max_depth = [int(x) for x in np.linspace(10,30,num=10)]
max_depth.append(None)
min_samples_split = [2,50,75,100,120,150]
min_samples_leaf = [2,50,75,100,120,150]

# parmeter yang disiapkan digabung dalam satu objek
param_grid = {'max_depth':max_depth
             ,'min_samples_split':min_samples_split
             ,'min_samples_leaf':min_samples_leaf}




# Model of Decision Tree Regression
tree = DecisionTreeRegressor()
# Conduct 10-fold CV
kfold = KFold(n_splits=10,random_state=42)
# Grid-search
grid_search_tree = GridSearchCV(tree, # Model
                                param_grid = param_grid, # Hyperparameters
                                scoring = ['neg_mean_absolute_error','neg_root_mean_squared_error'], # Scores
                                refit = 'neg_root_mean_squared_error', # Specific score for tunning
                                cv = kfold, # Cross validation method
                                return_train_score = True) # Return any scores in training data CV




# Hyperparameters tunning with training data
grid_search_tree.fit(X_train,y_train)




# Show scores in fitting models
pd.DataFrame(grid_search_tree.cv_results_)




# Best model
print('Best hyperparameters : \n',grid_search_tree.best_params_,'\n')
print('Best evaluation : \n',grid_search_tree.best_score_,'\n')
print('Best model of Decision Tree: \n',grid_search_tree.best_estimator_,'\n')




# model akhir berdasarkan fungsi GridSearchCV
tree_final = grid_search_tree.best_estimator_
tree_final.fit(X_train,y_train)




# Model evaluation - training data
RMSE_train_dc_no_pca = np.sqrt(mean_squared_error(np.exp(tree_final.predict(X_train)),np.exp(y_train)))
MAE_train_dc_no_pca = mean_absolute_error(np.exp(tree_final.predict(X_train)),np.exp(y_train))
corr_train_dc_no_pca = np.corrcoef(np.exp(tree_final.predict(X_train)),np.exp(y_train))[0,1]
# Model evaluation - testing data
RMSE_test_dc_no_pca = np.sqrt(mean_squared_error(np.exp(tree_final.predict(X_val)),np.exp(y_val)))
MAE_test_dc_no_pca = mean_absolute_error(np.exp(tree_final.predict(X_val)),np.exp(y_val))
corr_test_dc_no_pca = np.corrcoef(np.exp(tree_final.predict(X_val)),np.exp(y_val))[0,1]

print('RMSE for training data  :{}'.format(RMSE_train_dc_no_pca))
print('RMSE for testing data   :{}'.format(RMSE_test_dc_no_pca))
print('MAE for training data   :{}'.format(MAE_train_dc_no_pca))
print('MAE for testing data    :{}'.format(MAE_test_dc_no_pca))
print('Pearson Correlation for training data   :{}'.format(corr_train_dc_no_pca))
print('Pearson Correlation for testing data    :{}'.format(corr_test_dc_no_pca))




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_train),y=np.exp(tree_final.predict(X_train)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree - Training No PCA')




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_val),y=np.exp(tree_final.predict(X_val)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree - Testing No PCA')




# Predict testing data
y_pred_test = tree_final.predict(X_test)
y_pred_test = np.exp(y_pred_test)




pred_tree_no_pca = pd.DataFrame({'id':df_test_final.id,'price':y_pred_test})
pred_tree_no_pca.head()




object_var = list(train_data_transform.loc[:,train_data_transform.dtypes == np.object].columns)
num_var = [x for x in list(train_data_transform.columns) if x not in object_var]
num_var.remove('id')
num_var.remove('price')
print('Categorical variables : ',object_var)
print('Numerical variables   : ',num_var)




# Min-max scaler
scaler = MinMaxScaler()
scaler.fit(df_train_final[num_var])




len(df_train_final['price'])




# Min-max scaler of training data
df_scaled_train = scaler.transform(df_train_final[num_var])
df_scaled_train = pd.DataFrame(df_scaled_train,columns = num_var)
df_scaled_train = pd.concat([df_scaled_train,pd.Series(df_train_final['price']).reset_index(drop=True)],axis=1)
df_scaled_train.head()




df_scaled_train.describe()




# Determine whether it would belong to X or y
X = df_scaled_train[num_var]
y = df_scaled_train['price']
# Principal Component Analysis with n = 10
pca = PCA(n_components=5)
pca.fit(X)
X_pca = pca.transform(X)
X_pca




# Explained variance ratio of PCA
np.sum(pca.explained_variance_ratio_)




df_pca_train = pd.DataFrame(X_pca,columns=['pc'+str(i) for i in range(1,6)])
df_pca_train['price'] = y
df_pca_train.head()




# Pearson correlation
corrmat = df_pca_train.corr() # For calculating correlation
round(corrmat,2)




# Min-max scaler of testing data
df_scaled_test = scaler.transform(df_test_final[num_var])
df_scaled_test = pd.DataFrame(df_scaled_test,columns = num_var)
df_scaled_test.head()




df_scaled_test.describe()




# Determine whether it would belong to X or y
X = df_scaled_test
# PCA
X_pca = pca.transform(X)
X_pca




df_pca_test = pd.DataFrame(X_pca,columns=['pc'+str(i) for i in range(1,6)])
df_pca_test.head()




# Data partitioning - training data into training and validation
df_pca_train = df_pca_train.reset_index(drop=True)
X = df_pca_train[df_pca_train.columns[~df_pca_train.columns.isin(['price'])]]
y = df_pca_train['price']
# Training = 80% and validation = 20%
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
print('Dimension of training   :',X_train.shape)
print('Dimension of validation :',X_val.shape)




# Predictors of testing data
X_test = df_pca_test




RMSE = []
MAE = []
lm_model = LinearRegression()
cv = KFold(n_splits=10,random_state=42,shuffle=True)
i = 0
for train_index,test_index in cv.split(X_train):
    i += 1
    X_training,X_testing,y_training,y_testing = X_train.iloc[train_index],X_train.iloc[test_index],                                                y_train.iloc[train_index],y_train.iloc[test_index]
    # Fitting the model
    lm_model.fit(X_training,y_training)
    # Predicting the price
    y_pred = lm_model.predict(X_testing)
    # Evaluation
    RMSE_val = np.sqrt(mean_squared_error(np.exp(y_pred),np.exp(y_testing)))
    MAE_val = mean_absolute_error(np.exp(y_pred),np.exp(y_testing))
    RMSE.append(RMSE_val)
    MAE.append(MAE_val)
    print(f'RMSE in CV - {i}: {round(RMSE_val,6)}',f'and MAE: {round(MAE_val,6)}')
print('Average of RMSE: {}'.format(sum(RMSE)/len(RMSE)))
print('Average of MAE: {}'.format(sum(MAE)/len(MAE)))




print('Intercept:',lm_model.intercept_)
pd.DataFrame({'Variable':X.columns,'Intercept':lm_model.coef_})




# Fitting the model
lm_model.fit(X_train,y_train)




# Predict training data and validation data
y_pred_train = lm_model.predict(X_train)
y_pred_val = lm_model.predict(X_val)




# Model evaluation - training data
RMSE_train_linreg_pca = np.sqrt(mean_squared_error(np.exp(y_pred_train),np.exp(y_train)))
MAE_train_linreg_pca = mean_absolute_error(np.exp(y_pred_train),np.exp(y_train))
corr_train_linreg_pca = np.corrcoef(np.exp(y_pred_train),np.exp(y_train))[0,1]
# Model evaluation - testing data
RMSE_test_linreg_pca = np.sqrt(mean_squared_error(np.exp(y_pred_val),np.exp(y_val)))
MAE_test_linreg_pca = mean_absolute_error(np.exp(y_pred_val),np.exp(y_val))
corr_test_linreg_pca = np.corrcoef(np.exp(y_pred_val),np.exp(y_val))[0,1]

print('RMSE for training data                  :{}'.format(RMSE_train_linreg_pca))
print('RMSE for testing data                   :{}'.format(RMSE_test_linreg_pca))
print('MAE for training data                   :{}'.format(MAE_train_linreg_pca))
print('MAE for testing data                    :{}'.format(MAE_test_linreg_pca))
print('Pearson Correlation for training data   :{}'.format(corr_train_linreg_pca))
print('Pearson Correlation for testing data    :{}'.format(corr_test_linreg_pca))




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_train),y=np.exp(y_pred_train))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Linear Regression - Training PCA')




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_val),y=np.exp(y_pred_val))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Linear Regression - Testing PCA')




# Predict testing data
y_pred_test = lm_model.predict(X_test)
y_pred_test = np.exp(y_pred_test)




pred_linreg_pca = pd.DataFrame({'id':df_test_final.id,'price':y_pred_test})
pred_linreg_pca.head()




tree_model = DecisionTreeRegressor(random_state = 0)
tree_model.fit(X_train,y_train)




# Model evaluation - training data
RMSE_train_dc_baseline_pca = np.sqrt(mean_squared_error(np.exp(tree_model.predict(X_train)),np.exp(y_train)))
MAE_train_dc_baseline_pca = mean_absolute_error(np.exp(tree_model.predict(X_train)),np.exp(y_train))
corr_train_dc_baseline_pca = np.corrcoef(np.exp(tree_model.predict(X_train)),np.exp(y_train))[0,1]
# Model evaluation - testing data
RMSE_test_dc_baseline_pca = np.sqrt(mean_squared_error(np.exp(tree_model.predict(X_val)),np.exp(y_val)))
MAE_test_dc_baseline_pca = mean_absolute_error(np.exp(tree_model.predict(X_val)),np.exp(y_val))
corr_test_dc_baseline_pca = np.corrcoef(np.exp(tree_model.predict(X_val)),np.exp(y_val))[0,1]

print('RMSE for training data                  :{}'.format(RMSE_train_dc_baseline_pca))
print('RMSE for testing data                   :{}'.format(RMSE_test_dc_baseline_pca))
print('MAE for training data                   :{}'.format(MAE_train_dc_baseline_pca))
print('MAE for testing data                    :{}'.format(MAE_test_dc_baseline_pca))
print('Pearson Correlation for training data   :{}'.format(corr_train_dc_baseline_pca))
print('Pearson Correlation for testing data    :{}'.format(corr_test_dc_baseline_pca))




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_train),y=np.exp(tree_model.predict(X_train)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree Baseline - Training PCA')




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_val),y=np.exp(tree_model.predict(X_val)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree Baseline - Testing PCA')




# 10-fold cross validation
kfold = KFold(n_splits=10)
# List that contains metric evaluation of RMSE
train_score = []
val_score = []
# Choose max_depth using looping
depth_range = range(5,50)
for depth in depth_range:
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X_train,y_train)
    # Save the score of RMSE
    train_score.append(np.sqrt(mean_squared_error(np.exp(tree.predict(X_train)),np.exp(y_train))))
    val_score.append(np.sqrt(mean_squared_error(np.exp(tree.predict(X_val)),np.exp(y_val))))




# Show evaluation scores of looping
plt.figure(figsize=(9,6))
# Show scores in one window
plt.plot(depth_range,train_score,'o-')
plt.plot(depth_range,val_score,'o-')
# Show the legends
plt.legend(['Training RMSE Score','Validation RMSE Score'])
# Determine label for horizontal axis
plt.xlabel('max_depth',fontsize=12)
# Determine label for vertical axis
plt.ylabel('RMSE',fontsize=12)
# Show the plot
plt.show()




# Hyperparameters
max_depth = [int(x) for x in np.linspace(30,50,num=15)]
max_depth.append(None)
min_samples_split = [2,50,75,100,120,150]
min_samples_leaf = [2,50,75,100,120,150]

# parmeter yang disiapkan digabung dalam satu objek
param_grid = {'max_depth':max_depth
             ,'min_samples_split':min_samples_split
             ,'min_samples_leaf':min_samples_leaf}




# Model of Decision Tree Regression
tree = DecisionTreeRegressor()
# Conduct 10-fold CV
kfold = KFold(n_splits=10,random_state=42)
# Grid-search
grid_search_tree = GridSearchCV(tree, # Model
                                param_grid = param_grid, # Hyperparameters
                                scoring = ['neg_mean_absolute_error','neg_root_mean_squared_error'], # Scores
                                refit = 'neg_root_mean_squared_error', # Specific score for tunning
                                cv = kfold, # Cross validation method
                                return_train_score = True) # Return any scores in training data CV




# Hyperparameters tunning with training data
grid_search_tree.fit(X_train,y_train)




# Show scores in fitting models
pd.DataFrame(grid_search_tree.cv_results_)




# Best model
print('Best hyperparameters : \n',grid_search_tree.best_params_,'\n')
print('Best evaluation : \n',grid_search_tree.best_score_,'\n')
print('Best model of Decision Tree: \n',grid_search_tree.best_estimator_,'\n')




# model akhir berdasarkan fungsi GridSearchCV
tree_final = grid_search_tree.best_estimator_
tree_final.fit(X_train,y_train)




# Model evaluation - training data
RMSE_train_dc_pca = np.sqrt(mean_squared_error(np.exp(tree_final.predict(X_train)),np.exp(y_train)))
MAE_train_dc_pca = mean_absolute_error(np.exp(tree_final.predict(X_train)),np.exp(y_train))
corr_train_dc_pca = np.corrcoef(np.exp(tree_final.predict(X_train)),np.exp(y_train))[0,1]
# Model evaluation - testing data
RMSE_test_dc_pca = np.sqrt(mean_squared_error(np.exp(tree_final.predict(X_val)),np.exp(y_val)))
MAE_test_dc_pca = mean_absolute_error(np.exp(tree_final.predict(X_val)),np.exp(y_val))
corr_test_dc_pca = np.corrcoef(np.exp(tree_final.predict(X_val)),np.exp(y_val))[0,1]

print('RMSE for training data                  :{}'.format(RMSE_train_dc_pca))
print('RMSE for testing data                   :{}'.format(RMSE_test_dc_pca))
print('MAE for training data                   :{}'.format(MAE_train_dc_pca))
print('MAE for testing data                    :{}'.format(MAE_test_dc_pca))
print('Pearson Correlation for training data   :{}'.format(corr_train_dc_pca))
print('Pearson Correlation for testing data    :{}'.format(corr_test_dc_pca))




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_train),y=np.exp(tree_final.predict(X_train)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree - Training PCA')




plt.figure(figsize=(8,6))
sns.scatterplot(x=np.exp(y_val),y=np.exp(tree_final.predict(X_val)))
# Determine label for horizontal axis
plt.xlabel('Actual Price',fontsize=12)
# Determine label for vertical axis
plt.ylabel('Prediction Price',fontsize=12)
# Title
plt.title('Decision Tree - Testing PCA')




# Predict testing data
y_pred_test = tree_final.predict(X_test)
y_pred_test = np.exp(y_pred_test)




pred_tree_pca = pd.DataFrame({'id':df_test_final.id,'price':y_pred_test})
pred_tree_pca.head()




pred_tree_no_pca.to_csv('submission.csv',index=False)

