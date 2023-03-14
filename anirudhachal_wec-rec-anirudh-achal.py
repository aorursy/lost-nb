#!/usr/bin/env python
# coding: utf-8



# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import seaborn as sns
import tensorflow as tf
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor




# Function to load_data into np matrixs
def load_data_into_np(train_df, test_df):
    features = []
    label = []

    for column in train_df.columns:
        if column != 'O/P':
            features.append(column)
        else:
            label.append(column)

    X = train_df.loc[:, features].to_numpy().astype('float')
    y = train_df.loc[:, label].to_numpy().astype('float').ravel()
    X_predict = test_df.loc[:, :].to_numpy().astype('float')

    # Performing Test Train Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    return X, y, X_train, y_train, X_test, y_test, X_predict, train_df, test_df




# Function to load_data after new csv files have been saved
def load_data(train_file_name, test_file_name):
    # Mounting Google Drive
    drive.mount('/content/gdrive')
    # Importing Data
    train_df = pd.read_csv('gdrive/My Drive/wecrec2020/Data/' + train_file_name, index_col = 0)
    test_df = pd.read_csv('gdrive/My Drive/wecrec2020/Data/' + test_file_name, index_col = 0)

    features = []
    label = []

    for column in train_df.columns:
        if column != 'O/P':
            features.append(column)
        else:
            label.append(column)

    X = train_df.loc[:, features].to_numpy().astype('float')
    y = train_df.loc[:, label].to_numpy().astype('float').ravel()
    X_predict = test_df.loc[:, :].to_numpy().astype('float')

    # Performing Test Train Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    return X, y, X_train, y_train, X_test, y_test, X_predict, train_df, test_df




# Function to load_data with train / cv / test split after new csv files have been saved
def load_data_with_cv(train_file_name, test_file_name):
    # Mounting Google Drive
    drive.mount('/content/gdrive')
    # Importing Data
    train_df = pd.read_csv('gdrive/My Drive/wecrec2020/Data/' + train_file_name, index_col = 0)
    test_df = pd.read_csv('gdrive/My Drive/wecrec2020/Data/' + test_file_name, index_col = 0)

    features = []
    label = []

    for column in train_df.columns:
        if column != 'O/P':
            features.append(column)
        else:
            label.append(column)

    X = train_df.loc[:, features].to_numpy().astype('float')
    y = train_df.loc[:, label].to_numpy().astype('float').ravel()
    X_predict = test_df.loc[:, :].to_numpy().astype('float')

    # Performing Test Train Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    X_train, X_cv_temp, y_train, y_cv_temp = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)

    return X, y, X_train, y_train, X_test, y_test, X_cv_temp, y_cv_temp, X_predict, train_df, test_df




# Function to Test Model on random test train splits
def test_model(model, count, X_temp, y_temp):
    for i in range(count):
        # Test train split
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size = 0.2, random_state = i)

        # Fitting the model
        model.fit(X_train_temp, y_train_temp)
        
        # Making predictions and printing error
        y_predictions = np.maximum(model.predict(X_test_temp), np.ones(y_test_temp.shape[0]))
        
        print(f"Root Mean Square error of model Trial {i + 1}: ", mean_squared_error(y_predictions, y_test_temp, squared = False))




# Function to get submission file
def get_prediction_file(model, filename, num):
    # Making predictions
    test_X = test_df.loc[:, :].to_numpy().astype('float')

    # For XGB
    X_predict = xgb.DMatrix(test_X)
    predictions_y = (np.power(model.predict(X_predict), num))
    # predictions_y = np.power(np.maximum(model.predict(test_X), np.ones(test_X.shape[0])), num)

    # Setting up submission dataframe
    df_submission = pd.DataFrame({'Id' : test_df.index, 'PredictedValue' : predictions_y.ravel()})
    
    # Setting path
    submission_file_path = 'gdrive/My Drive/wecrec2020/SubmissionFiles/' + filename

    # write to the file
    df_submission.to_csv(submission_file_path, index = False)




# Function to evaluate a trained model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print('Model Performance')
    print('Root Mean Square Error: {:0.4f}'.format(mean_squared_error(predictions, test_labels, squared = False)))




# Function to compare the differences between prediction files
def compare_preditions(file1, file2):
    # Mounting Google Drive
    drive.mount('/content/gdrive')

    # Importing Data
    y1_df = pd.read_csv('gdrive/My Drive/wecrec2020/SubmissionFiles/' + file1, index_col = 0)
    y2_df = pd.read_csv('gdrive/My Drive/wecrec2020/SubmissionFiles/' + file2, index_col = 0)

    y1 = y1_df.to_numpy().astype('float').ravel()
    y2 = y2_df.to_numpy().astype('float').ravel()

    print(f"Root Mean Square error : ", mean_squared_error(y1, y2, squared = False))




# Function to plot Train, Crossvalidation vs number of epochs of iterations rmse
def plot_log(history, start):
    # summarize history for loss    
    plt.plot(history.history['loss'][start:])
    plt.plot(history.history['val_loss'][start:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'cross validation'], loc='upper left')
    plt.show()




# Function to plot Train, Crossvalidation vs number of epochs of iterations rmse for XGBoost
def plot_log_xgb(history, start):
    # summarize history for loss    
    plt.plot(history['validation_0']['rmse'][start:])
    plt.plot(history['validation_1']['rmse'][start:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'cross validation'], loc='upper left')
    plt.show()




# Importing Data
train_df = pd.read_csv('../input/wecrec2020/Train_data.csv', index_col = 0)
test_df = pd.read_csv('../input/wecrec2020/Test_data.csv', index_col = 0)
train_df.drop(['F1', 'F2'], inplace = True, axis = 1) 
test_df.drop(['F1', 'F2'], inplace = True, axis = 1) 




train_df.head()




test_df.head()




for column in train_df.columns:
    plt.scatter(train_df[column], train_df['O/P'], alpha = 0.1)
    plt.title(column)
    plt.show()




plt.hist(train_df['O/P'], bins = 100)
plt.show()




X, y, X_train, y_train, X_test, y_test, X_predict, train_df, test_df = load_data_into_np(train_df, test_df)




dummy_model = DummyRegressor(strategy = "mean")

test_model(dummy_model, 1, X, y)

# get_prediction_file(dummy_model, 'Dummy_model.csv', 1)




rf = RandomForestRegressor(n_estimators = 50, random_state = 43)

rf.fit(X, y)

test_model(rf, 1, X, y)

# get_prediction_file(rf, 'baseline.csv', 1)




lr_model_1 = LinearRegression()

test_model(lr_model_1, 1, X, y)

# get_prediction_file(dummy_model, 'Dummy_model.csv', 1)




clf = Lasso(alpha=10, max_iter=100000)

test_model(clf, 1, X, y)

# get_prediction_file(clf, 'lasso_1.0.csv', 1)




from sklearn.neural_network import MLPRegressor

regr_1 = MLPRegressor(random_state = 1, max_iter = 10000)

test_model(regr_1, 1, X, y)

# get_prediction_file(regr_1, 'nn_1.3.csv', 1)




# Create the parameter grid

param_grid = {
    'max_depth': [20, 30, 40],
    'n_estimators': [100, 110, 120,], 
    'min_samples_split': [2, 3, 4],
    'random_state' : [47],
}

# Create a based model
rf = RandomForestRegressor()

grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 10)

grid_search_rf.fit(X_train, y_train)

# Best Parameter Values
best_grid = grid_search_rf.best_estimator_
print(best_grid)

# Evaluating the model
evaluate(best_grid, X_test, y_test)

# Getting submission file
# get_prediction_file(best_grid, 'rfr_final.csv', 1)




param = {
    'max_depth': 10, 
    'eta': 10, 
    'objective': 'reg:squarederror', 
    'eval_metric' : 'rmse', 
    'n_estimators' : 400, 
    'reg_lambda' : 100, 
    'min_child_weight': 70, 
    'subsample' : 1, 
    'reg_alpha' : 50
    }

clf = XGBRegressor(**param)

clf.fit(X_train, y_train, 
        eval_set = [(X_train, y_train), (X_test, y_test)], 
        eval_metric = 'rmse', 
        verbose = True)

evals_result = clf.evals_result()

plot_log_xgb(evals_result, 20)
print(f"Root Mean Square error of model Trial : ", mean_squared_error(np.power(clf.predict(X_test), 1), np.power(y_test, 1), squared = False))
plt.hist(np.power(clf.predict(X_test), 1), bins = 50)
plt.show()




clf = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
clf.fit(X_train, y_train, 
        eval_set = [(X_train, y_train), (X_test, y_test)], 
        eval_metric = 'rmse', 
        verbose = True)

evals_result = clf.evals_result()

plot_log_xgb(evals_result, 20)
print(f"Root Mean Square error of model Trial : ", mean_squared_error(np.power(clf.predict(X_test), 1), np.power(y_test, 1), squared = False))
plt.hist(np.power(clf.predict(X_test), 1), bins = 50)
plt.show()




# Loading data in to DMatrix
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

# Create the parameter grid based
params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

num_boost_rounds = 10000

# Training on Default parameters
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round = num_boost_rounds,
    evals = [(dtest, "Test")],
    early_stopping_rounds = 10
)

print("Best RSME: {:.2f} with {} rounds".format(
                 xgb_model.best_score,
                 xgb_model.best_iteration+1))




# Testing cv 
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round = num_boost_rounds,
    seed = 42,
    nfold = 5,
    metrics = {'rmse'},
    early_stopping_rounds=10
)
cv_results




cv_results['test-rmse-mean'].min()




gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in np.arange(4, 10, 1)
    for min_child_weight in np.arange(1, 14, 2)
]

min_rmse = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed = 42,
        nfold = 5,
        metrics={'rmse'},
        early_stopping_rounds = 10
    )
    # Update best rmse

    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))




# Setting best parameters
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]




gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/20. for i in range(10,21)]
    for colsample in [i/20. for i in range(10,21)]
]

min_rmse = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(subsample, colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (subsample,colsample)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))




# Setting best parameters
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]




min_rmse = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, 0.02, 0.01]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_rounds,
            seed=42,
            nfold=5,
            metrics=['rmse'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds\n".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = eta
print("Best params: {}, RMSE: {}".format(best_params, min_rmse))




params['eta'] = best_params




gridsearch_params = [
    (reg_lambda)
    for reg_lambda in np.arange(1, 10)
]




min_rmse = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for reg_lambda in gridsearch_params:
    print("CV with reg_lambda={}".format(reg_lambda))
    # We update our parameters
    params['reg_lambda'] = reg_lambda

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = reg_lambda
print("Best params: {}, RMSE: {}".format(best_params, min_rmse))




params['reg_lambda'] = best_params




best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_rounds,
    evals=[(dtest, "Test")],
    early_stopping_rounds = 10
)

# get_prediction_file(xgb_model, 'XGB_final_1.csv', 1)




test_df['O/P'] = -1

# Concatinating train_f and test_df to process them simultaneously
df = pd.concat((train_df, test_df), axis = 0)




df = df.astype('float64') 
df.info()




# calculate the correlation matrix
corr = df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)




correlated_features = set()
correlation_matrix = df.corr()

removed_columns = []

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            print(i, j)
            correlated_features.add(colname)
            if colname not in removed_columns:
                removed_columns.append(colname)

df.drop(labels=correlated_features, axis=1, inplace=True)




print(correlated_features)
print(removed_columns)
# calculate the correlation matrix
corr = df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)




print(removed_columns)
df.info()




train_df = df.loc[0:(14000-1), :]
test_df = df.loc[14000:, :]
test_df.drop('O/P', inplace = True, axis = 1)




train_df.info()




test_df.info()




# save_files('Train_data_def_2.csv', 'Test_data_def_2.csv')
X, y, X_train, y_train, X_test, y_test, X_predict, train_df, test_df = load_data_into_np(train_df, test_df)




# Create the parameter grid

param_grid = {
    'max_depth': [10, 20, 30],
    'n_estimators': [90, 110, 120], 
    'min_samples_split': [1, 2, 3, 4, 5],
    'random_state' : [47],
}

# Create a based model
rf = RandomForestRegressor()

grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 10)

grid_search_rf.fit(X_train, y_train)

# Best Parameter Values
best_grid = grid_search_rf.best_estimator_
print(best_grid)

# Evaluating the model
evaluate(best_grid, X_test, y_test)

# Getting submission file
# get_prediction_file(best_grid, 'rfr_final_2.csv', 1)




param = {
    'max_depth': 10, 
    'eta': 10, 
    'objective': 'reg:squarederror', 
    'eval_metric' : 'rmse', 
    'n_estimators' : 400, 
    'reg_lambda' : 100, 
    'min_child_weight': 70, 
    'subsample' : 1, 
    'reg_alpha' : 50
    }

clf = XGBRegressor(**param)

clf.fit(X_train, y_train, 
        eval_set = [(X_train, y_train), (X_test, y_test)], 
        eval_metric = 'rmse', 
        verbose = True)

evals_result = clf.evals_result()

plot_log_xgb(evals_result, 20)
print(f"Root Mean Square error of model Trial : ", mean_squared_error(np.power(clf.predict(X_test), 1), np.power(y_test, 1), squared = False))
plt.hist(np.power(clf.predict(X_test), 1), bins = 50)
plt.show()




clf = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
clf.fit(X_train, y_train, 
        eval_set = [(X_train, y_train), (X_test, y_test)], 
        eval_metric = 'rmse', 
        verbose = True)

evals_result = clf.evals_result()

plot_log_xgb(evals_result, 20)
print(f"Root Mean Square error of model Trial : ", mean_squared_error(np.power(clf.predict(X_test), 1), np.power(y_test, 1), squared = False))
plt.hist(np.power(clf.predict(X_test), 1), bins = 50)
plt.show()




# Loading data in to DMatrix
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

# Create the parameter grid based
params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

num_boost_rounds = 10000

xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round = num_boost_rounds,
    evals = [(dtest, "Test")],
    early_stopping_rounds = 10
)

print("Best RSME: {:.2f} with {} rounds".format(
                 xgb_model.best_score,
                 xgb_model.best_iteration+1))




# Testing cv 
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round = num_boost_rounds,
    seed = 42,
    nfold = 5,
    metrics = {'rmse'},
    early_stopping_rounds=10
)
cv_results




cv_results['test-rmse-mean'].min()




gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in np.arange(4, 10, 1)
    for min_child_weight in np.arange(1, 14, 2)
]

min_rmse = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed = 42,
        nfold = 5,
        metrics={'rmse'},
        early_stopping_rounds = 10
    )
    # Update best rmse

    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))




# Setting best parameters
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]




gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_rmse = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(subsample, colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (subsample,colsample)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))




# Setting best parameters
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]




min_rmse = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, 0.02, 0.01]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_rounds,
            seed=42,
            nfold=5,
            metrics=['rmse'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds\n".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = eta
print("Best params: {}, RMSE: {}".format(best_params, min_rmse))




params['eta'] = best_params




gridsearch_params = [
    (reg_lambda)
    for reg_lambda in np.arange(0, 10)
]




min_rmse = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for reg_lambda in gridsearch_params:
    print("CV with reg_lambda={}".format(reg_lambda))
    # We update our parameters
    params['reg_lambda'] = reg_lambda

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = reg_lambda
print("Best params: {}, RMSE: {}".format(best_params, min_rmse))




params['reg_lambda'] = best_params




best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_rounds,
    evals=[(dtest, "Test")],
    early_stopping_rounds = 10
)

# get_prediction_file(best_model, 'XGB_final_2.csv', 1)




test_df['O/P'] = -1

# Concatinating train_f and test_df to process them simultaneously
df = pd.concat((train_df, test_df), axis = 0)




df.describe()




dummy_columns = ['F3', 'F4', 'F5', 'F7', 'F8', 'F9', 'F11', 'F12']

# Dropping all Skewed columns
"""
for column in df.columns:
    if df[column].skew() > 0.5 and column != 'O/P':
        df.drop(labels=column, axis=1, inplace=True)
        print(column)
"""

# Mean Std Transformation
"""
for column in df.columns:
    if column not in dummy_columns and column != 'O/P':
        standard_deviation = df[column].std(axis = 0)
        mean = df[column].mean(axis = 0)
        df[column] = (df[column] - mean) / standard_deviation
    if abs(df[column].skew()) > 1 and column != 'O/P' and column not in dummy_columns:
        df[column] = np.log(df[column] + 1)
    if abs(df[column].skew() > 1) and column != 'O/P':
        print(df[column].skew())
        df.drop(column, inplace = True, axis = 1)
        removed_columns.append(column)
"""

# Min Max Transformation

for column in df.columns:
    if column not in dummy_columns and column != 'O/P':
        minimum = df[column].min(axis = 0)
        maximum = df[column].max(axis = 0)
        df[column] = (df[column] - minimum) / (maximum - minimum)
    if abs(df[column].skew()) > 1 and column != 'O/P' and column not in dummy_columns:
        df[column] = np.log(df[column] + 1)
    if abs(df[column].skew() > 1) and column != 'O/P':
        print(df[column].skew())
        df.drop(column, inplace = True, axis = 1)
        removed_columns.append(column)




print(removed_columns)
df.info()




df.describe()




df.head()




# Getting Dummy Variables
real_dummies = [column for column in dummy_columns if column not in removed_columns]
print(removed_columns)
print(real_dummies)
df = pd.get_dummies(df, columns = real_dummies)




train_df = df.loc[0:(14000-1), :]
test_df = df.loc[14000:, :]
test_df.drop('O/P', inplace = True, axis = 1)




# save_files('Train_data_norm_unskew.csv', 'Test_data_norm_unskew.csv')
X, y, X_train, y_train, X_test, y_test, X_predict, train_df, test_df = load_data_into_np(train_df, test_df)




regr_1 = MLPRegressor(random_state = 1, max_iter = 10000)

test_model(regr_1, 1, X, y)

# get_prediction_file(regr_1, 'nn_1.3.csv', 1)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

model = Sequential()
model.add(Flatten())
model.add(Dense(500, activation = tf.nn.relu))
model.add(Dense(200, activation = tf.nn.relu))
model.add(Dense(100, activation = tf.nn.relu))
model.add(Dropout(0.3))
model.add(Dense(1, activation = tf.nn.relu))

model.compile(optimizer = 'adam' , loss = 'mse', metrics = ['mse'])

history = model.fit(X_train, y_train, epochs = 50, batch_size = 10,  validation_data=(X_test, y_test))




plot_log(history, 5)




model = Sequential()
model.add(Flatten())
model.add(Dense(100, activation = tf.nn.relu))
model.add(Dense(100, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(1, activation = tf.nn.relu))

model.compile(optimizer = 'adam' , loss = 'mse', metrics = ['mse'])

history = model.fit(X_train, y_train, epochs = 20, batch_size = 10,  validation_data=(X_test, y_test))




plot_log(history, 1)




model = Sequential()
model.add(Flatten())
model.add(Dense(1000, activation = tf.nn.relu))
model.add(Dense(1000, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(1, activation = tf.nn.relu))

model.compile(optimizer = 'adam' , loss = 'mse', metrics = ['mse'])

history = model.fit(X_train, y_train, epochs = 20, batch_size = 10,  validation_data=(X_test, y_test))




plot_log(history, 0)




model = Sequential()
model.add(Flatten())
model.add(Dense(1000, activation = 'linear'))
model.add(Dense(1000, activation = tf.nn.relu))
model.add(Dropout(0.3))
model.add(Dense(1, activation = tf.nn.relu))

model.compile(optimizer = 'adam' , loss = 'mse', metrics = ['mse'])

history = model.fit(X_train, y_train, epochs = 50, batch_size = 10,  validation_data=(X_test, y_test))




plot_log(history, 3)

