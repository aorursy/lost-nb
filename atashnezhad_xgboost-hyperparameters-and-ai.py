#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score




def De_Algorithm(fobj, bounds, mut=0.8, crossp=0.7, popsize=100, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
                    
        #print("Iteration number= %s" % (i))
        #print("Best Fitness= %s" % (fitness[best_idx]))
        #print("Best values= %s" % (best))
        yield best, fitness[best_idx]




def xgb2(X_training, y_training, X_valid, y_valid, w):
    
    w[1] = round(w[1])
    w[2] = round(w[2])
    w[6] = round(w[6])
    w[7] = round(w[7])
    w[8] = round(w[8])
    w[9] = round(w[9])
    w[10] = round(w[10])
    
    params = {'eta': w[0], # 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': w[1], # 1400,  
          'max_depth': w[2], # 0, 
          'subsample': w[3], # 0.9, 
          'colsample_bytree': w[4], # 0.7, 
          'colsample_bylevel': w[5], # 0.7,
          'min_child_weight': w[6], # 0,
          'alpha': w[7], # 4,
          'objective': 'binary:logistic', 
          'scale_pos_weight': w[8], # 9,
          'eval_metric': 'auc', 
          'nthread': w[9], # 8,
          'random_state': w[10], # 99, 
          'silent': True}
    
    dtrain = xgb.DMatrix(X_training, y_training)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    model = xgb.train(params, dtrain, 100, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=0)
    
    # make predictions for test data
    X_valid = xgb.DMatrix(X_valid)
    y_pred = model.predict(X_valid, ntree_limit=model.best_ntree_limit)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_valid, predictions)
    auc = roc_auc_score(y_valid, predictions)
    #print("Accuracy: %.2f%%" % (accuracy * 100.0))

    return model, auc




#address_train = '../input/talkingdata-adtracking-fraud-detection/train_sample.csv'
address_train = '../input/talkingdata-adtracking-fraud-detection/train.csv'




def Drop_cols(df, x):
    df.drop(labels = x, axis = 1, inplace = True)
    return df

def Plot_Hist_column(df, x):
    pyplot.hist(df[x], log = True)
    pyplot.title(x)
    pyplot.show()
    
def Plot_Hist_columns(df, xlist):
    [Plot_Hist_column(df, x) for x in xlist]  
    pyplot.show()
    
def Make_X_Y(df):
    Y = pd.DataFrame()
    Y['is_attributed'] = df['is_attributed']
    X = df.copy()
    X.drop(labels = ["is_attributed"], axis = 1, inplace = True)
    return X, Y

def Train_Test_training_valid(X, Y, ratio):
    Num_of_line = 100
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio)
    X_training, X_valid, y_training, y_valid =     train_test_split(X_train, y_train, test_size=ratio, random_state=0)
    return X_training, y_training, X_valid, y_valid

def read_train_test_data_balanced(address_train):
    
    #Read Training data, all class 1 and add same amount 0
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'])
    df_train_1 = pd.concat([chunk[chunk['is_attributed'] > 0] for chunk in iter_csv])
    
    
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'], skiprows = range(1,120000000), nrows=2000000)
    df_train_0 = pd.concat([chunk[chunk['is_attributed'] == 0] for chunk in iter_csv])
    #seperate same number values as train data with class 1
    df_train_0 = df_train_0.head(len(df_train_1))
    #Merge 0 and 1 data
    df_train = Merge_data(df_train_1, df_train_0)
    return df_train

def Merge_data(df1, df2):
    frames = [df1, df2]
    df = pd.concat(frames)
    return df




df_train = read_train_test_data_balanced(address_train)
df_train.head(3)




# See the output paramters distribution 
xlist = ['is_attributed']
Plot_Hist_columns(df_train, xlist)




# Lets drop click_time and attributed_time for sack of simplicity 
df_train.drop(labels = ["click_time", "attributed_time"], axis = 1, inplace = True)




# Devide data set to input (X) and output (Y) paramters.
X, Y = Make_X_Y(df_train); 
X_training, y_training, X_valid, y_valid = Train_Test_training_valid(X, Y, 0.1)




# check out the ML algorithm and make sure it works.
# Here you can run the XGBoost algorithm on data with your favorite hyperparameters.

# w = [learning_rate, 
#      n_estimators, 
#      max_depth, 
#      min_child_weight,
#      gamma, 
#      subsample, 
#      colsample_bytree, 
#      nthread, 
#      scale_pos_weight]

'''w = [0.1, 3, 3, 1, 0, 0.8, 0.8, 4, 1]
Trained_XGBoost_Model, XGBoost_accuracy = Train_XGBoost(X_training, y_training, X_valid, y_valid, w)'''




# check out the ML algorithm and make sure it works.
# Here you can run the XGBoost algorithm on data with your favorite hyperparameters.

# w = [learning_rate, 
#      n_estimators, 
#      max_depth, 
#      min_child_weight,
#      gamma, 
#      subsample, 
#      colsample_bytree, 
#      nthread, 
#      scale_pos_weight]

w = [0.1, 1400, 0, 0.9, 0.7, 0.7, 0, 4, 9, 8, 99]
model2, accuracy = xgb2(X_training, y_training, X_valid, y_valid, w)




'''#Define an objective funtion.
def Objective_Function(w):
    w = w.tolist()
    Trained_XGBoost_Model, XGBoost_accuracy = Train_XGBoost(X_training, y_training, X_valid, y_valid, w)
    return (1-XGBoost_accuracy)'''
    
    

#Define an objective funtion.
def Objective_Function2(w):
    w = w.tolist()
    model2, accuracy = xgb2(X_training, y_training, X_valid, y_valid, w)
    return (1-accuracy)




'''#Run the DE algorithm on objective function in your favorite range of hyperparameters.
result = list(De_Algorithm(Objective_Function,
                 [(0.001, 1),   #  learning_rate
                  (3, 1000),   #  n_estimators
                  (2, 20),   #  max_depth
                  (1, 20),   #  min_child_weight
                  (0.001, 1),   #  gamma
                  (0.001, 1),   #  subsample
                  (0.001, 1),   #  colsample_bytree
                  (2, 8),   #  nthread
                  (1, 8)],   #  scale_pos_weight]
                  mut=0.4, crossp=0.8, popsize=10, its=30))'''

#Run the DE algorithm on objective function in your favorite range of hyperparameters.
result = list(De_Algorithm(Objective_Function2,
                 [(0.001,1),   #  eta
                  (3,1500),   #  max_leaves
                  (0,20),   #  max_depth
                  (0,1),   #  subsample
                  (0.001,1),   #  colsample_bytree
                  (0.001,1),   #  colsample_bylevel
                  (0.001,1),   #  min_child_weight
                  (2,8),   #  alpha
                  (1,10),   # scale_pos_weight
                  (1,10),     # nthread
                  (1,10)], #  random_state
                  mut=0.4, crossp=0.8, popsize=10, its=40))




df = pd.DataFrame(result)
# seperate the best of hyperparamters.
def Best_coffs(df):
    
    #df['w1'], df['w2'], df['w3'], df['w4'], df['w5'], df['w6'], df['w7'], df['w8'], df['w9']   = zip(*df[0]) # Unzip
    df['w1'], df['w2'], df['w3'], df['w4'], df['w5'], df['w6'], df['w7'], df['w8'], df['w9'], df['w10'], df['w11']   = zip(*df[0]) # Unzip
    cols = [0] # Drop the first column
    df.drop(df.columns[cols],axis = 1,inplace = True) # Drop the first column
    df.columns.values[0] = "Fitness" # name the first column as Fitness
    best_coff = df.iloc[len(df)-1,1:] # insert the best coefficients into the best_coff
    return best_coff
Best_coffs(df)




def Plot_DEA_Evolution(df):
    
    data_ncol=len(df.columns) # number of paramters 
    fig = plt.figure(figsize=(20,15)) # you may change these to change the distance between plots.

    for i in range(1,(data_ncol+1)):
        if i<(data_ncol):
            
            plt.subplot(3, 4, i)
            plt.plot(df['w{}'.format(i)],'bo', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('w{}'.format(i))
            plt.grid(True)
        else:       
            plt.subplot(3, 4, data_ncol)
            plt.plot(df['Fitness'],'red', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('Fitness')
            plt.grid(True)
    plt.show()




Plot_DEA_Evolution(df)




df = pd.DataFrame(result)
def Best_coffs(df):

    #df['learning_rate'],df['n_estimators'], df['max_depth'],df['min_child_weight'], df['gamma'],df['subsample'], df['colsample_bytree'],df['nthread'], df['scale_pos_weight'] = zip(*df[0]) # Unzip
    df['eta'], df['max_leaves'],df['max_depth'], df['subsample'],df['colsample_bytree'], df['colsample_bylevel'],df['min_child_weight'], df['alpha'],df['scale_pos_weight'], df['nthread'], df['random_state'] = zip(*df[0]) # Unzip
    cols = [0] # Drop the first column
    df.drop(df.columns[cols],axis = 1,inplace = True) # Drop the first column
    df.columns.values[0] = "Fitness" # name the first column as Fitness
    best_Parameters = df.iloc[len(df)-1,1:] # insert the best coefficients into the best_coff

    return best_Parameters


def print_hyper_parameters(df):
    
    '''best_Parameters = Best_coffs(df)
    best_Parameters[1] = round(best_Parameters[1])
    best_Parameters[2] = round(best_Parameters[2])
    best_Parameters[3] = round(best_Parameters[3])
    best_Parameters[7] = round(best_Parameters[7])
    best_Parameters[8] = round(best_Parameters[8])'''
    
    best_Parameters = Best_coffs(df)
    best_Parameters[1] = round(best_Parameters[1])
    best_Parameters[2] = round(best_Parameters[2])
    best_Parameters[6] = round(best_Parameters[6])
    best_Parameters[7] = round(best_Parameters[7])
    best_Parameters[8] = round(best_Parameters[8])
    best_Parameters[9] = round(best_Parameters[9])
    best_Parameters[10] = round(best_Parameters[10])
    

    
    print(best_Parameters)
    
print_hyper_parameters(df)   




def xgb2(X_training, y_training, X_valid, y_valid, w):
    
    w[1] = round(w[1])
    w[2] = round(w[2])
    w[6] = round(w[6])
    w[7] = round(w[7])
    w[8] = round(w[8])
    w[9] = round(w[9])
    w[10] = round(w[10])
    
    params = {'eta': w[0], # 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': w[1], # 1400,  
          'max_depth': w[2], # 0, 
          'subsample': w[3], # 0.9, 
          'colsample_bytree': w[4], # 0.7, 
          'colsample_bylevel': w[5], # 0.7,
          'min_child_weight': w[6], # 0,
          'alpha': w[7], # 4,
          'objective': 'binary:logistic', 
          'scale_pos_weight': w[8], # 9,
          'eval_metric': 'auc', 
          'nthread': w[9], # 8,
          'random_state': w[10], # 99, 
          'silent': True}
    
    dtrain = xgb.DMatrix(X_training, y_training)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    model = xgb.train(params, dtrain, 1000, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)
    
    # make predictions for test data
    X_valid = xgb.DMatrix(X_valid)
    y_pred = model.predict(X_valid, ntree_limit=model.best_ntree_limit)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_valid, predictions)
    #print("Accuracy: %.2f%%" % (accuracy * 100.0))

    return model, accuracy




df = pd.DataFrame(result)
w = list(Best_coffs(df))
Trained_Model, accuracy = xgb2(X_training, y_training, X_valid, y_valid, w)




address_test = '../input/talkingdata-adtracking-fraud-detection/test.csv'
df_test = pd.read_csv(address_test, parse_dates=['click_time'])
df_test.head()




# Lets drop click_time and attributed_time for sack of simplicity 
df_test.drop(labels = ["click_time", "click_id"], axis = 1, inplace = True)
df_test.head()




def predict_And_Submit_using_xgb(df, Trained_Model):
    
    Num_of_line = 100
    print(Num_of_line*'=')
    #sub = pd.DataFrame()
    #sub['click_id'] = df['click_id'].astype('int')
    #df['clicks_by_ip'] = df['clicks_by_ip'].astype('uint16')
    
    data_to_submit = pd.DataFrame()
    data_to_submit['click_id'] = range(0, len(df))
    dtest = xgb.DMatrix(df)
    del df
    predict = Trained_Model.predict(dtest, ntree_limit=Trained_Model.best_ntree_limit)
    data_to_submit['is_attributed'] = predict

    print(Num_of_line*'=')
    print('data_to_submit = \n', data_to_submit.head(5))
    pyplot.hist(data_to_submit['is_attributed'], log = True)
    #data_to_submit.to_csv('Amin_csv_to_submit.csv', index = False)
    return data_to_submit




data_to_submit = predict_And_Submit_using_xgb(df_test, Trained_Model)




data_to_submit.to_csv('Amin_csv_to_submit.csv', index = False)




data_to_submit2 = pd.DataFrame()
data_to_submit2['click_id'] = range(0, len(df_test))
data_to_submit2['is_attributed'] = [0 if i < 0.07 else 1 for i in data_to_submit['is_attributed']]
pyplot.hist(data_to_submit2['is_attributed'], log = True)


data_to_submit3 = pd.DataFrame()
data_to_submit3['click_id'] = range(0, len(df_test))
data_to_submit3['is_attributed'] = [0 if i < 0.065 else 1 for i in data_to_submit['is_attributed']]
pyplot.hist(data_to_submit2['is_attributed'], log = True)


data_to_submit4 = pd.DataFrame()
data_to_submit4['click_id'] = range(0, len(df_test))
data_to_submit4['is_attributed'] = [0 if i < 0.064 else 1 for i in data_to_submit['is_attributed']]
pyplot.hist(data_to_submit2['is_attributed'], log = True)


data_to_submit5 = pd.DataFrame()
data_to_submit5['click_id'] = range(0, len(df_test))
data_to_submit5['is_attributed'] = [0 if i < 0.062 else 1 for i in data_to_submit['is_attributed']]
pyplot.hist(data_to_submit2['is_attributed'], log = True)

'''data_to_submit.to_csv('Amin_csv_to_submit.csv', index = False)
data_to_submit2.to_csv('Amin_csv_to_submit2.csv', index = False)
data_to_submit3.to_csv('Amin_csv_to_submit3.csv', index = False)
data_to_submit4.to_csv('Amin_csv_to_submit4.csv', index = False)
data_to_submit5.to_csv('Amin_csv_to_submit5.csv', index = False)'''

