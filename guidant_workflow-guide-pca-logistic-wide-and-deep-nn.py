#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import seaborn as sns
import category_encoders as ce
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from keras.layers import Dense, Dropout, Concatenate, Input, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model

from category_encoders import TargetEncoder
from plotnine import *

import os
for dirname, _, filenames in os.walk('..'):
    for filename in filenames:
       print(os.path.join(dirname, filename))




import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 6]
plt.rc('xtick', labelsize=13) 
plt.rc('ytick', labelsize=13) 

input_dir = '/kaggle/input/cat-in-the-dat-ii/'




df_train_raw = pd.read_csv(input_dir + 'train.csv')
df_test_raw = pd.read_csv(input_dir + 'test.csv')

df_train_raw.head()




df_train_raw_transformed = df_train_raw.    assign(bin_3 = df_train_raw['bin_3'].map(lambda X: 0 if (X == 'F') else                                             (1 if (X == 'T') else np.nan))).    assign(bin_4 = df_train_raw['bin_4'].map(lambda X: 0 if (X == 'N') else                                             (1 if (X == 'Y') else np.nan)))

df_train_raw_transformed.head()




df_test_raw_transformed = df_test_raw.    assign(bin_3 = df_test_raw['bin_3'].map(lambda X: 0 if (X == 'F') else                                             (1 if (X == 'T') else np.nan))).    assign(bin_4 = df_test_raw['bin_4'].map(lambda X: 0 if (X == 'N') else                                             (1 if (X == 'Y') else np.nan)))




print("ORD_0:")
print(df_train_raw_transformed['ord_0'].unique())
print("---\nORD1:")
print(df_train_raw_transformed['ord_1'].unique())
print("---\nORD2:")
print(df_train_raw_transformed['ord_2'].unique())
print("---\nORD3:")
print(df_train_raw_transformed['ord_3'].unique())
print("---\nORD4:")
print(df_train_raw_transformed['ord_4'].unique())
print("---\nORD5:")
print(df_train_raw_transformed['ord_5'].unique())




def apply_dict(X, dict_in):
    try:
        out = dict_in[X]
    except:
        out = X
    return out

dict_ord_1 = dict(Novice=1, Contributor=2, Expert=3, 
                  Master=4, Grandmaster=5)

dict_ord_2 = {'Freezing': 1, 'Cold': 2, 'Warm': 3, 
              'Hot': 4, 'Boiling Hot': 5, 'Lava Hot': 6}

dict_ord_3 = dict_ord_4 = dict()
for i in range(ord('a'), (ord('z') + 1)):
    dict_ord_3[chr(i)] = dict_ord_4[chr(i + ord('A') - ord('a'))] = i - ord('a') + 1
    
df_train = df_train_raw_transformed.    assign(ord_1 = df_train_raw_transformed['ord_1'].map(lambda X: apply_dict(X, dict_ord_1))).    assign(ord_2 = df_train_raw_transformed['ord_2'].map(lambda X: apply_dict(X, dict_ord_2))).    assign(ord_3 = df_train_raw_transformed['ord_3'].map(lambda X: apply_dict(X, dict_ord_3))).    assign(ord_4 = df_train_raw_transformed['ord_4'].map(lambda X: apply_dict(X, dict_ord_4)))




df_test = df_test_raw_transformed.    assign(ord_1 = df_test_raw_transformed['ord_1'].map(lambda X: apply_dict(X, dict_ord_1))).    assign(ord_2 = df_test_raw_transformed['ord_2'].map(lambda X: apply_dict(X, dict_ord_2))).    assign(ord_3 = df_test_raw_transformed['ord_3'].map(lambda X: apply_dict(X, dict_ord_3))).    assign(ord_4 = df_test_raw_transformed['ord_4'].map(lambda X: apply_dict(X, dict_ord_4)))




df_train.head()




len(list(df_train_raw_transformed.index))




df_count_per_col = pd.DataFrame(df_train.nunique())
df_count_per_col.columns = ['Values']
df_count_per_col.index.name = 'Feature'
df_count_per_col.reset_index(inplace=True)

ggplot(df_count_per_col[df_count_per_col['Feature'] != 'id'], 
       aes(x = 'Feature', y = 'Values', fill = 'Feature')) + geom_bar(stat = 'identity', color = 'black') +\
    theme(axis_text_x = element_text(angle = 90, hjust = 1), legend_position = 'none') +\
    ggtitle('Different Features per Column') + ylab('Count')




df_missing_col = pd.DataFrame(dict(PercMissing = df_train.isnull().sum() / 
                                   len(df_train.index))).reset_index()
df_missing_col.loc[df_missing_col['PercMissing'] > 0, :]

ggplot(df_missing_col[df_missing_col['Feature'] != 'id'], 
       aes(x = 'Feature', y = 'PercMissing', fill = 'Feature')) + geom_bar(stat = 'identity', color = 'black') +\
    theme(axis_text_x = element_text(angle = 90, hjust = 1), legend_position = 'none') +\
    ggtitle('Different Features per Column') + ylab('Count')




df_missing_row = pd.DataFrame(dict(PercMissing = df_train.isnull().sum(axis=1) / 
                                   len(df_train.index))).reset_index()

df_missing_row.columns = ['Index', 'PercMissing']
print(str(100 * max(df_missing_row['PercMissing'])) + ' %')




binary_features = ['bin_' + str(i) for i in range(0, 5)]
nominal_features_low_count = ['nom_' + str(i) for i in range(0, 5)]
nominal_features_high_count = ['nom_' + str(i) for i in range(5, 10)]
ordinal_features_low_count = ['ord_' + str(i) for i in range(0, 5)]
ordinal_features_high_count = ['ord_5']
date_features = ['day', 'month']




class SimpleImputerCorrected(BaseEstimator, TransformerMixin):
    
    def __init__(self, strategy='most_frequent', verbose=False):
        
        self.strategy = strategy
        self.preprocessor = None
        self.verbose = verbose

    def fit(self, X, y=None):
        
        if self.strategy == 'most_frequent':
            
            col_list = list(X.columns)
            col_transformers = list()
            col_mode = X.mode(axis=0).iloc[0]
        
            for curr_col in col_list:
                curr_transformer_name = 'T_' + curr_col
                curr_imputer = SimpleImputer(strategy='constant', 
                                             fill_value=col_mode[curr_col])
                col_transformers.append((curr_transformer_name, curr_imputer, 
                                         [curr_col]))
            
            self.preprocessor = ColumnTransformer(transformers=col_transformers, verbose=self.verbose)
            
        else:
            self.preprocessor = SimpleImputer(strategy=self.strategy)
            
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)




pass_features = binary_features + ordinal_features_low_count
one_hot_features = nominal_features_low_count
avg_features = nominal_features_high_count + ordinal_features_high_count

pass_pipeline = Pipeline(steps = [
    ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
    ('scalling', StandardScaler())
], verbose = True)

one_hot_pipeline  = Pipeline(steps = [
    ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
    ('encoding', OneHotEncoder(sparse = False))
], verbose = True)

avg_pipeline = Pipeline(steps = [
    ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
    ('encoding', TargetEncoder()),
    ('scalling', StandardScaler())
], verbose = True)

encoder = ColumnTransformer(
    transformers=[
        ('pass_pipeline', pass_pipeline, pass_features),
        ('one_hot_pipeline', one_hot_pipeline, one_hot_features),
        ('avg_pipeline', avg_pipeline, avg_features)
], verbose = True, sparse_threshold=0)

print(one_hot_features)




df_train.iloc[1:5, 1:-2]




array_train = encoder.fit_transform(df_train.iloc[:, 1:-2], df_train['target'])
print(array_train)




data_pca = PCA().fit_transform(array_train) 




data_pca




data_pca.shape




var_pca = data_pca.var(axis = 0)
imp_pca = var_pca / sum(var_pca)
cum_imp_pca = np.cumsum(imp_pca)

list_vars = var_pca.tolist() + imp_pca.tolist() + cum_imp_pca.tolist()
named_vars = ['Variance'] * len(var_pca) +             ['Importance'] * len(imp_pca) +             ['Cumulative_Importance'] * len(cum_imp_pca)
        
df_pca = pd.DataFrame(dict(
    IndexStr=[str(i) for i in 3 * list(range(0, len(named_vars) // 3))],
    Index=3 * list(range(0, len(named_vars) // 3)),
    Variable=named_vars,
    Value=list_vars
))

df_pca.head()




ggplot(aes(x = 'Index', y = 'Value')) +    geom_bar(aes(fill = 'IndexStr'), color = 'black', stat = 'identity', data = df_pca[df_pca['Variable'] == 'Importance']) +    theme(legend_position = 'none') + ggtitle('Cumulative Variables Importance') +    geom_line(data = df_pca[df_pca['Variable'] == 'Cumulative_Importance'], color = 'blue') +    geom_point(data = df_pca[df_pca['Variable'] == 'Cumulative_Importance'], color = 'blue') +    geom_point(data = df_pca[(df_pca['Variable'] == 'Cumulative_Importance') &                             (df_pca['Value'] > 0.9999999999)], color = 'red', shape = 'x', size = 5)




imp_pca[35:-1]




def get_preprocessor(pass_features=binary_features + ordinal_features_low_count, 
                     one_hot_features=nominal_features_low_count, 
                     avg_features=nominal_features_high_count + ordinal_features_high_count, 
                     te_smoothing=1,
                     pca_threshold=0.9999):

    pass_pipeline = Pipeline(steps = [
        ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
        ('scalling', StandardScaler())])

    one_hot_pipeline  = Pipeline(steps = [
        ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
        ('encoding', OneHotEncoder(sparse = False))])

    avg_pipeline = Pipeline(steps = [
        ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
        ('encoding', TargetEncoder(smoothing = te_smoothing)),
        ('scalling', StandardScaler())])

    encoder = ColumnTransformer(
    transformers=[
        ('pass_pipeline', pass_pipeline, pass_features),
        ('one_hot_pipeline', one_hot_pipeline, one_hot_features),
        ('avg_pipeline', avg_pipeline, avg_features)
    ], sparse_threshold=0)
    
    if (pca_threshold > 0):
        preprocessor = Pipeline(steps = [('encoder', encoder), ('pca', PCA(n_components=pca_threshold))])
    else:
        preprocessor = Pipeline(steps = [('encoder', encoder)])
    return preprocessor




pass_list = [pass_features, pass_features, pass_features]

one_hot_list = [nominal_features_low_count + ordinal_features_low_count,
                nominal_features_low_count, []]

avg_list = [avg_features,
            avg_features + ordinal_features_low_count,
            avg_features + ordinal_features_low_count + nominal_features_low_count]

transformers_name = ['minimum_te', 'medium_te', 'maximum_te']
transformers_list = zip(transformers_name, pass_list, one_hot_list, avg_list)

hyper_pipe_dict = {
    'lor_model__solver': ['saga'],
    'lor_model__penalty': ['elasticnet'],
    'lor_model__C': [0.1, 1, 10],
    'lor_model__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'preprocessor__transformers': transformers_list,
    'preprocessor__target_smoothing': [0.1, 1, 10],
    'preprocessor__pca_threshold': [0.9999]
}




y_train = df_train['target']
k_fold_n_splits, k_fold_use = 20, 3

iter_list = []
C_list = []
l1_list = []
t_list = []
smoothing_list = []
solver_list = []
penalty_list = []
train_score_list = []
pca_list = []
test_score_list = []

for t_name, t_pass, t_one_hot, t_avg in hyper_pipe_dict['preprocessor__transformers']:
    for C in hyper_pipe_dict['lor_model__C']:
        for l1_ratio in hyper_pipe_dict['lor_model__l1_ratio']:
            for target_smoothing in hyper_pipe_dict['preprocessor__target_smoothing']:
                for solver in hyper_pipe_dict['lor_model__solver']:
                    for penalty in hyper_pipe_dict['lor_model__penalty']:
                        for pca_threshold in hyper_pipe_dict['preprocessor__pca_threshold']:
                            iter_list.append([C, l1_ratio, t_name, t_pass, t_one_hot, t_avg, 
                                              target_smoothing, solver, penalty, pca_threshold])




cross_validate = False




if cross_validate:
    
    for i in range(0, len(iter_list)):
        
        print('Progress:')
        print('{0:.2%}'.format(i / len(iter_list)))
        print('-----')
        
        C, l1_ratio, t_name, t_pass, t_one_hot, t_avg, target_smoothing, solver, penalty, pca_threshold = iter_list[i]
        preprocessing_pipeline = get_preprocessor(t_pass, t_one_hot, t_avg, target_smoothing, pca_threshold=pca_threshold)
        x_train = preprocessing_pipeline.fit_transform(df_train.iloc[:, 1:-2], y_train)
                        
        k_fold_obj = StratifiedKFold(n_splits=k_fold_n_splits, shuffle=True)
        k_fold_count = 0
        for train_index, test_index in k_fold_obj.split(x_train, y_train):
            if k_fold_count >= k_fold_use:
                break
            else:
                k_fold_count += 1
                            
                x_cv_train, y_cv_train = x_train[train_index], y_train[train_index]
                x_cv_test, y_cv_test = x_train[test_index], y_train[test_index]
                            
                curr_lor = LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, penalty=penalty, n_jobs=-1)
                curr_lor.fit(x_cv_train, y_cv_train)
                            
                y_cv_train_pred = curr_lor.predict_proba(x_cv_train)[:, 1]
                y_cv_test_pred = curr_lor.predict_proba(x_cv_test)[:, 1]
                            
                roc_auc_train = roc_auc_score(y_cv_train, y_cv_train_pred)
                roc_auc_test = roc_auc_score(y_cv_test, y_cv_test_pred)
                            
                C_list.append(C)
                l1_list.append(l1_ratio)
                t_list.append(t_name)
                smoothing_list.append(target_smoothing)
                solver_list.append(solver)
                penalty_list.append(penalty)
                pca_list.append(pca_threshold)
                train_score_list.append(roc_auc_train)
                test_score_list.append(roc_auc_test)
            
    print('Progress:')
    print('100%')
    print('-----')
    
    df_cv = pd.DataFrame(
        dict(
            C=C_list,
            L1=l1_list,
            T=t_list,
            Smooth=smoothing_list,
            Solver=solver_list,
            Penalty=penalty_list,
            PCA_Threshold=pca_list,
            Train_AUC=train_score_list,
            Test_AUC=test_score_list
        )
    )
    df_cv.to_csv('/kaggle/working/df_cv.csv', index=False)
            
else:
    df_cv = pd.read_csv('../input/cv-results-cat-challenge/df_cv.csv')




df_cv.sort_values(by='Test_AUC', ascending=False).iloc[0:10, :]




df_cv_grouped = df_cv.groupby(['C', 'L1', 'T', 'Smooth', 'Solver', 'Penalty']).mean().sort_values(by='Test_AUC', ascending=False)
df_cv_grouped.iloc[0:10, :]




ggplot(df_cv, aes(x='Test_AUC')) + geom_histogram(bins = 50, fill = 'lightblue') +    ggtitle('CV Test AUC Distribution (Before Grouping)') + xlab('ROC-AUC') + ylab('Count - 50 Bins')




ggplot(df_cv_grouped, aes(x='Test_AUC')) + geom_histogram(bins = 50, fill = 'lightgreen') +    ggtitle('CV Test AUC Distribution (After Grouping)') + xlab('ROC-AUC') + ylab('Count - 50 Bins')




# C 	L1 	    T 	        Smooth 	Solver 	Penalty
# 0.1 	0.25 	maximum_te 	0.1 	saga 	elasticnet
# 1.0 	0.50 	minimum_te 	10.0 	saga 	elasticnet

opt_pipe_A = get_preprocessor(pass_features, [], avg_features + ordinal_features_low_count + nominal_features_low_count, 0.1)
opt_pipe_B = get_preprocessor(pass_features, ordinal_features_low_count + nominal_features_low_count, avg_features, 10.)

opt_pipe_A.fit(df_train.iloc[:, 1:-2], y_train)
opt_pipe_B.fit(df_train.iloc[:, 1:-2], y_train)

opt_model_A = LogisticRegression(C=0.1, l1_ratio=0.25, solver='saga', penalty='elasticnet')
opt_model_B = LogisticRegression(C=1.0, l1_ratio=0.50, solver='saga', penalty='elasticnet')

x_pipe_A = opt_pipe_A.transform(df_train.iloc[:, 1:-2])
x_pipe_B = opt_pipe_B.transform(df_train.iloc[:, 1:-2])

opt_model_A.fit(x_pipe_A, y_train)
opt_model_B.fit(x_pipe_B, y_train)




target_A = opt_model_A.predict_proba(opt_pipe_A.transform(df_test.iloc[:, 1:-1]))[:, 1]
target_B = opt_model_B.predict_proba(opt_pipe_B.transform(df_test.iloc[:,1:-1]))[:, 1]




pd.DataFrame({
    'id': df_test['id'],
    'target': target_A
}).to_csv('/kaggle/working/df_out_logistic_model_A.csv', index=False)

pd.DataFrame({
    'id': df_test['id'],
    'target': target_B
}).to_csv('/kaggle/working/df_out_logistic_model_B.csv', index=False)




default_pipeline = get_preprocessor(pass_features=binary_features, 
                                    one_hot_features=nominal_features_low_count + ordinal_features_low_count, 
                                    avg_features=nominal_features_high_count + ordinal_features_high_count)

x_train = default_pipeline.fit_transform(df_train.iloc[:, 1:-2], y_train)
x_test = default_pipeline.transform(df_test.iloc[:, 1:-1])
x_train




x_train.shape




def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

def get_wide_and_deep(use_montecarlo=False):

    inputs = Input(shape=(81,))

    deep = Dense(81, activation='elu')(inputs)
    deep = Dropout(0.5)(deep) if not use_montecarlo else Dropout(0.5)(deep, training=True)
    deep = Dense(40, activation='elu')(deep)
    deep = Dropout(0.5)(deep) if not use_montecarlo else Dropout(0.5)(deep, training=True)
    deep = Dense(20, activation='elu')(deep)
    deep = Dropout(0.5)(deep) if not use_montecarlo else Dropout(0.5)(deep, training=True)
    deep = Dense(10, activation='elu')(deep)

    deep_and_wide = Concatenate()([deep, inputs])
    deep_and_wide = BatchNormalization()(deep_and_wide)
    deep_and_wide = Dense(1, activation='sigmoid')(deep_and_wide)

    model_nn = Model(inputs=inputs, outputs=deep_and_wide)
    model_nn.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy', metrics=['accuracy', auc])
    
    return model_nn




model_nn = get_wide_and_deep()
model_nn.summary()




plot_model(model_nn, to_file='model.png', show_shapes=True, show_layer_names=True)




df_train['target'].mean()




train_idx.shape




es = EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,verbose=1, mode='max', baseline=None, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_auc', factor=0.5,patience=3, min_lr=1e-6, mode='max', verbose=1)

n_folds = 10
sfk = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
pred_list, hist_list = [], []

fold_index = 1
for train_idx, val_idx in sfk.split(x_train, y_train):
    
    print('\n---\nFold Index\n---\n: ' + str(fold_index))
    fold_index += 1
    
    model_nn = get_wide_and_deep()
    history_nn = model_nn.fit(x_train[train_idx], y_train[train_idx],
                              validation_data = (x_train[val_idx], y_train[val_idx]),
                              callbacks=[es, rlr], 
                              epochs=100, batch_size=1024, 
                              class_weight={0: 0.2, 1: 0.8},
                              verbose=0)
    
    pred_list.append(model_nn.predict(x_test))
    hist_list.append(history_nn)




pred_list = [list(X) for X in pred_list]
pred_list_formatted = [np.array([Y[0] for Y in X]) for X in pred_list]
target_nn = list(np.mean(pred_list_formatted, axis=0))




out_nn = model_nn.predict(x_test)




history_nn.history.keys()




history_nn.history.keys()




list_val = history_nn.history['val_loss'] +           history_nn.history['val_accuracy'] +           history_nn.history['val_auc'] +           history_nn.history['loss'] +           history_nn.history['accuracy'] +           history_nn.history['auc']

n_epochs = len(history_nn.history['val_loss'])

list_steps = 6 * (list(range(1, n_epochs + 1)))

list_metrics = (n_epochs * ['Loss']) + (n_epochs * ['Accuracy']) +               (n_epochs * ['AUC']) + (n_epochs * ['Loss']) +               (n_epochs * ['Accuracy']) + (n_epochs * ['AUC'])

list_kind = (n_epochs * ['Validation']) + (n_epochs * ['Validation']) +            (n_epochs * ['Validation']) + (n_epochs * ['Training']) +            (n_epochs * ['Training']) + (n_epochs * ['Training'])

df_nn_history = pd.DataFrame(dict(Step=list_steps, Value=list_val, Metric=list_metrics, Kind=list_kind))
df_nn_history.head()




ggplot(df_nn_history[df_nn_history['Step'] > 5], aes(x='Step', y='Value', colour='Kind')) +    geom_line(aes(group='Kind')) +    facet_grid('Metric ~ .', scales='free') +    ggtitle('Training / Validation Metrics') + geom_point(aes(shape = 'Kind'))




target_nn = [X[0] for X in out_nn.tolist()]
target_nn[1:5]




pd.DataFrame({
    'id': df_test['id'],
    'target': target_nn
}).to_csv('/kaggle/working/df_out_wide_and_deep.csv', index=False)




pd.DataFrame({
    'id': df_test['id'],
    'target': (np.array(target_nn) + np.array(target_B)) / 2
}).to_csv('/kaggle/working/df_out_models_avg.csv', index=False)




model_nn_mc = get_wide_and_deep(use_montecarlo=True)
model_nn_mc.set_weights(model_nn.get_weights())




predictions_list = []
n_montecarlo_sims = 100
for i in range(n_montecarlo_sims):
    if (i % 10) == 0:
        print('Monte Carlo Simulation - Iteration: ' + str(i + 1) + '/' + str(n_montecarlo_sims))
    predictions_list.append([X[0] for X in model_nn_mc.predict(x_test)])




predictions_list = [np.array(X) for X in predictions_list]
target_nn_mc = sum(predictions_list) / n_montecarlo_sims




pd.DataFrame({
    'id': df_test['id'],
    'target': target_nn_mc
}).to_csv('/kaggle/working/df_out_models_mc.csv', index=False)




pd.DataFrame({
    'id': df_test['id'],
    'target': (target_nn_mc + np.array(target_B)) / 2
}).to_csv('/kaggle/working/df_out_models_avg_mc.csv', index=False)

