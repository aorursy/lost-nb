#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")




from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])




def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
#                 ('male_over_female', 'r4h3', 'r4m3'),
#                 ('man12plus_over_women12plus', 'r4h2', 'r4m2'),
#                 ('pesioner_over_working', 'hogar_mayor', 'hogar_adul'),
#                 ('children_over_working', 'hogar_nin', 'hogar_adul')
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean', 'count'],
                'escolari': ['min', 'max', 'mean', 'std']
               }
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df




def convert_OHE2LE(df):
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
        sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
        #deal with those OHE, where there is a sum over columns == 0
        if 0 in sum_ohe:
            print('The OHE in {} is incomplete. A new column will be added before label encoding'
                  .format(s_))
            # dummy colmn name to be added
            col_dummy = s_+'_dummy'
            # add the column to the dataframe
            tmp_df[col_dummy] = (tmp_df[cols_s_].sum(axis=1) == 0).astype(np.int8)
            # add the name to the list of columns to be label-encoded
            cols_s_.append(col_dummy)
            # proof-check, that now the category is complete
            sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                 print("The category completion did not work")
        tmp_cat = tmp_df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')




def process_df(df_):
    # fix categorical features
    encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(-1)
    # do feature engineering and drop useless columns
    return do_features(df_)

train = process_df(train)
test = process_df(test)




train.info()




def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_




train, test = train_test_apply_func(train, test, convert_OHE2LE)




train.info()




cols_2_ohe = ['eviv_LE', 'etecho_LE', 'epared_LE', 'elimbasu_LE', 
              'energcocinar_LE', 'sanitario_LE', 'manual_elec_LE',
              'pared_LE']
cols_nums = ['age', 'meaneduc', 'dependency', 
             'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',
             'bedrooms', 'overcrowding']

def convert_geo2aggs(df_):
    tmp_df = pd.concat([df_[(['lugar_LE', 'idhogar']+cols_nums)],
                        pd.get_dummies(df_[cols_2_ohe], 
                                       columns=cols_2_ohe)],axis=1)
    geo_agg = tmp_df.groupby(['lugar_LE','idhogar']).mean().groupby('lugar_LE').mean().astype(np.float32)
    geo_agg.columns = pd.Index(['geo_' + e + '_MEAN' for e in geo_agg.columns.tolist()])
    
    del tmp_df
    return df_.join(geo_agg, how='left', on='lugar_LE')

train, test = train_test_apply_func(train, test, convert_geo2aggs)




train.info()




X = train.query('parentesco1==1')
#X = train

# pull out the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)




cols_2_drop = ['abastagua_LE', 'agg18_estadocivil1_MEAN', 'agg18_instlevel6_MEAN', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_MEAN', 'agg18_parentesco12_MEAN', 'agg18_parentesco4_MEAN', 'agg18_parentesco5_MEAN', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_MEAN', 'agg18_parentesco8_MEAN', 'agg18_parentesco9_MEAN', 'fe_people_not_living', 'fe_people_weird_stat', 'geo_elimbasu_LE_3_MEAN', 'geo_elimbasu_LE_4_MEAN', 'geo_energcocinar_LE_0_MEAN', 'geo_energcocinar_LE_1_MEAN', 'geo_energcocinar_LE_2_MEAN', 'geo_epared_LE_0_MEAN', 'geo_epared_LE_2_MEAN', 'geo_etecho_LE_2_MEAN', 'geo_eviv_LE_0_MEAN', 'geo_hogar_mayor_MEAN', 'geo_hogar_nin_MEAN', 'geo_manual_elec_LE_1_MEAN', 'geo_manual_elec_LE_2_MEAN', 'geo_manual_elec_LE_3_MEAN', 'geo_pared_LE_0_MEAN', 'geo_pared_LE_1_MEAN', 'geo_pared_LE_3_MEAN', 'geo_pared_LE_4_MEAN', 'geo_pared_LE_5_MEAN', 'geo_pared_LE_6_MEAN', 'geo_pared_LE_7_MEAN', 'hacapo', 'hacdor', 'mobilephone', 'parentesco1', 'parentesco_LE', 'rez_esc', 'techo_LE', 'v14a', 'v18q']
#cols_2_drop = ['agg18_estadocivil1_MEAN', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_MEAN', 'agg18_parentesco12_MEAN', 'agg18_parentesco4_MEAN', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_MEAN', 'agg18_parentesco8_MEAN', 'fe_people_weird_stat', 'hacapo', 'hacdor', 'mobilephone', 'parentesco1', 'parentesco_LE', 'rez_esc', 'v14a']
#cols_2_drop=[]

X.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
test.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)




XY = pd.concat([X,y], axis=1)
max_corr = XY.corr()['Target'].loc[lambda x: abs(x)>0.2].index
#min_corr = XY.corr()['Target'].loc[lambda x: abs(x)<0.05].index




_ = plt.figure(figsize=(10,7))
_ = sns.heatmap(XY[max_corr].corr(), vmin=-0.5, vmax=0.5, cmap='coolwarm')




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)




X_test.info(max_cols=20)




from sklearn.metrics import f1_score
def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

import lightgbm as lgb
fit_params={"early_stopping_rounds":300, 
            "eval_metric" : evaluate_macroF1_lgb, 
            #"eval_set" : [(X_train,y_train), (X_test,y_test)],
            'eval_names': ['train', 'valid'],
            'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_power_0997)],
            'verbose': False,
            'categorical_feature': 'auto'}

fit_params['verbose'] = 200




# %%time
# from bayes_opt import BayesianOptimization
# import lightgbm as lgb


# def bayes_parameter_opt_lgb(X, y, init_round=15, opt_roun=25, n_folds=7, random_seed=42, n_estimators=10000, learning_rate=0.02, output_process=False,colsample_bytree=0.93,min_child_samples=56,subsample=0.84):
#     # prepare data
#     train_data = lgb.Dataset(data=X, label=y)
#     # parameters
#     def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, colsample_bytree,min_child_samples,subsample):
#         params = {'application':'multiclass','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':300, 'metric':'macroF1'}
#         params["num_leaves"] = int(round(num_leaves))
#         params["num_class"] = 4
#         params['feature_fraction'] = max(min(feature_fraction, 1), 0)
#         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
#         params['max_depth'] = int(round(max_depth))
#         params['lambda_l1'] = max(lambda_l1, 0)
#         params['lambda_l2'] = max(lambda_l2, 0)
#         params['min_split_gain'] = min_split_gain
#         params['min_child_weight'] = min_child_weight
#         params['colsample_bytree'] = 1
#         params['min_child_samples'] = 90,
#         params['subsample'] = 0.96
#         cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
#         return max(cv_result['auc-mean'])
#     # range 
#     lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (19, 45),
#                                             'feature_fraction': (0.1, 0.9),
#                                             'bagging_fraction': (0.8, 1),
#                                             'max_depth': (5, 8.99),
#                                             'lambda_l1': (0, 5),
#                                             'lambda_l2': (0, 3),
#                                             'min_split_gain': (0.001, 0.1),
#                                             'min_child_weight': (5, 50),
#                                             'colsample_bytree' : (0.7,1.0),
#                                             'min_child_samples' : (40,65),
#                                             'subsample' : (0.7,1.0)
#                                            }, random_state=0)
#     # optimize
#     lgbBO.maximize(init_points=init_round, n_iter=opt_roun)
    
#     # output optimization process
#     if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
#     # return best parameters
#     return lgbBO.res['max']['max_params']

# opt_params = bayes_parameter_opt_lgb(X_train, y_train, init_round=10, opt_roun=10, n_folds=6, random_seed=42, n_estimators=500, learning_rate=0.02,colsample_bytree=0.93)




opt_params




#v8
#opt_parameters = {'colsample_bytree': 0.93, 'min_child_samples': 56, 'num_leaves': 19, 'subsample': 0.84}
#v9
#opt_parameters = {'colsample_bytree': 0.89, 'min_child_samples': 70, 'num_leaves': 17, 'subsample': 0.96}
#v14
#opt_parameters = {'colsample_bytree': 0.88, 'min_child_samples': 90, 'num_leaves': 16, 'subsample': 0.94}
#v17
# opt_parameters = {'colsample_bytree': 0.89, 'min_child_samples': 90, 'num_leaves': 14, 'subsample': 0.96}

opt_parameters = {
                'bagging_fraction': 1.0,
                 'colsample_bytree': 0.75,
                 'feature_fraction': 0.1,
                 'lambda_l1': 5.0,
                 'lambda_l2': 3.0,
                 'max_depth': 5,
                 'min_child_samples': 90,
                 'min_child_weight': 5.0,
                 'min_split_gain': 0.001,
                 'num_leaves': 19,
                 'subsample': 0.7,
                'min_sum_hessian_in_leaf': 1,
                'importance_type': 'gain'
                }




from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble.voting_classifier import _parallel_fit_estimator

def _parallel_fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator

class VotingClassifierLGBM(VotingClassifier):
    '''
    This implements the fit method of the VotingClassifier propagating fit_params
    '''
    def fit(self, X, y, sample_weight=None, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        if sample_weight is not None:
            for name, step in self.estimators:
                if not has_fit_parameter(step, 'sample_weight'):
                    raise ValueError('Underlying estimator \'%s\' does not'
                                     ' support sample weights.' % name)
        names, clfs = zip(*self.estimators)
        self._validate_names(names)

        n_isnone = np.sum([clf is None for _, clf in self.estimators])
        if n_isnone == len(self.estimators):
            raise ValueError('All estimators are None. At least one is '
                             'required to be a classifier!')

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimators_ = []

        transformed_y = self.le_.transform(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y,
                                                 sample_weight=sample_weight, **fit_params)
                for clf in clfs if clf is not None)

        return self




from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

class VotingPrefitClassifier(VotingClassifier):
    '''
    This implements the VotingClassifier with prefitted classifiers
    '''
    def fit(self, X, y, sample_weight=None, **fit_params):
        self.estimators_ = [x[1] for x in self.estimators]
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        return self
    




#clfs = []
#for i in range(3):
#    clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
#                             random_state=314+i, silent=True, metric='None', 
#                             n_jobs=4, n_estimators=5000, class_weight='balanced')
#    clf.set_params(**opt_parameters)
#    clfs.append(('lgbm{}'.format(i), clf))
    
#vc = VotingClassifierLGBM(clfs, voting='soft')
#del clfs
##Train the final model with learning rate decay
#_ = vc.fit(X_train, y_train, **fit_params)
#
#clf_final = vc.estimators_[0]




from sklearn.model_selection import StratifiedKFold

def train_lgbm_model(X_, y_, random_state_=None, opt_parameters_={}, fit_params_={}):
    clf  = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=random_state_, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced')
    clf.set_params(**opt_parameters_)
    return clf.fit(X_, y_, **fit_params_)

# the list of classifiers for voting ensable
clfs = []

# nested CV parameters
inner_seed = 31416
inner_n = 10
outer_seed = 314
outer_n = 10

# performance 
perf_eval = {'f1_oof': [],
             'f1_ave': [],
             'f1_std': []}

outer_cv = StratifiedKFold(outer_n, shuffle=True, random_state=outer_seed)
for n_outer_fold, (outer_trn_idx, outer_val_idx) in enumerate(outer_cv.split(X,y)):
    print('--- Outer loop iteration: {} ---'.format(n_outer_fold))
    X_out, y_out = X.iloc[outer_trn_idx], y.iloc[outer_trn_idx]
    X_stp, y_stp = X.iloc[outer_val_idx], y.iloc[outer_val_idx]
    
    inner_cv = StratifiedKFold(inner_n, shuffle=True, random_state=inner_seed+n_outer_fold)
    
    y_oof = pd.Series(np.zeros(shape=(X_out.shape[0],)), 
                      index=X_out.index)
    f1_scores_inner = []
    
    for n_inner_fold, (inner_trn_idx, inner_val_idx) in enumerate(inner_cv.split(X_out,y_out)):
        X_trn, y_trn = X_out.iloc[inner_trn_idx], y_out.iloc[inner_trn_idx]
        X_val, y_val = X_out.iloc[inner_val_idx], y_out.iloc[inner_val_idx]
        
        # use _stp data for early stopping
        fit_params["eval_set"] = [(X_trn,y_trn), (X_stp,y_stp)]
        
        clf = train_lgbm_model(X_trn, y_trn, 314+n_inner_fold, opt_parameters, fit_params)
        
        # evaluate performance
        y_oof.iloc[inner_val_idx] = clf.predict(X_val)        
        f1_scores_inner.append(f1_score(y_val, y_oof.iloc[inner_val_idx], average='macro'))
        #cleanup
        del clf, X_trn, y_trn, X_val, y_val
    # Store performance info for theis outer fold
    perf_eval['f1_oof'].append(f1_score(y_out, y_oof, average='macro'))
    perf_eval['f1_ave'].append(np.array(f1_scores_inner).mean())
    perf_eval['f1_std'].append(np.array(f1_scores_inner).std())
    # Train main model for the voting average
    fit_params["eval_set"] = [(X_out,y_out), (X_stp,y_stp)]
    print('Fit the final model on the outer loop iteration: ')
    clf = train_lgbm_model(X_out, y_out, 314+n_outer_fold, opt_parameters, fit_params)
    clfs.append(('lgbm{}'.format(n_outer_fold), clf))
    # cleanup
    del inner_cv, X_out, y_out, X_stp, y_stp




vc = VotingPrefitClassifier(clfs)
vc = vc.fit(X,y)
clf_final = vc.estimators_[0]




global_score = np.mean(perf_eval['f1_oof'])
global_score_std = np.std(perf_eval['f1_oof'])
#vc.voting = 'soft'
#global_score_soft = f1_score(y, vc.predict(X), average='macro')
#vc.voting = 'hard'
#global_score_hard = f1_score(y, vc.predict(X), average='macro')

print('Mean validation score LGBM Classifier: {:.4f}'.format(global_score))
print('Std  validation score LGBM Classifier: {:.4f}'.format(global_score_std))
#print('Validation score of a VotingClassifier on 3 LGBMs with soft voting strategy: {:.4f}'.format(global_score_soft))
#print('Validation score of a VotingClassifier on 3 LGBMs with hard voting strategy: {:.4f}'.format(global_score_hard))




from sklearn.metrics import precision_score, recall_score, classification_report




# print(classification_report(y_test, clf_final.predict(X_test)))




#vc.voting = 'hard'
#print(classification_report(y_test, vc.predict(X_test)))




#vc.voting = 'soft'
#print(classification_report(y_test, vc.predict(X_test)))




def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    # Plot feature importances
    if not doWorst:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_feat].index        
    else:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[-n_feat:].index
    
    mean_imp = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    df_2_neglect = mean_imp[mean_imp['importance'] < 1e-3]
    print('The list of features with 0 importance: ')
    print(df_2_neglect.index.values.tolist())
    del mean_imp, df_2_neglect
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    #plt.savefig('lgbm_importances.png')
    
importance_df = pd.DataFrame()
importance_df["feature"] = X.columns.tolist()      
importance_df["importance"] = clf_final.booster_.feature_importance('gain')
display_importances(feature_importance_df_=importance_df, n_feat=20)




#display_importances(feature_importance_df_=importance_df, doWorst=True, n_feat=20)




# import shap
# shap_values = shap.TreeExplainer(clf_final.booster_).shap_values(X)

# #shap_df = pd.DataFrame()
# #shap_df["feature"] = X_train.columns.tolist()    
# #shap_df["importance"] = np.sum(np.abs(shap_values), 0)[:-1]




#display_importances(feature_importance_df_=shap_df, n_feat=20)




# shap.summary_plot(shap_values, X, plot_type='bar')




y_subm = pd.read_csv('../input/sample_submission.csv')




y_subm['Target'] = clf_final.predict(test) + 1

vc.voting = 'soft'
y_subm_soft = y_subm.copy(deep=True)
y_subm_soft['Target'] = vc.predict(test) + 1

vc.voting = 'hard'
y_subm_hard = y_subm.copy(deep=True)
y_subm_hard['Target'] = vc.predict(test) + 1




# nor needed anymore
#y_subm_0forNonHeads = y_subm.copy(deep=True)
#y_subm_0forNonHeads.loc[y_subm_0forNonHeads[test['parentesco1'] == 0].index,'Target'] = 1




from datetime import datetime
now = datetime.now()

sub_file = 'submission_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_soft = 'submission_soft_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_hard = 'submission_hard_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_0forNonHeads = 'submission_0forNonHead_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))

y_subm.to_csv(sub_file, index=False)
y_subm_soft.to_csv(sub_file_soft, index=False)
y_subm_hard.to_csv(sub_file_hard, index=False)
# not needed anymore
#y_subm_0forNonHeads.to_csv(sub_file_0forNonHeads, index=False)






