#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt # for plotting
import matplotlib.gridspec as gridspec # to do the grid of plots
# jupyter cell magic for inline visualization
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns # for plotting
sns.set(style='whitegrid') # for plotting style

from IPython.display import display

from sklearn.model_selection import train_test_split, cross_val_score
#target encoding
from category_encoders.target_encoder import TargetEncoder
#category encoding
from category_encoders import WOEEncoder

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

from itertools import combinations

import gc; gc.enable()

#setting to suppress SettingWithCopy
pd.set_option('mode.chained_assignment', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#set random seed
SEED = 42
np.random.seed(SEED)

# Any results you write to the current directory are saved as output.




#######preprocess data#######
def preprocess(df_train, add_interact = 0, add_all_combs = 0):
    """This function takes cat-in-dat dataframe 
    and preprocess the columns as follows:
    map binary to T,F values,
    MinMax normalizes cyclic_cols,
    label encode the ordinal and fix some orders.
    It can add selected "best" interactions between features, 
    i.e by setting add_interact=1,
    or add interactions from all combinations of features,
    i.e. add_all_combs=1.
    """
    df = df_train.copy()
    
    #group columns by type
    bin_cols = [col for col in df.columns if 'bin' in col]
    nom_cols = [col for col in df.columns if 'nom' in col]
    ord_cols = [col for col in df.columns if 'ord' in col]
    cyclic_cols = ['day','month']
   
    #relabel the binary columns to 0,1
    for col in ['bin_3', 'bin_4']:
        df[col] = df[col].map({"T": 1, "F": 0, "Y":1, "N": 0})
        
    #transform bin columns to T,F for woe encoding
    bin_cols = [col for col in df.columns if 'bin' in col]
    for col in bin_cols:
        df[col] = df[col].apply(lambda x: 'T' if x==1 else 'F')
           
    #transform cyclic columns for woe
    for col in cyclic_cols:
        df[col] = df[col].astype('str')
        
#     #Minmax normalization 
#     for col in cyclic_cols:
#         df[col] = (df[col] - df[col].min()) / \
#                   (df[col].max() - df[col].min())

    #ordinary variables: transform to categorical features
    for col in ord_cols:
        df[col] = df[col].astype('category').cat.as_ordered()
      
    #change order for the specific ordinal columns
    df.ord_1 = df.ord_1.cat.reorder_categories(['Novice','Contributor',
                                                'Expert','Master','Grandmaster'])
    df.ord_2 = df.ord_2.cat.reorder_categories(['Freezing', 'Cold','Warm','Hot',
                                                'Boiling Hot','Lava Hot'])


    #Label encode the ordinary variables and normalize
    for col in ord_cols:
        df[col] = df[col].cat.codes
        df[col] = df[col] / df[col].nunique()
        

    #Add interactions
    if add_interact:
        add_combs = [('nom_0', 'nom_5'), ('nom_0', 'nom_4'), ('nom_2', 'nom_3')]                     + [('bin_1', 'bin_4'), ('bin_0', 'bin_1'), ('bin_3', 'bin_4')]                     + [('bin_1', 'nom_0'), ('bin_1', 'nom_5'), ('bin_4', 'nom_4')]                     + [('month', 'nom_0'), ('month', 'bin_1'), ('day', 'nom_0')]
        
        if add_all_combs:
            no_target_cols = [col for col in df.columns if col not in ['target']]
            add_combs = combinations(no_target_cols,2)

        for comb in add_combs:
            df[str(comb)] = list(zip(df[comb[0]],df[comb[1]]))

    return df


#functions
def get_score(model, X_train, y_train, X_test, y_test, _print = 1, _gam = 0):
    """This function takes trained model instance, train and test data.
    It fits the model and computes ROC AUC score and accuracy
    for test and train respectively."""
        
    #predicted probas, to account for different format for GAM model output:
    if _gam:
        y_test_prob = model.predict_proba(X_test)
        y_train_prob = model.predict_proba(X_train)      
    else:
        y_test_prob = model.predict_proba(X_test)[:,1]
        y_train_prob = model.predict_proba(X_train)[:,1]       
  
    #compute the scores
    #roc auc score
    auc_score_test = roc_auc_score(y_test, y_test_prob)
    auc_score_train = roc_auc_score(y_train, y_train_prob)
    auc = [auc_score_test, auc_score_train]

    #accuracy
    accuracy_test = accuracy_score(y_test, model.predict(X_test))    
    accuracy_train = accuracy_score(y_train,model.predict(X_train))
    acc = [accuracy_test, accuracy_train]
    
    if _print:
        print(f'test ROC_AUC = {round(auc_score_test,5)},               train ROC_AUC = {round(auc_score_train,5)}', 
              f'test Accuracy = {round(auc_score_train,5)}, \
              train Accuracy = {round(accuracy_train,5)}', sep='\n')
    return auc, acc


def get_score_cv(model, X, y, scoring='roc_auc',cv=5,_print = 1):
    """This function takes trained model, features and target
    and scoring metric. It returns the cv score vector."""

    y_pred = model.predict_proba(X)[:,1]
    crossValScores = cross_val_score(model, X, y, cv=10,scoring=scoring)
    score_mean = crossValScores.mean()
    score_std = crossValScores.std()
    
    if _print:
        print(f"CV {scoring} score is:               {score_mean.round(4)} +/- {score_std.round(4)}") 
    
    return crossValScores


def plot_target_dist(df, cols, figsize = (16,10), grid_r=3, grid_c=3):
    
    grid = gridspec.GridSpec(grid_r,grid_c) # The grid of chart
    fig = plt.figure(figsize=figsize) # size of figure
    total = df.shape[0] # total number of observations

    # loop to get column and the count of plots
    for n, col in enumerate(df[cols]): 
        ax = plt.subplot(grid[n]) # feeding the figure of grid
        
        #for low cardinality data
        if df[col].nunique() < 14:
            #count plot
            sns.countplot(x=col, data=df, hue='target', palette='Paired',
                          order=df[col].sort_values().unique(),ax=ax) 
            #df.groupby([col,'target'])[col].count().unstack(level=1)\
            #  .plot(kind='bar', color = ["#a6cee3", "#1f78b4"], width=0.8, ax=ax)
            sizes=[] # Get highest values in y
            for p in ax.patches: # loop to all objects
                height = p.get_height()
                sizes.append(height)
                ax.text(p.get_x()+p.get_width()/2.,
                        height * 1.02,
                        '{:1.1f}%'.format(height/total*100),
                        ha="center", fontsize=14) 
            ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

        #for high cardinality data
        else:
            df_col = df.groupby([col,'target'])[col].count()                       .unstack(level=1).fillna(0).sort_index()
            #define the plot type 
            if df[col].nunique() < 200:
                df_col.plot(kind='bar', ax=ax, stacked=True,
                            color = ["#a6cee3", "#1f78b4"], width=1)
            else: 
                df_col.plot(kind='line', ax=ax, 
                            color = ["#a6cee3", "#1f78b4"])
                        
            #force number of xticks to show
            ax.xaxis.set_major_locator(plt.MaxNLocator(20) )
            
            
        #set labels
        ax.set_ylabel('Count', fontsize=15) # y axis label
        ax.set_title(f'{col} distribution by target', fontsize=16) # title label
        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label
        _xlim = ax.get_xlim()
        
        #calculate pct of class 1
        d = df.groupby([col,'target'])[col].count()              .unstack(level=1).fillna(0).sort_index()
        d['class1_pct'] = d[1] / (d.sum(axis=1))
        if (d.index.dtype == 'int'):
            d.index = d.index - d.index.min()
        elif (d.index.dtype == 'float'):
            #to fix the scale
            d.index = d.index * df[col].nunique()
        
        #add another y-axis to show the pct of class 1
        ax2 = ax.twinx()
        if df[col].nunique() < 200:
            d.class1_pct.plot(marker='o',markersize=5,ax=ax2,color=["#6a3d9a"])
        else:
            d.class1_pct.plot(marker='o',markersize=5,linewidth=0,
                              ax=ax2,color=["#6a3d9a"])
            
        ax2.set_ylabel('class 1 fraction', color="#6a3d9a", fontsize=15)
        ax2.set_xlim(_xlim)
        ax2.set_ylim([-0.1,1.1])
        ax2.grid(False)
        
    #!!!!!!!need to fix missing xlabels, problem appears when twinx axis is added
    plt.tight_layout()
    plt.show()




'''Read in train and test data from csv files'''
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv',index_col=0)
df_test = pd.read_csv('../input/cat-in-the-dat/test.csv',index_col=0)

#group columns by type
target = 'target'
bin_cols = [col for col in df_train.columns if 'bin' in col]
cyclic_cols = ['day','month']
ord_cols = [col for col in df_train.columns if 'ord' in col]
nom_cols = [col for col in df_train.columns if 'nom' in col]
no_target = [col for col in df_train.columns if 'target' not in col]

#preprocess data 
###########preprocess###########
df = preprocess(df_train)
#################################

X = df[no_target]
y = df[target]

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3333, 
                                                    random_state=SEED)
display(X_train.head())

#delete df
del df; gc.collect()




#Logistic regression model
logit = LogisticRegression(random_state=SEED,class_weight='balanced',solver='liblinear')




##install the package
get_ipython().system('pip install vtreat')




#importing the package
import vtreat

#apply vtreat
vtreat_enc = vtreat.BinomialOutcomeTreatment(
    outcome_name='target',    # outcome variable
    outcome_target=True) # outcome of interest

#vtreat encoding using different techniques
#here use all features, including high-cardinality
X_train_vtreat = vtreat_enc.fit_transform(X_train, y_train)
X_test_vtreat = vtreat_enc.transform(X_test)

#feature selection: only recommended
vtreat_fs = vtreat_enc.score_frame_[vtreat_enc.score_frame_.recommended == True]                      .sort_values(by = 'significance')
print(f'There are {len(vtreat_fs)} recommended features.')

#display first 10 recommended features
vtreat_fs.head(10)




#use all recommended features
used_cols = vtreat_fs.variable.to_list()

#fit the model
logit.fit(X_train_vtreat[used_cols], y_train)
#print the ROC-AUC score and Accuracy
get_score(logit, X_train_vtreat[used_cols], 
                         y_train, X_test_vtreat[used_cols], y_test)
print('\n')
get_score_cv(logit, X_train_vtreat[used_cols], y_train)




#pd.DataFrame(vtreat_fs.orig_variable.value_counts())
print('Included recommended original features are: ', vtreat_fs.orig_variable.unique())




#how many encoded features per original variable?
print("\nExample for 'day' feature:")
vtreat_fs[vtreat_fs.orig_variable == 'day']




#select only best among recomennded for a given original variable
select_cols = []
for var in vtreat_fs.orig_variable.value_counts().index:
    #select only one encoded feature for 
    select_cols.append(vtreat_fs[vtreat_fs.orig_variable == var].iloc[0,0])
    
#print(select_cols)
print(f'There are {len(select_cols)} selected features out of {len(vtreat_fs)} recommended.')
select_cols




#use all recommended features
used_cols = select_cols

#fit the model
logit.fit(X_train_vtreat[used_cols], y_train)
#print the ROC-AUC score and Accuracy
get_score(logit, X_train_vtreat[used_cols], 
                         y_train, X_test_vtreat[used_cols], y_test)
print('\n')
get_score_cv(logit, X_train_vtreat[used_cols], y_train)




#define feature sets
no_high_card_cols = bin_cols + cyclic_cols + nom_cols[:5] + ord_cols 
#all features for modeling
used_cols = bin_cols + cyclic_cols + nom_cols + ord_cols 

#WoE encoding
woe_enc = WOEEncoder(random_state=SEED, randomized=True).fit(X_train[used_cols], y_train)
X_train_woe = woe_enc.transform(X_train[used_cols].reset_index(drop=True))
X_test_woe = woe_enc.transform(X_test[used_cols].reset_index(drop=True))




#smoothing function
def sig(n,m,sh):
    return 1/(1+np.exp(-m*(n-sh)))

#manually optimized parameters
sig_params = {'nom_5':[10000,0.0001], 'nom_6':[8000,0.00005],
              'nom_7':[5000,0.002],'nom_8':[1000,0.0015],
              'nom_9':[300,0.01]}




### Target smooting
y_avg = y_train.mean()

for col in nom_cols[5:]:
    #smothing parameters
    m = sig_params[col][0]
    sh = sig_params[col][1]
    
    #targer encoding
    t_enc = TargetEncoder(smoothing=10.0)
    X_train[col + '_te'] = t_enc.fit_transform(X_train[col],y_train)
    X_test[col + '_te'] = t_enc.transform(X_test[col])

    ### FREQUENCY ENCODING and weight
    ccol = col + '_te'
    #for ccol in [col + '_te']:
    # size of each category
    encoding = X_train.groupby(ccol).size()
    # get frequency of each category
    encoding = encoding/len(X_train)
    X_train[ccol + '_freq'] = X_train[ccol].map(encoding)
    X_train[ccol + '_weight'] = sig(X_train[ccol + '_freq'],m,sh)

    #encoding the test set
    encoding = X_test.groupby(ccol).size()
    encoding = encoding/len(X_test)
    X_test[ccol + '_freq'] = X_test[ccol].map(encoding)

    #target smooting
    n = X_train[ccol + '_freq']
    y_est = np.exp(X_train[ccol])/(1+np.exp(X_train[ccol]))
    y_adj = sig(n,m,sh)*y_est + (1-sig(n,m,sh))*y_avg
    X_train[ccol.strip('_te') + '_adj'] = np.log(y_adj / (1-y_adj))

    #test set
    n = X_test[ccol + '_freq']
    y_est = np.exp(X_test[ccol])/(1+np.exp(X_test[ccol]))
    y_adj = sig(n,m,sh)*y_est + (1-sig(n,m,sh))*y_avg
    X_test[ccol.strip('_te') + '_adj'] = np.log(y_adj / (1-y_adj))

    #X_train[[col + '_te',col + '_adj',col + '_te_freq',col + '_te_weight']].head(10)

    #visualize how low in number observations are suppressed for nom_9
    if col == 'nom_9':
        fig, (ax,ax3) = plt.subplots(1,2,figsize=(12,4))

        #plot frequency of given observation
        X_train[col + '_te_freq'].plot(kind='hist',bins=20, ax=ax)

        #add second y-axis
        x = np.linspace(0,X_train[col + '_te_freq'].max(),30)#ax.get_xticks()
        ax2 = ax.twinx()
        ax2.plot(x,sig(x,m,sh),color="#6a3d9a")
        ax2.set_ylabel('weight', color="#6a3d9a", fontsize=15)
        ax2.set_ylim([-0.1,1.1])
        ax2.grid(False)
        ax.set_xlabel('count_pct')
        ax.set_title(f'Distribution of count percentage for {col}.')

        X_train_woe[col].plot(kind='hist',ax=ax3,color="#fb9a99",label='original WoE',alpha=0.5)
        X_train[col + '_adj'].plot(kind='hist',ax=ax3,color="#33a02c",label='adjusted WoE',alpha=0.5)
        ax3.legend()
        ax3.set_title(f'Comparison of initial and smoothed WoE for {col}')
        plt.tight_layout()

    #drop no longer needed columns
    X_train.drop([ccol for ccol in X_train.columns if col + '_te' in ccol],axis=1,inplace=True); gc.collect()
    X_test.drop([ccol for ccol in X_test.columns if col + '_te' in ccol],axis=1,inplace=True); gc.collect()

#     #add adjusted
#     X_tn = pd.concat([X_train_woe[no_high_card_cols],
#                       X_train[[col + '_adj']].reset_index(drop=True)],axis=1)
#     X_ts = pd.concat([X_test_woe[no_high_card_cols],
#                       X_test[[col + '_adj']].reset_index(drop=True)],axis=1)

#     #fit the model
#     logit.fit(X_tn, y_train)
#     #print the ROC-AUC score and Accuracy
#     utility.get_score(logit, X_tn, y_train, X_ts, y_test)

#     #delete temporary variables
#     del X_tn, X_ts; gc.collect()




#adjusted nominal features with vtreat encoding for the rest
nom_adj = [col for col in X_train.columns if '_adj' in col]
no_high_card_vtreat_cols = [col for col in select_cols 
                            if col not in 
                            [col + '_logit_code' for col in nom_cols[5:]]]

X_tn = pd.concat([X_train_vtreat[no_high_card_vtreat_cols],
                      X_train[nom_adj].reset_index(drop=True)],axis=1)
X_ts = pd.concat([X_test_vtreat[no_high_card_vtreat_cols],
                      X_test[nom_adj].reset_index(drop=True)],axis=1)

#fit the model
logit.fit(X_tn, y_train)
#print the ROC-AUC score and Accuracy
get_score(logit, X_tn, y_train, X_ts, y_test)
print('\n')
get_score_cv(logit, X_tn, y_train)




#WoE encoding with high-cardinality data, no smoothing
#to demonstrate that high-cardinality nominal feature cause overfitting
all_cols = bin_cols + cyclic_cols + nom_cols + ord_cols 

used_cols = all_cols
#fit the model
logit.fit(X_train_woe[used_cols], y_train)
#print the ROC-AUC score and Accuracy
get_score(logit, X_train_woe[used_cols], y_train, X_test_woe[used_cols], y_test)
print('\n')
get_score_cv(logit, X_train_woe[used_cols], y_train)




#WoE encoding without high-cardinality data
used_cols = no_high_card_cols

#fit the model
logit.fit(X_train_woe[no_high_card_cols], y_train)
#print the ROC-AUC score and Accuracy
get_score(logit, X_train_woe[used_cols], y_train, X_test_woe[used_cols], y_test)
print('\n')
get_score_cv(logit, X_train_woe[used_cols], y_train)




#WoE encoding with high-cardinality data, with smoothing
nom_adj = [col for col in X_train.columns if '_adj' in col]
no_high_card_cols = bin_cols + cyclic_cols + nom_cols[:5] + ord_cols 

X_tn = pd.concat([X_train_woe[no_high_card_cols],
                      X_train[nom_adj].reset_index(drop=True)],axis=1)
X_ts = pd.concat([X_test_woe[no_high_card_cols],
                      X_test[nom_adj].reset_index(drop=True)],axis=1)

#fit the model
logit.fit(X_tn, y_train)
#print the ROC-AUC score and Accuracy
get_score(logit, X_tn, y_train, X_ts, y_test)
print('\n')
get_score_cv(logit, X_tn, y_train)




#plot the roc_auc curve
y_score = logit.decision_function(X_ts)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
    
plt.figure()
plt.plot(fpr, tpr, color='darkgreen',
         lw=2, label='ROC curve (area = %0.3f)' %roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('Receiver operating characteristic', fontsize=15)
plt.legend(loc="lower right", fontsize=15)
plt.show()

