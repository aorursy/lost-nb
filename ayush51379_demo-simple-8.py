#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:



train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')


# In[3]:


train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


# In[4]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[5]:


def test_in_sub(test):
    tgms = test.groupby('installation_id').last().game_session
    tgms1 = tgms.reset_index()
    test_ass = test[test.type == "Assessment"]
    tgms1["title"] = str(test_ass[test_ass.game_session == tgms1["game_session"][0]].title.reset_index(drop=True)[0])
    
    for i in range(0,len(tgms1)):
        tgms1["title"][i] = str(test_ass[test_ass.game_session==tgms1["game_session"][i]].title.reset_index(drop=True)[0])
    return tgms1


# In[6]:


def c_accuracy_group(df):
    df["accuracy_group"]=0
    for i in range(0,len(df)):
        acc = float(df["accuracy"][i])
        if (acc == float(0)):
            df["accuracy_group"][i]=0
        elif (acc < float(0.5)):
            df["accuracy_group"][i]=1
        elif (acc < float(1)):
            df["accuracy_group"][i]=2
        elif (acc == float(1)):
            df["accuracy_group"][i]=3
        else:
            df["accuracy_group"][i] = None
    return df
            


# In[7]:


def test_to_label(test):
    print("Converting to label format, as of submissions done in assessment")
    test_ass = test[test.type == "Assessment"]
    test_ass_sub = test_ass[(((test_ass.event_code == 4100) & (test_ass.title != 'Bird Measurer (Assessment)'))) | (((test_ass.event_code == 4110) & (test_ass.title == 'Bird Measurer (Assessment)')))]
    test_ass_sub_inf = test_ass_sub[["installation_id","game_session","timestamp","title","event_data"]]
    test_ass_sub_inf0 = test_ass_sub_inf
    test_ass_sub_inf0["correct"] = 0
    test_ass_sub_inf0["incorrect"] = 0
    
    for i in range(0,len(test_ass_sub_inf0)):
        if "\"correct\":true" in test_ass_sub_inf0["event_data"][test_ass_sub_inf0.index[i]]:
            test_ass_sub_inf0["correct"][test_ass_sub_inf0.index[i]] = 1
        else:
            test_ass_sub_inf0["incorrect"][test_ass_sub_inf0.index[i]] = 1
    test_ass_sub_inf1 = test_ass_sub_inf0.groupby(by=["installation_id","game_session","title"],sort=False).sum()
    test_ass_sub_inf2 = test_ass_sub_inf1
    test_ass_sub_inf2 = test_ass_sub_inf2.reset_index()
    test_ass_sub_inf2["accuracy"] =float(0)
    
    for i in range(0,len(test_ass_sub_inf2)):
        corr = test_ass_sub_inf2["correct"][i]
        incor = test_ass_sub_inf2["incorrect"][i]
        test_ass_sub_inf2["accuracy"][i] = float(corr)/(incor+corr)
    
    test_ass_sub_inf3 = test_ass_sub_inf2
    test_ass_sub_inf3 = c_accuracy_group(test_ass_sub_inf3)
    return test_ass_sub_inf3
    


# In[8]:


def get_time_gm(train, train_labels):
    print("Adding cumulative game played time for each session")
    train_data_1 = train[["installation_id", "game_session", "title","game_time"]]
    train_time_god_2 = train[["installation_id", "game_session", "title","game_time"]]
    ttg_time = train_time_god_2.groupby(by=['game_session'], sort=False).last().game_time.reset_index()
    ttg_time0 = train.groupby(by = ["installation_id", "game_session", "title"],sort=False).size().reset_index().drop(columns=[0]).merge(ttg_time, on = 'game_session', how = 'left')
    ttg_time00 = ttg_time0.groupby(by=['installation_id','game_session'], sort=False).sum().groupby(level=[0]).cumsum()
    ttg_time1 = ttg_time00.reset_index()
    ttg_time2 = ttg_time1[["game_session","game_time"]]
    train_labels1 = train_labels
    # join train with train labels
    train_labels_t = train_labels1.merge(ttg_time2, on = 'game_session', how = 'left')

    return train_labels_t


# In[9]:



def get_final_feat2(train, train_labels_derive_time):   
    # train, train_labels_derive_time
    print("Adding more time sensitive features such as: all correct, all incorrect,avg_corr_all,avg_incorr_all,avg_inacc_r, avg_acc_r,score, average score, corr to incorr ratio and vice versa until any game session")
    
    train_edit_c = train[["installation_id", "game_session", "title", "event_data"]]
    trc_ic = train_edit_c[(train_edit_c.event_data.str.contains("\"correct\":true") | train_edit_c.event_data.str.contains("\"correct\":false")) ]
    trc_ic1 = trc_ic.groupby(["installation_id", "game_session", "title"]).size().reset_index().drop(columns = [0])
    
    trc = train_edit_c[train_edit_c.event_data.str.contains("\"correct\":true")]
    trc_edit = trc[["installation_id","game_session"]]
    trc_edit["correct_all"] = 1
    trc_edit_all1 = trc_edit.groupby(by=['installation_id','game_session'], sort = False).sum().groupby(level=[0]).cumsum()
    trc_edit_all1 = trc_edit_all1.reset_index()
    
    tric = train_edit_c[train_edit_c.event_data.str.contains("\"correct\":false")]
    tric_edit = tric[["installation_id","game_session"]]
    tric_edit["incorrect_all"] = 1
    tric_edit_all1 = tric_edit.groupby(by=['installation_id','game_session'], sort=False).sum().groupby(level=[0]).cumsum()
    tric_edit_all1 = tric_edit_all1.reset_index()
    
    print("Adding correct all and incorrect all feature, later we might wanna add specific accuracy groups of titles/assesssments, to record history of gameplay of user")
    # join train with train labels
    train_c_1 = trc_ic1.merge(trc_edit_all1[["game_session","correct_all"]], on = 'game_session', how = 'left')
    train_c_2 = train_c_1.merge(tric_edit_all1[["game_session","incorrect_all"]], on = 'game_session', how = 'left')
    
    # join train with train labels
    train_c_2["correct_all"].fillna(0, inplace=True)
    train_c_2["incorrect_all"].fillna(0, inplace=True)
    to_get_acc = train_c_2 # contains all the gms with either true or false
    
    print("Adding score and score count")
    to_get_acc1 = to_get_acc
    
    to_get_acc1["score"] = 0.000001
    to_get_acc1["score_c"] = 0
    to_get_acc1["acc_r_single"] = 0.000001
    to_get_acc1["inacc_r_single"] = 0.000001
   # to_get_acc1["acc_r"] = 0.000001
    #to_get_acc1["inacc_r"] = 0.000001
    
   
    for i in range(0,len(to_get_acc1)):
        acc = to_get_acc1["correct_all"][i]
        ina = to_get_acc1["incorrect_all"][i]
        if((acc == 0) and (ina) == 0):
            to_get_acc1["score_c"][i] = 0
            to_get_acc1["score"][i] = 0
            to_get_acc1["acc_r_single"][i] = 0
            to_get_acc1["inacc_r_single"][i] = 0
         #   to_get_acc1["acc_r"][i] = 0
         #   to_get_acc1["inacc_r"][i] = 0
            continue
        elif(acc == 0):
            to_get_acc1["score"][i] = round(float(ina),3)*(-5)
            to_get_acc1["score_c"][i] = 1
            to_get_acc1["acc_r_single"][i] = 0
            to_get_acc1["inacc_r_single"][i] = ina
            #to_get_acc1["acc_r"][i] = 0
          #  to_get_acc1["inacc_r"][i] = ina
        elif(ina == 0):
            to_get_acc1["score"][i] = round(float(acc),3)*(5)
            to_get_acc1["score_c"][i] = 1
            to_get_acc1["acc_r_single"][i] = acc
            to_get_acc1["inacc_r_single"][i] = 0
          #  to_get_acc1["acc_r"][i] = acc
          #  to_get_acc1["inacc_r"][i] = 0
        elif((ina != 0) and (acc != 0)):
            to_get_acc1["score"][i] = round((float(acc)),3)*3-round((float(ina)),3)*1
            to_get_acc1["score_c"][i] = 1
            to_get_acc1["acc_r_single"][i] = round(float(acc)/ina,3)
            to_get_acc1["inacc_r_single"][i] = round(float(ina)/acc,3)
           # to_get_acc1["acc_r"][i] = round(float(acc)/ina,3)
           # to_get_acc1["inacc_r"][i] = round(float(ina)/acc,3)
            
    #to_get_acc1
    
    train_copy = train[["installation_id", "game_session", "title"]].groupby(by = ["installation_id","game_session","title"], sort=False).size().reset_index()
    # join train with train labels
    train_t_1 = train_copy.drop(columns=[0]).merge(to_get_acc1[["game_session","correct_all", "incorrect_all","acc_r_single","inacc_r_single", "score", "score_c"]], on = 'game_session', how = 'left')
    train_t_2 = train_t_1
    train_t_2["correct_all"].fillna(0, inplace=True)
    train_t_2["incorrect_all"].fillna(0, inplace=True)
    train_t_2["score"].fillna(0, inplace=True)
    train_t_2["score_c"].fillna(0, inplace=True)
    train_t_2["acc_r_single"].fillna(0, inplace=True)
    train_t_2["inacc_r_single"].fillna(0, inplace=True)
    
    train_t_3 = train_t_2
    train_t_3 = train_t_3.groupby(by=['installation_id','game_session','title'],sort=False).sum().groupby(level=[0]).cumsum()
    train_t_3 = train_t_3.reset_index()
    
    print("Adding average score, avg acc ratio and avg inacc ratio")
    train_t_4 = train_t_3
    # now count average score till that point, acc_r, inacc_r
    train_t_4["average_score"] = float(0)
    train_t_4["acc_r"] = float(0)
    train_t_4["inacc_r"] = float(0)
    train_t_4["avg_acc_r"] = float(0)
    train_t_4["avg_inacc_r"] = float(0)
    for i in range(0,len(train_t_4)):
        acc = train_t_4["correct_all"][i]
        inacc = train_t_4["incorrect_all"][i]
        score = round(float(train_t_4["score"][i]))
        count = train_t_4["score_c"][i]
        acc_r_single = train_t_4["acc_r_single"][i]
        inacc_r_single = train_t_4["inacc_r_single"][i]
        
        if (count!=0):
            train_t_4["average_score"][i] = round(float(score)/count,3)
            train_t_4["avg_acc_r"][i] = round(float(acc_r_single)/count,3)
            train_t_4["avg_inacc_r"][i] = round(float(inacc_r_single)/count,3)
        else:
            train_t_4["average_score"][i] = 0
            train_t_4["avg_acc_r"][i] = 0
            train_t_4["avg_inacc_r"][i] = 0
        if((inacc == 0)&(acc == 0)):
            train_t_4["acc_r"][i] = 0
            train_t_4["inacc_r"][i] = 0
        elif(inacc == 0):
            train_t_4["acc_r"][i] = acc
            train_t_4["inacc_r"][i] = 0
        elif(acc == 0):
            train_t_4["acc_r"][i] = 0
            train_t_4["inacc_r"][i] = inacc
        elif((inacc != 0) & (acc != 0)):
            train_t_4["acc_r"][i] = round(float(acc)/inacc,3)
            train_t_4["inacc_r"][i] = round(float(inacc)/acc,3)
                
    train_t_5 = train_t_4        
    
    print("Almost done, combining all into label format")
    # join train with train labels
    train_labels_derive_time_corr = train_labels_derive_time.merge(train_t_5[["game_session","correct_all","incorrect_all","avg_acc_r","avg_inacc_r","acc_r_single","inacc_r_single","score","score_c","average_score","acc_r","inacc_r"]], on = 'game_session', how = 'left')
    return train_labels_derive_time_corr


# In[10]:


def get_misses_all(train,train_all_other):
    train_miss = train[train.event_data.str.contains("misses")]
    train_miss1 = train_miss.reset_index()
    train_miss1["misses"] = 0
    for i in range(0,len(train_miss1)):
        miss = eval(train_miss1.event_data[train_miss1.event_data.index[i]])["misses"]
        train_miss1["misses"][i] = miss
    train_miss2 = train_miss1[["installation_id", "game_session","title","world","misses"]]
    train_miss3 = train_miss2
    train_miss3c = train_miss3[train_miss3.world=="CRYSTALCAVES"]
    train_miss3m = train_miss3[train_miss3.world=="MAGMAPEAK"]
    train_miss3t = train_miss3[train_miss3.world=="TREETOPCITY"]
    
    train_miss3c["missc"] = train_miss3c["misses"]
    train_miss3m["missm"] = train_miss3m["misses"]
    train_miss3t["misst"] = train_miss3t["misses"]
    
    train_miss4c = train_miss3c.groupby(by = ["installation_id","game_session"],sort=False).sum().reset_index()#.groupby(level=[0]).cumsum().reset_index()
    train_miss4m = train_miss3m.groupby(by = ["installation_id","game_session"],sort=False).sum().reset_index()#.groupby(level=[0]).cumsum().reset_index()
    train_miss4t = train_miss3t.groupby(by = ["installation_id","game_session"],sort=False).sum().reset_index()#.groupby(level=[0]).cumsum().reset_index()
    train_miss4 = train_miss3.groupby(by = ["installation_id","game_session"],sort=False).sum().reset_index()#.groupby(level=[0]).cumsum().reset_index()
    train_data_1 = train.groupby(by = ["installation_id", "game_session", "title"],sort=False).size().reset_index().drop(columns=[0])
    
    train_missc = train_data_1.merge(train_miss4c[["game_session", "missc"]], on = 'game_session', how = 'left')
    train_misscm = train_missc.merge(train_miss4m[["game_session", "missm"]], on = 'game_session', how = 'left')
    train_misscmt = train_misscm.merge(train_miss4t[["game_session", "misst"]], on = 'game_session', how = 'left')
    train_missall = train_misscmt.merge(train_miss4[["game_session", "misses"]], on = 'game_session', how = 'left')
    train_missall["misses"].fillna(0, inplace=True)
    train_missall["missc"].fillna(0, inplace=True)
    train_missall["missm"].fillna(0, inplace=True)
    train_missall["misst"].fillna(0, inplace=True)
    
    train_missall1 = train_missall.groupby(by=["installation_id","game_session"],sort=False).sum().groupby(level=[0]).cumsum().reset_index()
    
    train_all_other1 = train_all_other.merge(train_missall1[['game_session','misses','missc','missm','misst']], on = 'game_session', how = 'left')
 
    return train_all_other1


# In[11]:


def get_all2(train):
    train_labels_derive2 = test_to_label(train)
    train_labels_derive_time2 = get_time_gm(train,train_labels_derive2)
    get2 = get_final_feat2(train, train_labels_derive_time2)
    get_miss = get_misses_all(train,get2)
    return get_miss


# In[12]:


def get_sub2(test):
    test_labels_derive2 = test_in_sub(test)
    test_labels_derive_time2 = get_time_gm(test,test_labels_derive2)
    get2 = get_final_feat2(test, test_labels_derive_time2)
    get_miss = get_misses_all(test,get2)
    return get_miss


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from scipy.stats import mode

#modeling
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

#LGB imports
import lightgbm as lgb


# In[14]:


import numpy as np
import pandas as pd
import datetime
from catboost import CatBoostClassifier
from time import time
from tqdm import tqdm_notebook as tqdm


# In[15]:


from sklearn.metrics import confusion_matrix
# this function is the quadratic weighted kappa (the metric used for the competition submission)
def qwk(act,pred,n=4,hist_range=(0,3)):
    
    # Calculate the percent each class was tagged each label
    O = confusion_matrix(act,pred)
    # normalize to sum 1
    O = np.divide(O,np.sum(O))
    
    # create a new matrix of zeroes that match the size of the confusion matrix
    # this matriz looks as a weight matrix that give more weight to the corrects
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    # make two histograms of the categories real X prediction
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    # multiply the two histograms using outer product
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E)) # normalize to sum 1
    
    # apply the weights to the confusion matrix
    num = np.sum(np.multiply(W,O))
    # apply the weights to the histograms
    den = np.sum(np.multiply(W,E))
    
    return 1-np.divide(num,den)
    


# In[16]:


# this function makes the model and sets the parameters
# for configure others parameter consult the documentation below:
# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
def make_classifier1():
    clf1 = CatBoostClassifier(
                               loss_function='MultiClass',
                                # eval_metric="AUC",
                               task_type="CPU",
                               learning_rate=0.01,
                               iterations=2000,
                               od_type="Iter",
                               verbose=None,
                               depth=8,
                               early_stopping_rounds=50,
                                #l2_leaf_reg=1,
                                #border_count=96,
                               random_seed=42
                              )
        
    return clf1


# In[17]:


get2_test = get_all2(test)


# In[18]:


get2_test.to_csv("get2_test.csv")


# In[19]:


get2_train = get_all2(train)


# In[20]:


get2_train.to_csv("get2_train.csv")


# In[21]:


get2_sub = get_sub2(test)


# In[22]:


get2_sub.to_csv("get2_sub.csv")


# In[23]:


get2A = get2_train.drop(columns = ["installation_id","game_session"]).sample(frac=1,random_state = 42).reset_index(drop=True)


# In[24]:


get2A_test = get2_test.drop(columns = ["installation_id","game_session"]).sample(frac=1,random_state = 42).reset_index(drop=True)


# In[25]:


get2A_sub = get2_sub.drop(columns = ["installation_id","game_session"])#.sample(frac=1,random_state = 42).reset_index(drop=True)


# In[26]:


get2A.to_csv("get2A_train.csv")


# In[27]:


labels_map = {"Mushroom Sorter (Assessment)":1,"Bird Measurer (Assessment)":2,"Cauldron Filler (Assessment)":3,"Chest Sorter (Assessment)":4,"Cart Balancer (Assessment)":5}


# In[28]:


world_map = {"Mushroom Sorter (Assessment)":1,"Bird Measurer (Assessment)":1,"Cauldron Filler (Assessment)":2,"Chest Sorter (Assessment)":3,"Cart Balancer (Assessment)":3}


# In[29]:


get2A['world'] = get2A['title'].map(world_map)
get2A_test['world'] = get2A_test['title'].map(world_map)

get2A['title'] = get2A['title'].map(labels_map)
get2A_test['title'] = get2A_test['title'].map(labels_map)


# In[30]:


get2A_sub['world'] = get2A_sub['title'].map(world_map)
get2A_sub['title'] = get2A_sub['title'].map(labels_map)


# In[31]:


get2A.to_csv("get2A_train.csv", index=None)


# In[32]:


get2A_test.to_csv("get2A_test.csv")


# In[33]:


get2A_sub.to_csv("get2A_sub.csv")


# In[34]:


get2_train


# In[35]:


get2A


# In[36]:


get2A1_test = get2A_test.drop(columns = ["accuracy_group","correct","incorrect","accuracy"])
get2A1_test["accuracy_group"] = get2A_test["accuracy_group"]
get2A1_test


# In[37]:


get2A1 = get2A.drop(columns = ["accuracy_group","correct","incorrect","accuracy"])
get2A1["accuracy_group"] = get2A["accuracy_group"]
get2A1


# In[38]:


Xmod1 = get2A.drop(columns = ["accuracy_group","correct","incorrect","accuracy"])
Ymod1 = get2A["accuracy_group"]


# In[39]:


Xmod1


# In[40]:


# CART Classification
#from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
dataframe = get2A1
array = dataframe.values
X = array[:,0:18]
Y = array[:,18]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[41]:


model.fit(X,Y)


# In[42]:


y_pr = model.predict(X)
qwk(Y,y_pr)


# In[43]:


dataframe1 = get2A1_test
array1 = dataframe1.values
X1 = array1[:,0:18]
Y1 = array1[:,18]
y_pr1 = model.predict(X1)
qwk(Y1,y_pr1)


# In[44]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring='f1_macro')
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[45]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring='f1_micro')
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[46]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(cohen_kappa_score))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[47]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CAT',CatBoostClassifier()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(cohen_kappa_score))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[48]:


X1


# In[49]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CAT',CatBoostClassifier(verbose=999)))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(cohen_kappa_score))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[50]:


get2A1


# In[51]:


get2A1_1 = get2A1[get2A1.title == 1]
get2A1_2 = get2A1[get2A1.title == 2]
get2A1_3 = get2A1[get2A1.title == 3]
get2A1_4 = get2A1[get2A1.title == 4]
get2A1_5 = get2A1[get2A1.title == 5]
df1 = get2A1_1.values
df2 = get2A1_2.values
df3 = get2A1_3.values
df4 = get2A1_4.values
df5 = get2A1_5.values
X_1 = df1[:,1:18]
Y_1 = df1[:,18]
X_2 = df2[:,1:18]
Y_2 = df2[:,18]
X_3 = df3[:,1:18]
Y_3 = df3[:,18]
X_4 = df4[:,1:18]
Y_4 = df4[:,18]
X_5 = df5[:,1:18]
Y_5 = df5[:,18]
get2A1_1


# In[52]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CAT',CatBoostClassifier(verbose=999,early_stopping_rounds=300)))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X_1, Y_1, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[53]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CAT',CatBoostClassifier(verbose=999,early_stopping_rounds=300)))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X_2, Y_2, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[54]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CAT',CatBoostClassifier(verbose=999,early_stopping_rounds=300)))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X_3, Y_3, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[55]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CAT',CatBoostClassifier(verbose=999,early_stopping_rounds=300)))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X_4, Y_4, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[56]:


# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CAT',CatBoostClassifier(verbose=999,early_stopping_rounds=300)))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X_5, Y_5, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[57]:


import lightgbm as lgb
import xgboost as xgb
# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('LGB', lgb.LGBMClassifier(verbose=999)))
models.append(('CAT',CatBoostClassifier(verbose=999)))
models.append(('XGB', xgb.XGBClassifier(verbose=999)))

#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[58]:


from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('LGB', lgb.LGBMClassifier(verbose=999)))
models.append(('CAT',CatBoostClassifier(verbose=999)))
models.append(('XGB', xgb.XGBClassifier(verbose=999)))

#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[59]:


from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
# Compare Algorithms
#from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
# load dataset
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('LGB', lgb.LGBMClassifier(verbose=999)))
models.append(('CAT',CatBoostClassifier(verbose=999)))
models.append(('XGB', xgb.XGBClassifier(verbose=999)))

#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    #cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=scoring)
    cv_results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(qwk))
    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[60]:


# Create a pipeline that extracts features from the data then creates a model
#from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# load data
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
print(features)
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('random forest', RandomForestClassifier()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(qwk))
print(results.mean())
feature_union


# In[61]:


# Create a pipeline that extracts features from the data then creates a model
#from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# load data
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
print(features)
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('Catboost', CatBoostClassifier(verbose=999)))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(qwk))
print(results.mean())
features


# In[62]:


# Create a pipeline that extracts features from the data then creates a model
#from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# load data
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=8)))
print(features)
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('Catboost', CatBoostClassifier(verbose=999)))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X1, Y1, cv=kfold, scoring=make_scorer(qwk))
print(results.mean())
features


# In[63]:


# Create a pipeline that extracts features from the data then creates a model
#from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# load data
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=8)))
print(features)
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('Catboost', CatBoostClassifier(verbose=999)))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold, scoring=make_scorer(qwk))
print(results.mean())
features


# In[ ]:





# In[64]:


dataframe1 = get2A2_test
array1 = dataframe1.values
X1 = array1[:,0:18]
Y1 = array1[:,18]
y_pr1 = model.predict(X1)
qwk(Y1,y_pr1)


# In[ ]:





# In[65]:


Xmod2 = Xmod1.reset_index()
Xmod2


# In[66]:


Xmod2.columns


# In[67]:


Xmod2.describe()


# In[68]:


from numba import jit 

@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))
    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)
    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)
    e = e / a1.shape[0]
    return 1 - o / e


# In[69]:


#define the parameters for lgbm.

SEED = 42
N_FOLD = 10
params = {
    'min_child_weight': 10.0,
    'objective': 'multi:softprob',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.5,
    'num_class':4,
    'learning_rate':0.05,
    'n_estimators':2000,
    'eta': 0.025,
    'gamma': 0.65,
    'eval_metric':'mlogloss'
    }

features = [i for i in final_train_df.columns if i not in ['accuracy_group']]


# In[70]:


X_train = Xmod1
y_train = Ymod1
final_test_df = Xmod1test


# In[ ]:





# In[71]:


def model(train_X,train_Y, test, params, n_splits=N_FOLD):
    
    #define KFold Strategy
    folds = StratifiedKFold(n_splits=N_FOLD,shuffle=True, random_state=SEED)
    scores = []
    
    #out of the fold 
    y_pre = np.zeros((len(test),4), dtype=float)
    target = ["accuracy_group"]
    #print("done")
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_Y)):
        print("------------------------ fold {} -------------------------".format(fold_ + 1))
        
        X_train, X_valid = train_X.iloc[trn_idx], train_X.iloc[val_idx]
        y_train, y_valid = train_Y.iloc[trn_idx], train_Y.iloc[val_idx]
        
        # Convert our data into XGBoost format
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        
        xgb_model = xgb.train(params,
                      d_train,
                      num_boost_round=1600,
                      evals=[(d_train, 'train'), (d_valid, 'val')],
                      verbose_eval=False,
                      early_stopping_rounds=70
                     )
        
        d_val = xgb.DMatrix(X_valid)
        pred_val = [np.argmax(x) for x in xgb_model.predict(d_val)]
        
        #calculate cohen kappa score
        score = cohen_kappa_score(pred_val,y_valid,weights='quadratic')
        scores.append(score)

        pred = xgb_model.predict(xgb.DMatrix(test))
        #save predictions
        y_pre += pred
        
        print(f'Fold: {fold_+1} quadratic weighted kappa score: {np.round(score,4)}')

    pred = np.asarray([np.argmax(line) for line in pred])
    print('Mean choen_kappa_score:',np.round(np.mean(scores),6))
    
    return xgb_model,pred


# In[72]:


xgb_model,pred = model(X_train,y_train,final_test_df,params)


# In[73]:


new_str = str(Xmod2['title'][0])+" "+str(Xmod2['game_time'][0])
new_str


# In[74]:


Xmod2.index[0]


# In[75]:


Xmod2_q = Xmod2["index"].to_frame()
Xmod2_q


# In[76]:


Xmod2_q["new_event"] = "a"
Xmod2_q


# In[77]:


i=0
new_str1 = str(Xmod2['title'][i])+" "+str(Xmod2['game_time'][i])+" "+str(Xmod2['correct_all'][i])+" "+str(Xmod2['incorrect_all'][i])+" "+str(Xmod2['avg_acc_r'][i])+" "+str(Xmod2['avg_inacc_r'][i])+" "+str(Xmod2['acc_r_single'][i])+" "+str(Xmod2['inacc_r_single'][i])+" "+str(Xmod2['score'][i])+" "+str(Xmod2['score_c'][i])+" "+str(Xmod2['average_score'][i])+" "+str(Xmod2['acc_r'][i])+" "+str(Xmod2['inacc_r'][i])+" "+str(Xmod2['misses'][i])+" "+str(Xmod2['missc'][i])+" "+str(Xmod2['missm'][i])+" "+str(Xmod2['misst'][i])+" "+str(Xmod2['world'][i])
new_str1


# In[78]:


get_ipython().run_cell_magic('time', '', 'for i in range(0,len(Xmod2_q)):\n    Xmod2_q["new_event"][i] = str(Xmod2[\'title\'][i]))+" "+str(Xmod2[\'game_time\'][i])+" "+str(Xmod2[\'correct_all\'][i])+" "+str(Xmod2[\'incorrect_all\'][i])+" "+str(Xmod2[\'avg_acc_r\'][i])+" "+str(Xmod2[\'avg_inacc_r\'][i])+" "+str(Xmod2[\'acc_r_single\'][i])+" "+str(Xmod2[\'inacc_r_single\'][i])+" "+str(Xmod2[\'score\'][i])+" "+str(Xmod2[\'score_c\'][i])+" "+str(Xmod2[\'average_score\'][i])+" "+str(Xmod2[\'acc_r\'][i])+" "+str(Xmod2[\'inacc_r\'][i])+" "+str(Xmod2[\'misses\'][i])+" "+str(Xmod2[\'missc\'])+" "+str(Xmod2[\'missm\'][i])+" "+str(Xmod2[\'misst\'][i])+" "+str(Xmod2[\'world\'][i])')


# In[79]:


label = []
for i in range(0,len(Ymod1)):
    if Ymod1[i] == 3:
        label.append([0, 0,0,1])  # class 3
    elif Ymod1[i] == 2:
        label.append([0, 0,1,0])  # class 2
    elif Ymod1[i] == 1:
        label.append([0, 1,0,0])  # class 1
    elif Ymod1[i] == 0:
        label.append([1, 0,0,0])  # class 0
label


# In[80]:


len(label)


# In[81]:


Xmod1test = get2A_test.drop(columns = ["accuracy_group","correct","incorrect","accuracy"])
Ymod1test = get2A_test["accuracy_group"]


# In[ ]:





# In[82]:


labelt = []
for i in range(0,len(Ymod1test)):
    if Ymod1test[i] == 3:
        labelt.append([0, 0,0,1])  # class 3
    elif Ymod1test[i] == 2:
        labelt.append([0, 0,1,0])  # class 2
    elif Ymod1test[i] == 1:
        labelt.append([0, 1,0,0])  # class 1
    elif Ymod1test[i] == 0:
        labelt.append([1, 0,0,0])  # class 0
labelt


# In[83]:


len(labelt)


# In[84]:


Xmod1sub = get2A_sub


# In[85]:


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


# In[86]:


get2A1 = get2A.drop(columns = ["accuracy", "correct","incorrect","accuracy_group"])
get2A1["accuracy_group"] = get2A["accuracy_group"]
get2A1


# In[87]:


get2A1.to_csv("daat.csv",header=False,index=None) #df.to_csv('file.csv', header=False, index=False)


# In[88]:



daat = numpy.loadtxt("daat.csv", delimiter=",")
daat


# In[89]:


Xk = daat[:,0:18]
Yk = daat[:,18]
Xk


# In[90]:


Yk


# In[91]:


Xmod1.to_csv("Xmod1.csv",header=False,index=None) #df.to_csv('file.csv', header=False, index=False)


# In[92]:



dataf = numpy.loadtxt("dataf.csv", delimiter=",")
dataf


# In[93]:



dataset = numpy.loadtxt("Xmod1.csv", delimiter=",")
dataset


# In[94]:


Xmod1


# In[95]:


# create model
model = Sequential()
model.add(Dense(12, input_dim=18, init= 'uniform' , activation= 'relu' ))
#model.add(Dense(8, init= 'uniform' , activation= 'relu' ))
model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' ))


# In[96]:


# Compile model
model.compile(loss= 'mean_squared_error' , optimizer= 'adam' , metrics=[ 'accuracy' ])


# In[97]:


# Fit the model
model.fit(dataset, Ymod1, nb_epoch=100,validation_split=0.2, batch_size=64)


# In[98]:


get_ipython().run_cell_magic('time', '', "import numpy as np\nimport catboost\nfrom catboost import CatBoostClassifier, Pool\n\n# train model on all data once\npoolmod2 = Pool(Xmod1, Ymod1,cat_features=['title','world'], feature_names=list(Xmod1.columns))\nclfmode2 = make_classifier1()\nclfmode2.fit(poolmod2, plot=True)")


# In[99]:


predictedmod2 = clfmode2.predict(Xmod1test)
from sklearn.metrics import classification_report
reportcatmod2 = classification_report(Ymod1test, predictedmod2)
print(reportcatmod2)


# In[100]:


predictedmodtr = clfmode2.predict(Xmod1)
from sklearn.metrics import classification_report
reportcatmodtr = classification_report(Ymod1, predictedmodtr)
print(reportcatmodtr)


# In[101]:


clfmode2.get_feature_importance()


# In[102]:


Xmod1test.columns


# In[103]:


# oof is an zeroed array of the same size of the input dataset
print('OOF QWK:', qwk(Ymod1test, predictedmod2))


# In[104]:


# oof is an zeroed array of the same size of the input dataset
print('OOF QWK:', qwk(Ymod1, predictedmodtr))


# In[105]:


X


# In[106]:


all_features = Xmod1.columns
all_features


# In[107]:


import numpy as np
import pandas as pd
import datetime
from catboost import CatBoostClassifier
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats


# In[108]:


get_ipython().run_cell_magic('time', '', '# CV\nX = Xmod1\ny = Ymod1\ncat_features = ["title"]\nfrom sklearn.model_selection import KFold\n# oof is an zeroed array of the same size of the input dataset\noof = np.zeros(len(X))\nNFOLDS = 5\n# here the KFold class is used to split the dataset in 5 diferents training and validation sets\n# this technique is used to assure that the model isn\'t overfitting and can performs aswell in \n# unseen data. More the number of splits/folds, less the test will be impacted by randomness\nfolds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)\ntraining_start_time = time()\nmodels = []\nfor fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):\n    # each iteration of folds.split returns an array of indexes of the new training data and validation data\n    start_time = time()\n    print(f\'Training on fold {fold+1}\')\n    # creates the model\n    clfm = make_classifier1()\n    # fits the model using .loc at the full dataset to select the splits indexes and features used\n    clfm.fit(X.loc[trn_idx, all_features], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),\n                          use_best_model=True, verbose=500, cat_features=cat_features)\n    \n    # then, the predictions of each split is inserted into the oof array\n    oof[test_idx] = clfm.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))\n    models.append(clfm)\n    oof1 = np.zeros(len(X.loc[trn_idx, all_features]))\n    oof1 = clfm.predict(X.loc[trn_idx, all_features])\n    print(\'OOF validation QWK:\', qwk(y.loc[test_idx], oof[test_idx]))\n    print(\'OOF training QWK:\', qwk(y.loc[trn_idx], oof1))\n    print(\'Fold {} finished in {}\'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))\n    print(\'____________________________________________________________________________________________\\n\')\n    #break\n    \nprint(\'-\' * 30)\n# and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)\nprint(\'OOF QWK:\', qwk(y, oof))\nprint(\'-\' * 30)')


# In[109]:


from sklearn.metrics import classification_report
reportcatmodo = classification_report(y, oof)
print(reportcatmodo)


# In[110]:


# make predictions on test set once
predictions = []
for model in models:
    predictions.append(model.predict(Xmod1test))
predictions = np.concatenate(predictions, axis=1)
print(predictions.shape)
predictions = stats.mode(predictions, axis=1)[0].reshape(-1)
print(predictions.shape)
#del X_test


# In[111]:


predictions


# In[112]:


# oof is an zeroed array of the same size of the input dataset
print('OOF QWK:', qwk(Ymod1test, predictions))


# In[113]:


from sklearn.metrics import classification_report
reportcatmodo1 = classification_report(Ymod1test, predictions)
print(reportcatmodo1)


# In[114]:


oof


# In[115]:


len(oof)


# In[116]:


train_labels_exp = train_labels


# In[117]:


train_labels_exp


# In[118]:


for i in range(0, len(train_labels_exp)):
    train_labels_exp["accuracy_group"][i] = oof[i]
train_labels_exp.groupby(["accuracy_group"]).size()


# In[119]:


train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')


# In[120]:


train_labels.groupby(["accuracy_group"]).size()


# In[121]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()

# Embedding layer
model.add(
    Embedding(input_dim=Xmod1test,
              input_length = training_length,
              output_dim=len,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_words, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[122]:


predictedmod2120 = clfmode2.predict(Xmod1sub)


# In[123]:


sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


# In[124]:


sample_submission2120 = sample_submission


# In[125]:


for i in range(0,len(sample_submission2120)):
    sample_submission2120["accuracy_group"][i] = int(predictedmod2120[i])


# In[126]:


sample_submission2120.to_csv("submission.csv", index=None)


# In[127]:


sample_submission2120.groupby("accuracy_group").size()


# In[128]:


import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import scipy as sp
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm


# In[129]:


tqdm.pandas()


# In[ ]:





# In[ ]:





# In[130]:


def cv_train(X, y, cv, **kwargs):
    """
    Author: https://www.kaggle.com/xhlulu/
    Source: https://www.kaggle.com/xhlulu/ds-bowl-2019-simple-lgbm-using-aggregated-data
    """
    models = []
    
    kf = KFold(n_splits=cv, random_state=2019)
    
    for train, test in kf.split(X):
        x_train, x_val, y_train, y_val = X[train], X[test], y[train], y[test]
        
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)
        
        model = lgb.train(train_set=train_set, valid_sets=[train_set, val_set], **kwargs)
        models.append(model)
        
        if kwargs.get("verbose_eval"):
            print("\n" + "="*50 + "\n")
    
    return models


# In[131]:


def cv_predict(models, X):
    """
    Author: https://www.kaggle.com/xhlulu/
    Source: https://www.kaggle.com/xhlulu/ds-bowl-2019-simple-lgbm-using-aggregated-data
    """
    return np.mean([model.predict(X) for model in models], axis=0)


# In[132]:


Xmod1


# In[133]:


Ymod1


# In[134]:


get_ipython().run_cell_magic('time', '', "X = Xmod1.values\ny = Ymod1.values\nparams = {\n    'learning_rate': 0.01,\n    'bagging_fraction': 0.95,\n    'feature_fraction': 0.2,\n    'max_height': 10,\n    'lambda_l1': 10,\n    'lambda_l2': 10,\n    'metric': 'multiclass',\n    'objective': 'multiclass',\n    'num_classes': 4,\n    'random_state': 2019\n}\nmodels11 = cv_train(X, y, cv=10, params=params, num_boost_round=5000,\n                  early_stopping_rounds=100, verbose_eval=250)")


# In[135]:


X_test11 = Xmod1test.values
test_pred11 = cv_predict(models=models11, X=X_test11).argmax(axis=1)


# In[136]:


from sklearn.metrics import classification_report
reportcatmod2f11 = classification_report(Ymod1test,test_pred11)
print(reportcatmod2f11)


# In[137]:


# oof is an zeroed array of the same size of the input dataset
print('OOF QWK:', qwk(Ymod1test, test_pred11))


# In[138]:


test_pred1b1 = cv_predict(models=models11, X=X).argmax(axis=1)


# In[139]:


from sklearn.metrics import classification_report
reportcatmod2f1b1 = classification_report(y,test_pred1b1)
print(reportcatmod2f1b1)


# In[140]:


# oof is an zeroed array of the same size of the input dataset
print('OOF QWK:', qwk(y, test_pred1b1))


# In[141]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(Xmod1)
x_test = sc.transform(Xmod1test)


# In[142]:


import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=Ymod1)
params = {
    'learning_rate': 0.01,
    'bagging_fraction': 0.95,
    'feature_fraction': 0.2,
    'max_height': 3,
    'lambda_l1': 10,
    'lambda_l2': 10,
    'metric': 'multiclass',
    'objective': 'multiclass',
    'num_classes': 4,
    'random_state': 2019
}
#params = {}
#params['learning_rate'] = 0.003
#params['boosting_type'] = 'gbdt'
#params['objective'] = 'binary'
#params['metric'] = 'binary_logloss'
#params['sub_feature'] = 0.5
#params['num_leaves'] = 10
#params['min_data'] = 50
#params['max_depth'] = 10
clflgb = lgb.train(params, d_train, 100)


# In[143]:


#Prediction
y_pred1=clflgb.predict(x_test)
y_pred1


# In[144]:


len(Ymod1test)


# In[145]:


for i in range(0,len(Ymod1test)):
    if (y_pred1[i][0]>=.25):
        Y_pred1.append(0)
    elif (y_pred1[i][1]>=.25):
        Y_pred1.append(1)
    elif (y_pred1[i][2]>=.25):
        Y_pred1.append(2)
    elif (y_pred1[i][3]>=.25):
        Y_pred1.append(3)


# In[146]:


Y_pred1


# In[147]:


from sklearn.metrics import classification_report
reportcatmod2l = classification_report(Ymod1test, Y_pred1)
print(reportcatmod2l)


# In[ ]:




