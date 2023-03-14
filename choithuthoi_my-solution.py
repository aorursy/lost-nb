#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from operator import itemgetter


# In[2]:


def get_station_ohe():
    directory = '../input/'
    trainfile = 'train_date.csv'
    testfile = 'test_date.csv'
    
    features = None
    subset = None
    train_date_part = pd.read_csv(directory + trainfile, nrows=10000)
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
    stations = list([f.split('_')[1] for f in date_cols ])
    stations = sorted(stations,key= lambda x: int(x[1:]))
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          usecols=['Id'] + date_cols,
                                          chunksize=50000,
                                          low_memory=False)):
        
        if features is None:
            features = list(chunk.columns)
            features.remove('Id')
        
        chunk.columns = ['Id'] + stations
        chunk['start_station'] = -1
        chunk['end_station'] = -1
        for s in stations:
            chunk[s] = 1 * (chunk[s] >= 0)
            id_not_null = chunk[chunk[s] == 1].Id
            chunk.loc[(chunk['start_station']== -1) & (chunk.Id.isin(id_not_null)),'start_station'] = int(s[1:])
            chunk.loc[chunk.Id.isin(id_not_null),'end_station'] = int(s[1:])   
        subset = pd.concat([subset, chunk])
        del chunk
        gc.collect()

    for i, chunk in enumerate(pd.read_csv(directory + testfile,
                                          usecols=['Id'] + date_cols,
                                          chunksize=50000,
                                          low_memory=False)):
        #print(i)
        
        chunk.columns = ['Id'] + stations
        chunk['start_station'] = -1
        chunk['end_station'] = -1
        for s in stations:
            chunk[s] = 1 * (chunk[s] >= 0)
            id_not_null = chunk[chunk[s] == 1].Id
            chunk.loc[(chunk['start_station']== -1) & (chunk.Id.isin(id_not_null)),'start_station'] = int(s[1:])
            chunk.loc[chunk.Id.isin(id_not_null),'end_station'] = int(s[1:])   
        subset = pd.concat([subset, chunk])
        del chunk
        gc.collect()      
        
    return subset,stations,date_cols
    


# In[3]:


station_ohe,stations,date_cols = get_station_ohe()


# In[4]:


station_ohe['path_len'] = station_ohe[stations].sum(axis=1)


# In[5]:


station_ohe.head(20)


# In[6]:


station_ohe.shape


# In[7]:


def get_date_features():
    directory = '../input/'
    trainfile = 'train_date.csv'
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          chunksize=1,
                                          low_memory=False)):
        features = list(chunk.columns)
        del chunk
        break

    seen = np.zeros(52)
    rv = []
    for f in features:
        if f == 'Id' or 'S24' in f or 'S25' in f:
            rv.append(f)
            continue
            
        station = int(f.split('_')[1][1:])
        
        if seen[station]:
            continue
        
        seen[station] = 1
        rv.append(f)
        
    return rv
        
usefuldatefeatures = get_date_features()


# In[8]:


def create_new_feats():
    directory = '../input/'
    trainfile = 'train_date.csv'
    testfile = 'test_date.csv'
    
    features = None
    subset = None
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          usecols=usefuldatefeatures,
                                          chunksize=50000,
                                          low_memory=False)):
        #print(i)
       
        if features is None:
            features = list(chunk.columns)
            features.remove('Id')
        week_duration = 1679
        df_mindate_chunk = chunk[['Id']].copy()
        df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
        df_mindate_chunk['maxdate'] = chunk[features].max(axis=1).values
        df_mindate_chunk['duration'] =  df_mindate_chunk['maxdate'] - df_mindate_chunk['mindate']
        df_mindate_chunk['part_week'] = ((df_mindate_chunk['mindate'].values * 100)  % week_duration).astype(np.int64)
        df_mindate_chunk['min_time_station'] =  chunk[features].idxmin(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
        df_mindate_chunk['max_time_station'] =  chunk[features].idxmax(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)

        
        if subset is None:
            subset = df_mindate_chunk.copy()
        else:
            subset = pd.concat([subset, df_mindate_chunk])
            
        del chunk
        gc.collect()

    for i, chunk in enumerate(pd.read_csv(directory + testfile,
                                          usecols=usefuldatefeatures,
                                          chunksize=50000,
                                          low_memory=False)):
        #print(i)
        df_mindate_chunk = chunk[['Id']].copy()
        df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
        df_mindate_chunk['maxdate'] = chunk[features].max(axis=1).values
        df_mindate_chunk['duration'] =  df_mindate_chunk['maxdate'] - df_mindate_chunk['mindate']
        df_mindate_chunk['part_week'] = ((df_mindate_chunk['mindate'].values * 100)  % week_duration).astype(np.int64)
        df_mindate_chunk['min_time_station'] =  chunk[features].idxmin(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
        df_mindate_chunk['max_time_station'] =  chunk[features].idxmax(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
        
        subset = pd.concat([subset, df_mindate_chunk])
        
        del chunk
        gc.collect()    
        
        
    return subset


# In[9]:


new_features = create_new_feats()


# In[10]:


new_features.head()


# In[11]:


new_features.sort_values(by=['mindate', 'Id'], inplace=True)


# In[12]:


new_features['mindate_id_diff'] = new_features.Id.diff()


# In[13]:


midr = np.full_like(new_features.mindate_id_diff.values, np.nan)


# In[14]:


midr[0:-1] = -new_features.mindate_id_diff.values[1:]


# In[15]:


new_features['mindate_id_diff_reverse'] = midr


# In[16]:


def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


# In[17]:


def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


# In[18]:


def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc


# In[19]:


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


# In[20]:


directory = '../input/'
trainfiles = ['train_date.csv',
                  'train_numeric.csv']
testfiles = ['test_date.csv',
                 'test_numeric.csv']


# In[21]:


#feature generate from Xgboost with 200,000 records
num_feats = ['Id',
 'L3_S33_F3855',
 'L3_S32_F3850',
 'L3_S33_F3865',
 'L1_S24_F1581',
 'L3_S38_F3952',
 'L1_S24_F1672',
 'L1_S24_F1632',
 'L1_S24_F1846',
 'L1_S24_F1844',
 'L1_S24_F1609',
 'L1_S24_F1667',
 'L1_S24_F1842',
 'L0_S13_F356',
 'L3_S29_F3342',
 'L3_S29_F3407',
 'L3_S34_F3876',
 'L0_S11_F302',
 'L3_S29_F3461',
 'L3_S30_F3494',
 'L0_S3_F100',
 'L0_S1_F28',
 'L0_S6_F122',
 'L0_S0_F0',
 'L0_S0_F20',
 'L3_S30_F3704',
 'Response']


# In[22]:


len(date_cols)


# In[23]:


cols = [['Id']+date_cols,num_feats]


# In[24]:


cols


# In[25]:


traindata = None
testdata = None


# In[26]:


for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
            #print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if traindata is None:
            traindata = subset.copy()
        else:
            traindata = pd.merge(traindata, subset.copy(), on="Id")
        del subset
        gc.collect()


# In[27]:


del cols[1][-1]


# In[28]:


for i, f in enumerate(testfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
            #print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if testdata is None:
            testdata = subset.copy()
        else:
            testdata = pd.merge(testdata, subset.copy(), on="Id")
        del subset
        gc.collect()


# In[29]:


del midr
gc.collect()


# In[30]:


traindata = traindata.merge(new_features, on='Id')
traindata = traindata.merge(station_ohe, on='Id')
testdata = testdata.merge(new_features, on='Id')
testdata = testdata.merge(station_ohe, on='Id')


# In[31]:


del new_features
del station_ohe
gc.collect()


# In[32]:


testdata['Response'] = 0 


# In[33]:


visibletraindata = traindata[::2]


# In[34]:


blindtraindata = traindata[1::2]


# In[35]:


del traindata
gc.collect()


# In[36]:


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName)['Response'].mean().reset_index()
    grpCount = data1.groupby(columnName)['Response'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.Response
    if(useLOO):
        grpOutcomes = grpOutcomes[grpOutcomes.cnt > 1]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['Response'].values
    x = pd.merge(data2[[columnName, 'Response']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['Response']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
        #  x = x + np.random.normal(0, .01, x.shape[0])
    return x.fillna(x.mean())


# In[37]:


for i in range(2):
        for col in cols[i][1:]:
            blindtraindata.loc[:, col] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, False).values
            testdata.loc[:, col] = LeaveOneOut(visibletraindata,
                                               testdata, col, False).values


# In[38]:


num_rounds =52
params = {}
params['objective'] = "binary:logistic"
params['eta'] = 0.02105
params['max_depth'] = 28
params['colsample_bytree'] = 0.999
params['subsample'] = 0.999999
params['min_child_weight'] = 3
params['base_score'] = 0.0044
params['silent'] = True
params['eval_metric']='auc'
print('Fitting')


# In[39]:


trainpredictions = None
testpredictions = None
features = list(blindtraindata.columns)
features.remove('Response')
features.remove('Id')
dvisibletrain =         xgb.DMatrix(blindtraindata[features],
                    blindtraindata.Response,
                    silent=True)
dtest =         xgb.DMatrix(testdata[features],
                    silent=True)

folds = 1


# In[40]:


for i in range(folds):
        print('Fold:', i)
        params['seed'] = i
        watchlist = [(dvisibletrain, 'train'), (dvisibletrain, 'val')]
        clf = xgb.train(params, dvisibletrain,
                        num_boost_round=num_rounds,
                        evals=watchlist,
                        early_stopping_rounds=20,
                        feval=mcc_eval,
                        maximize=True
                        
                        )
        limit = clf.best_iteration+1
        predictions =             clf.predict(dvisibletrain, ntree_limit=limit)


# In[41]:


best_proba, best_mcc, y_pred = eval_mcc(dvisibletrain.Response,
                                                predictions,
                                                True)
print(best_proba)
print(best_mcc)


# In[42]:


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


# In[43]:


if(trainpredictions is None):
            trainpredictions = predictions
else:
            trainpredictions += predictions
predictions = clf.predict(dtest, ntree_limit=limit)
if(testpredictions is None):
            testpredictions = predictions
else:
            testpredictions += predictions
imp = get_importance(clf, features)


# In[44]:


y_pred = (testpredictions/folds > 0.4).astype(int)
submission = pd.DataFrame({"Id": testdata.Id.values,
                               "Response": y_pred})
submission[['Id', 'Response']].to_csv('xgbsubmission'+str(folds)+'.csv',
                                          index=False)


# In[45]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax,importance_type='gain')
plt.show()


# In[ ]:





# In[ ]:




