#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# for text processing
import re 
import string

# for numeric processing
from sklearn.preprocessing import MinMaxScaler

# Setting the seed for reproducibility
np.random.seed(42)


# In[ ]:


# Installing the AutoML library
# import sys
# !{sys.executable} -m pip install tpot
# Note - there are optional extras needed to use other elements of TPOT - see the docs: http://epistasislab.github.io/tpot/installing/


# In[ ]:


from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[ ]:


train_raw = pd.read_csv("../input/TrainData.csv",low_memory=False)
test_raw = pd.read_csv("../input/TestData.csv",low_memory=False)


# In[ ]:


def prep_pipeline(data_prep):
    
    # Set Index
    data_prep = data_prep.set_index("id")
    
    # Specify features to exclude
    cols_exclude_total = ['abn_hashed','address_line_1','address_line_2','address_type','ais_due_date',        'ais_due_date_processed','brc','country','conducted_activities',        'description_of_purposes_change__if_applicable_','fin_report_from',        'operating_countries','other_activity_description','other_beneficiaries_description',        'postcode','registration_status','type_of_financial_statement','fin_report_to','accrual_accounting',        'charity_activities_and_outcomes_helped_achieve_charity_purpose']
    
    # Sort Columns
    def feature_sort(cols_num,cols_bool,cols_date,cols_cat,cols_other,cols_exclude_total):
        for col in data_prep.columns:
            if col not in cols_exclude_total + cols_num + cols_bool + cols_date + cols_cat + cols_other:
                if col in data_prep.columns[(data_prep.dtypes == np.float64) | (data_prep.dtypes == np.float32)]:
                    cols_num.append(col)
                elif (data_prep[col].nunique() == 2) or ("true" in data_prep[col].unique()) or ("false" in data_prep[col].unique())                 or ("yes" in data_prep[col].unique()) or ("no" in data_prep[col].unique()):
                    cols_bool.append(col)
                elif 'date' in str(col):
                    cols_date.append(col)
                elif data_prep[col].nunique() < data_prep.shape[0]/100: # Arbitrary limit
                    cols_cat.append(col)
                else:
                    cols_other.append(col)
        return cols_num,cols_bool,cols_date,cols_cat,cols_other
    
    cols_num = []
    cols_bool = ['purpose_change_in_next_fy']
    cols_date = []
    cols_cat = []
    cols_other = ['staff_volunteers','staff_full_time','town_city']
    
    cols_num,cols_bool,cols_date,cols_cat,cols_other = feature_sort(cols_num,cols_bool,cols_date,cols_cat,cols_other,cols_exclude_total)
    
    # Drop exclusion columns
    data_prep = data_prep.drop(labels=cols_exclude_total,axis = 1)
    
    # Clean up other feature columns
    staff_volunteer_cleanupdict = {'1TO10':5,'11TO50':30,'51TO100':75,'101TO500':350,'0TO50':25,'OVER1000':1250,'501TO1000':750,'NONE':0,'None':0,'1-10':5,'11to50':30,'0-30':15,                '11-50':30,'0 to 9.': 5}
    data_prep['staff_volunteers'] = data_prep['staff_volunteers'].apply((lambda x: staff_volunteer_cleanupdict.get(x,x)))
    data_prep['staff_volunteers'] = pd.to_numeric(data_prep['staff_volunteers'],errors='coerce')
    data_prep['staff_full_time'] = pd.to_numeric(data_prep['staff_full_time'],errors='coerce')
    cols_num+=['staff_volunteers','staff_full_time']
    
    punct_reg = re.compile('[%s+]' % re.escape(string.whitespace + string.punctuation))
    def text_proc(text):
        proc = str(text)
        proc = punct_reg.sub('_', proc)
        return proc
    
    data_prep['town_city'] = data_prep['town_city'].str.lower().apply(lambda x: text_proc(x))
    capital_dict = {'melbourne':1,'sydney':1,'adelaide':1,'brisbane':1,'hobart':1,'perth':1,'canberra':1,'darwin':1}
    data_prep['located_in_capital_city'] = data_prep['town_city'].apply(lambda x: capital_dict.get(x,0))
    data_prep.drop(columns='town_city',inplace=True)
    
    # Fill Nulls
    object_nulls_cols = data_prep.columns[(data_prep.isna().any()) & (data_prep.dtypes=='O')].tolist()
    
    # For object cols, replace nulls with most common value
    for c in object_nulls_cols:
        data_prep[c] = data_prep[c].fillna(data_prep[c].mode().iloc[0])
    
    # For staff numeric cols, replace nans with 0
    staff_cols_num = [col for col in cols_num if 'staff_' in col]
    for c in staff_cols_num:
        data_prep[c] = data_prep[c].fillna(0)
    # For other numeric cols, replace nans with median
    other_cols_num = [col for col in cols_num if 'staff_' not in col]
    for c in other_cols_num:
        data_prep[c] = data_prep[c].fillna(data_prep[c].median())
    
    # Numeric cleaning - clip and log transformation
    for c in cols_num:
        data_prep[c].clip(lower=0,inplace=True)
        data_prep[c] = data_prep[c].apply(lambda x: np.log(x+1))
    
    # Numeric cleaning - minmax scaling (on all except target)
    cols_num_trans = cols_num.copy()
    try:
        cols_num_trans.remove('donations_and_bequests')
    except ValueError:
        pass  # do nothing!
    
    scaler = MinMaxScaler()
    data_prep[cols_num_trans] = scaler.fit_transform(data_prep[cols_num_trans])
    
    # Boolean cleaning
    for col in cols_bool:
        data_prep[col] = data_prep[col].replace({ 'nan':0, 'n':0, 'N':0, 'y':1, 'Y':1, 'false':0, 'true':1})
        data_prep[col] = data_prep[col].astype(np.float32)
        
    # Category feature cleaning
    def text_proc_2(text):
        proc = str(text)
        proc = proc.lower() #changes case to lower
        proc = proc.strip() #removes leading and trailing spaces/tabs/new lines
        proc = punct_reg.sub('_', proc)
        return proc
    
    for col in cols_cat:
        data_prep[col] = data_prep[col].apply(lambda x: text_proc_2(x))
        
    # Ordinal encoding    
    scale_mapper = {"charity_size":
                    {'small':1,
                    'medium':2,
                    'large':3,},
                "sample_year":
                    {'fy2016':3,
                    'fy2015':2,
                    'fy2014':1},
                "state":
                    {'victoria':'vic'}
               }
    data_prep.replace(to_replace=scale_mapper,inplace=True)
    
    # Reducing cardinality for nominal features
    def reduce_cardinality(feats_cols,data):
        feats_distros = dict()
        for c in feats_cols:
            df = data[c]
            df = df.value_counts()
            df.fillna(0, inplace=True)
            df = df.astype(np.int32)
            df.sort_values(ascending = False, inplace = True)
            df = pd.DataFrame(df)
            df.columns = [c + ' count']
            df[c + ' distribution'] = 100*df/df.sum()
            feats_distros.update({c:df})

        for feat in feats_cols:
            feat_distro = feats_distros[feat][feat + ' distribution']
            feat_index = feat_distro[feat_distro < 1].index.tolist()
            lean_feat_index = len(feat_index)
            if lean_feat_index > 0:
                feat_sub = lean_feat_index*[np.nan]
                feat_dict = dict(zip(feat_index, feat_sub))
                data[feat] = data[feat].replace(feat_dict)
            data[feat].fillna('other',inplace=True)

        return data
    
    data_prep = reduce_cardinality(['main_activity'],data_prep)
    
    # Getting dummies for Nominal features
    data_prep = pd.get_dummies(data_prep, prefix = None, prefix_sep = '-', dummy_na = False, columns = ['state','main_activity'])
    
    return data_prep


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_prep = prep_pipeline(train_raw)\ntest_prep = prep_pipeline(test_raw)')


# In[ ]:


# Relabelling target variable in training data so TPOT works over data
train_prep.rename(columns={'donations_and_bequests':'target'},inplace=True)


# In[ ]:


test_prep['main_activity-animal_protection'] = 0
set(train_prep.columns).difference(set(test_prep.columns)) # confirming that only the label differs between train and test


# In[ ]:


train_prep.corr()['donations_and_bequests'].sort_values(ascending=False).head(20)


# In[ ]:


train_prep.corr()['donations_and_bequests'].sort_values(ascending=False).tail(20)


# In[ ]:


feat_subset = [
    'previous__donations_and_bequests',
    'previous__total_gross_income',
    'previous__total_assets',
    'staff_volunteers',
    'previous__net_surplus_deficit',
    'charity_size',
    'operates_overseas',
    'social_services',
    'staff_part_time',
    'emergency_relief',
    'other_health_service_delivery',
    'other_education',
    'international_activities',
    'staff_casual',
    'adults_25_to_under_65',
    'operates_in_act',
    'staff_full_time',
    'females',
    'overseas_communities_or_charities',
    'located_in_capital_city',
    'main_activity-religious_activities'
]
len(feat_subset)


# In[ ]:


# Split Training Data into Test Train
# X_train, X_test, y_train, y_test = train_test_split(train_prep[feat_subset].sample(100),\
#                                                     train_prep['target'],\
#                                                     train_size=0.75,\
#                                                     test_size=0.25)


# In[ ]:


# Instantiating the estimator
# tpot = TPOTRegressor(generations=5, population_size=30, verbosity=2)


# In[ ]:


# When you invoke fit method, TPOT will create generations of populations, 
# seeking best set of parameters. Arguments you have used to create
# TPOTClassifier such as generations and population_size will affect the
# search space and resulting pipeline.
# tpot.fit(X_train, y_train)


# In[ ]:


# tpot.export('charity_feat_subset_tpot_pipeline.py')


# In[ ]:


# !cat charity_feat_subset_tpot_pipeline.py


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = train_prep[feat_subset+['target']]
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target =             train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-13.793020398101522
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    RandomForestRegressor(bootstrap=True, max_features=0.15000000000000002, min_samples_leaf=8, min_samples_split=10, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


exported_pipeline.score(testing_features,testing_target)
# That accuracy is fairly poor.


# In[ ]:


predicted_target = exported_pipeline.predict(testing_features)


# In[ ]:


# RMSE?
np.sqrt(mean_squared_error(testing_target,predicted_target))
# Pretty rough, but still might generalize.


# In[ ]:


# What are the features weighted in the model?
names = train_prep[feat_subset].columns.tolist()
list(sorted(zip(map(lambda x: round(x, 4), exported_pipeline.steps[-1][1].feature_importances_), names),reverse=True))


# In[ ]:


y_submission = exported_pipeline.predict(test_prep[feat_subset])
submission = pd.DataFrame({'id': test_raw['id'], 'log__donations_and_bequests': y_submission})
submission.to_csv('tpot_test_submission.csv', index = False)


# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')\n// Table of contents")

