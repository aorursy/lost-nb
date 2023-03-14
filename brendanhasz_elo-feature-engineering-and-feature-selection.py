#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor

# Plot settings
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
sns.set()

get_ipython().system('pip install git+http://github.com/brendanhasz/dsutils.git')
from dsutils.encoding import one_hot_encode
from dsutils.encoding import TargetEncoderCV
from dsutils.printing import print_table
from dsutils.evaluation import permutation_importance_cv
from dsutils.evaluation import plot_permutation_importance


# In[2]:


# Cards data

# Datatypes for each column
# no nulls in test/train except for ONE ROW in test! (first_active_month)
dtypes = {
  'card_id':            'str',     # 201917 unique vals
  'target':             'float32', # -33.22 thru ~18
  'first_active_month': 'str',     # 2011-10 thru 2018-02
  'feature_1':          'uint8',   # 1 thru 5
  'feature_2':          'uint8',   # 1 thru 3
  'feature_3':          'uint8',   # 0 and 1
}
train = pd.read_csv('../input/train.csv',
                    usecols=dtypes.keys(),
                    dtype=dtypes)
del dtypes['target']
test = pd.read_csv('../input/test.csv',
                   usecols=dtypes.keys(),
                   dtype=dtypes)

# Add target col to test
test['target'] = np.nan

# Merge test and train
cards = pd.concat([train, test])

del train, test
gc.collect()

# Convert first_active_month to datetime
cards['first_active_month'] = pd.to_datetime(cards['first_active_month'],
                                             format='%Y-%m')

# Make card_id the index
cards.set_index('card_id', inplace=True)
gc.collect()
cards.sample(10)


# --------------------------------------------------

# Merchants data

# Datatypes of each column
# (don't load cols which are in transactions data, just use those vals)
# Nulls: NO nulls except for 13 rows in avg_sales_lag{3,6,12}
dtypes = {
  'merchant_id':                 'str',     # 334633 unique values
  'merchant_group_id':           'uint32',  # 1 thru 112586 (w/ some missing, ~109k uniques)
  'numerical_1':                 'float32', # ~ -0.06 thru ~ 183.8 (only 951 unique vals?)
  'numerical_2':                 'float32', # roughly the same as above
  'most_recent_sales_range':     'str',     # A, B, C, D, or E
  'most_recent_purchases_range': 'str',     # A, B, C, D, or E
  'avg_sales_lag3':              'float32', # most between 0 and 2, if you transform by 1/x, all but 3 are between 0 and 4
  'avg_purchases_lag3':          'float32', # most between 0 and 2, if you transform by 1/x, all but 3 are between 0 and 4
  'active_months_lag3':          'uint8',   # 1 to 3 
  'avg_sales_lag6':              'float32', # similar to avg_sales_lag3
  'avg_purchases_lag6':          'float32', # similar to avg_purchases_lag3
  'active_months_lag6':          'uint8',   # 1 to 6
  'avg_sales_lag12':             'float32', # similar to avg_sales_lag3
  'avg_purchases_lag12':         'float32', # similar to avg_purchases_lag3
  'active_months_lag12':         'uint8',   # 1 to 12
  'category_4':                  'str',     # Y or N
}

# Load the data
merchants = pd.read_csv('../input/merchants.csv',
                        usecols=dtypes.keys(),
                        dtype=dtypes)

# Map merchant_id to integer
merch_id_map = dict(zip(
    merchants['merchant_id'].values,
    merchants['merchant_id'].astype('category').cat.codes.values
))

def preprocess_merch_data(df):
    
    # Convert merchant ID to numbers
    df['merchant_id'] = df['merchant_id'].map(merch_id_map).astype('float32')

    # Inverse transforms
    inversions = [
        'avg_sales_lag3',
        'avg_sales_lag6',
        'avg_sales_lag12',
        'avg_purchases_lag3',
        'avg_purchases_lag6',
        'avg_purchases_lag12',
    ]
    for col in inversions:
        df[col] = 1.0/df[col]

    # Encode categorical columns
    bool_map = {'Y': 1, 'N': 0}
    five_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    conversions = [
        ('category_4', bool_map, 'uint8'),
        ('most_recent_sales_range', five_map, 'uint8'),
        ('most_recent_purchases_range', five_map, 'uint8')
    ]
    for col, mapper, new_type in conversions:
        df[col] = df[col].map(mapper).astype(new_type)
        
    # Clean up
    gc.collect()

# Preprocess the merchants data
preprocess_merch_data(merchants)


# ------------------------------------------------------

# Transactions data

# Datatypes of each column
# only NaNs are in category_3, merchant_id, and category_2
dtypes = {
    'authorized_flag':      'str',     # Y or N
    'card_id':              'str',     # 325540 unique values
    'city_id':              'int16',   # -1 then 1 to 347 (is -1 supposed to be nan?)
    'category_1':           'str',     # Y or N
    'installments':         'int8',    # -25, then -1 thru 12 (-1 supposed to be nan?)
    'category_3':           'str',     # A, B, C, and nan (ordinal?)
    'merchant_category_id': 'int16',   # 2 to 891
    'merchant_id':          'str',     # 334633 unique values and nans (164697 nans!)
    'month_lag':            'int8',    # -13 thru 0
    'purchase_amount':      'float32', # min: -0.746, med: -0.699, max: 11269.667
    'purchase_date':        'str',     # YYYY-MM-DD hh:mm:ss
    'category_2':           'float32', # 1 thru 5 and nan (ordinal?)
    'state_id':             'int8',    # -1 then 1 thru 24
    'subsector_id':         'int8'     # 1 thru 41
}

# Load the data
hist_trans = pd.read_csv('../input/historical_transactions.csv', 
                         usecols=dtypes.keys(),
                         dtype=dtypes)
new_trans = pd.read_csv('../input/new_merchant_transactions.csv', 
                        usecols=dtypes.keys(),
                        dtype=dtypes)

def preprocess_trans_data(df):
    
    # Convert merchant_id to numbers
    df['merchant_id'] = df['merchant_id'].map(merch_id_map).astype('float32')

    # Convert purchase_date to datetime
    df['purchase_date'] = df['purchase_date'].str.slice(0, 19)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'],
                                         format='%Y-%m-%d %H:%M:%S')

    # Encode categorical columns
    bool_map = {'Y': 1, 'N': 0}
    three_map = {'A': 0, 'B': 1, 'C': 2}
    conversions = [
        ('authorized_flag', bool_map, 'uint8'),
        ('category_1', bool_map, 'uint8'),
        ('category_3', three_map, 'float32'), #has NaNs so have to use float
    ]
    for col, mapper, new_type in conversions:
        df[col] = df[col].map(mapper).astype(new_type)
        
    # Clean up
    gc.collect()

# Preprocess the transactions data
preprocess_trans_data(hist_trans)
preprocess_trans_data(new_trans)


# In[3]:


# Merge transactions with merchants data
hist_trans = pd.merge(hist_trans, merchants, on='merchant_id')
new_trans = pd.merge(new_trans, merchants, on='merchant_id')

# Clean up
del merchants
gc.collect()


# In[4]:


# One-hot encode category 2 and 3
cat_cols = ['category_2', 'category_3']
hist_trans = one_hot_encode(hist_trans,
                            cols=cat_cols)
new_trans = one_hot_encode(new_trans,
                           cols=cat_cols)
gc.collect()


# In[5]:


# Time-based features for purchases
ref_date = np.datetime64('2017-09-01')
one_hour = np.timedelta64(1, 'h')
for df in [hist_trans, new_trans]:
    tpd = df['purchase_date']
    df['purchase_hour'] = tpd.dt.hour.astype('uint8')
    df['purchase_day'] = tpd.dt.dayofweek.astype('uint8')
    df['purchase_week'] = tpd.dt.weekofyear.astype('uint8')
    df['purchase_dayofyear'] = tpd.dt.dayofyear.astype('uint16')
    df['purchase_month'] = tpd.dt.month.astype('uint8')
    df['purchase_weekend'] = (df['purchase_day'] >=5 ).astype('uint8')
    df['purchase_time'] = ((tpd - ref_date) / one_hour).astype('float32')
    df['ref_date'] = ((tpd - pd.to_timedelta(df['month_lag'], 'M')
                          - ref_date ) / one_hour).astype('float32')

    # Time sime first active
    tsfa = pd.merge(df[['card_id']], 
                    cards[['first_active_month']].copy().reset_index(),
                    on='card_id', how='left')
    df['time_since_first_active'] = ((tpd - tsfa['first_active_month'])
                                     / one_hour).astype('float32')
    
    # Clean up
    del tsfa
    del df['purchase_date']
    gc.collect()


# In[6]:


cards['first_active_month'] = (12*(cards['first_active_month'].dt.year-2011) + 
                               cards['first_active_month'].dt.month).astype('float32')


# In[7]:


# Group transactions by card id
hist_trans = hist_trans.groupby('card_id', sort=False)
new_trans = new_trans.groupby('card_id', sort=False)


# In[8]:


def entropy(series):
    """Categorical entropy"""
    probs = series.value_counts().values.astype('float32')
    probs = probs / np.sum(probs)
    probs[probs==0] = np.nan
    return -np.nansum(probs * np.log2(probs))


# In[9]:


def mean_diff(series):
    """Mean difference between consecutive items in a series"""
    ss = series.sort_values()
    return (ss - ss.shift()).mean()


# In[10]:


def period(series):
    """Period of a series (max-min)"""
    return series.max() - series.min()


# In[11]:


def mode(series):
    """Most common element in a series"""
    tmode = series.mode()
    if len(tmode) == 0:
        return np.nan
    else:
        return tmode[0]


# In[12]:


# Aggregations to perform for each predictor type
binary_aggs = ['sum', 'mean', 'nunique']
categorical_aggs = ['nunique', entropy, mode]
continuous_aggs = ['min', 'max', 'sum', 'mean', 'std', 'skew', mean_diff, period]


# In[13]:


# Aggregations to perform on each column
aggs = {
    'authorized_flag':             binary_aggs,
    'city_id':                     categorical_aggs,
    'category_1':                  binary_aggs,
    'installments':                continuous_aggs,
    'category_3_nan':              ['mean'],
    'category_3_0.0':              ['mean'],
    'category_3_1.0':              ['mean'],
    'category_3_2.0':              ['mean'],
    'category_2_nan':              ['mean'],
    'category_2_1.0':              ['mean'],
    'category_2_2.0':              ['mean'],
    'category_2_3.0':              ['mean'],
    'category_2_4.0':              ['mean'],
    'category_2_5.0':              ['mean'],
    'merchant_category_id':        categorical_aggs,
    'merchant_id':                 categorical_aggs,
    'month_lag':                   continuous_aggs,
    'purchase_amount':             continuous_aggs,
    'purchase_time':               continuous_aggs + ['count'],
    'purchase_hour':               categorical_aggs + ['mean'],
    'purchase_day':                categorical_aggs + ['mean'],
    'purchase_week':               categorical_aggs + continuous_aggs,
    'purchase_month':              categorical_aggs + continuous_aggs,
    'purchase_weekend':            binary_aggs,
    'ref_date':                    continuous_aggs,
    'time_since_first_active':     continuous_aggs,
    'state_id':                    categorical_aggs,
    'subsector_id':                categorical_aggs,
    'merchant_group_id':           categorical_aggs,
    'numerical_1':                 continuous_aggs,
    'numerical_2':                 continuous_aggs,
    'most_recent_sales_range':     categorical_aggs + ['mean'], #ordinal?
    'most_recent_purchases_range': categorical_aggs + ['mean'], #orindal?
    'avg_sales_lag3':              continuous_aggs,
    'avg_purchases_lag3':          continuous_aggs,
    'active_months_lag3':          continuous_aggs,
    'avg_sales_lag6':              continuous_aggs,
    'avg_purchases_lag6':          continuous_aggs,
    'active_months_lag6':          continuous_aggs,
    'avg_sales_lag12':             continuous_aggs,
    'avg_purchases_lag12':         continuous_aggs,
    'active_months_lag12':         continuous_aggs,
    'category_4':                  binary_aggs,
}


# In[14]:


get_ipython().run_cell_magic('time', '', "\n# Perform each aggregation\nfor col, funcs in aggs.items():\n    for func in funcs:\n        \n        # Get name of aggregation function\n        if isinstance(func, str):\n            func_str = func\n        else:\n            func_str = func.__name__\n            \n        # Name for new column\n        new_col = col + '_' + func_str\n            \n        # Compute this aggregation\n        cards['hist_'+new_col] = hist_trans[col].agg(func).astype('float32')\n        cards['new_'+new_col] = new_trans[col].agg(func).astype('float32')")


# In[15]:


def remove_noninformative(df):
    """Remove non-informative columns (all nan, or all same value)"""
    for col in df:
        if df[col].isnull().all():
            print('Removing '+col+' (all NaN)')
            del df[col]
        elif df[col].nunique()<2:
            print('Removing '+col+' (only 1 unique value)')
            del df[col]

remove_noninformative(cards)
gc.collect()


# In[16]:


cards.info()


# In[17]:


# Test data
test = cards['target'].isnull()
X_test = cards[test].copy()
del X_test['target']

# Training data
y_train = cards.loc[~test, 'target'].copy()
X_train = cards[~test].copy()
del X_train['target']

# Save file w/ all features
cards.reset_index(inplace=True)
cards.to_feather('card_features_all.feather')

# Clean up 
del cards
gc.collect()


# In[18]:


def mutual_information(xi, yi, res=20):
    """Compute the mutual information between two vectors"""
    ix = ~(np.isnan(xi) | np.isinf(xi) | np.isnan(yi) | np.isinf(yi))
    x = xi[ix]
    y = yi[ix]
    N, xe, ye = np.histogram2d(x, y, res)
    Nx, _ = np.histogram(x, xe)
    Ny, _ = np.histogram(y, ye)
    N = N / len(x) #normalize
    Nx = Nx / len(x)
    Ny = Ny / len(y)
    Ni = np.outer(Nx, Ny)
    Ni[Ni == 0] = np.nan
    N[N == 0] = np.nan
    return np.nansum(N * np.log(N / Ni))


# In[19]:


# Show mutual information vs correlation
x = 5*np.random.randn(1000)
y = [x + np.random.randn(1000),
     2*np.sin(x) + np.random.randn(1000),
     x + 10*np.random.randn(1000)]
plt.figure(figsize=(10, 4))
for i in range(3):    
    plt.subplot(1, 3, i+1)
    plt.plot(x, y[i], '.')
    rho, _ = spearmanr(x, y[i])
    plt.title('Mutual info: %0.3f\nCorr coeff: %0.3f'
              % (mutual_information(x, y[i]), rho))
    plt.gca().tick_params(labelbottom=False, labelleft=False)


# In[20]:


def quantile_transform(v, res=101):
    """Quantile-transform a vector to lie between 0 and 1"""
    x = np.linspace(0, 100, res)
    prcs = np.nanpercentile(v, x)
    return np.interp(v, prcs, x/100.0)
    
    
def q_mut_info(x, y):
    """Mutual information between quantile-transformed vectors"""
    return mutual_information(quantile_transform(x),
                              quantile_transform(y))


# In[21]:


"""
%%time

# Compute the mutual information
cols = []
mis = []
for col in X_train:
    mi = q_mut_info(X_train[col], y_train)
    cols.append(col)
    mis.append(mi)
    
# Print mut info of each feature
print_table(['Column', 'Mutual_Information'],
            [cols, mis])
"""


# In[22]:


"""
# Create DataFrame with scores
mi_df = pd.DataFrame()
mi_df['Column'] = cols
mi_df['mut_info'] = mis

# Sort by mutual information
mi_df = mi_df.sort_values('mut_info', ascending=False)
top200 = mi_df.iloc[:200,:]
top200 = top200['Column'].tolist()

# Keep only top 200 columns
X_train = X_train[top200]
X_test = X_test[top200]
"""


# In[23]:


"""
# Regression pipeline
cat_cols = [c for c in X_train if 'mode' in c] 
reg_pipeline = Pipeline([
    ('target_encoder', TargetEncoderCV(cols=cat_cols)),
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', CatBoostRegressor(verbose=False))
])
"""


# In[24]:


"""
%%time 

# Compute the cross-validated feature importance
imp_df = permutation_importance_cv(
    X_train, y_train, reg_pipeline, 'rmse', n_splits=2)
"""


# In[25]:


"""
# Plot the feature importances
plt.figure(figsize=(8, 100))
plot_permutation_importance(imp_df)
plt.show()
"""


# In[26]:


"""
# Get top 100 most important features
df = pd.melt(imp_df, var_name='Feature', value_name='Importance')
dfg = (df.groupby(['Feature'])['Importance']
       .aggregate(np.mean)
       .reset_index()
       .sort_values('Importance', ascending=False))
top100 = dfg['Feature'][:100].tolist()
"""


# In[27]:


"""
# Save file w/ 100 most important features
cards = pd.concat([X_train[top100], X_test[top100]])
cards['target'] = y_train
cards.reset_index(inplace=True)
cards.to_feather('card_features_top100.feather')
"""

