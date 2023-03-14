#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import gc

# Plot settings
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
sns.set()




get_ipython().system('pip install git+http://github.com/brendanhasz/dsutils.git')




from dsutils.printing import print_table, describe_df
from dsutils.eda import countplot2d




# Load card data
dtypes = {
  'card_id':            'str',
  'target':             'float32',
  'first_active_month': 'str',
  'feature_1':          'uint8',
  'feature_2':          'uint8',
  'feature_3':          'uint8',
}
train = pd.read_csv('../input/train.csv',
                    usecols=dtypes.keys(),
                    dtype=dtypes)
del dtypes['target']
test = pd.read_csv('../input/test.csv',
                   usecols=dtypes.keys(),
                   dtype=dtypes)




describe_df(train)




describe_df(test)




# Add target col to test
test['target'] = np.nan

# Merge test and train
cards = pd.concat([train, test])




print('Num unique in train:  ', test['card_id'].nunique())
print('Num unique in test:   ', train['card_id'].nunique())
print('The sum:              ', test['card_id'].nunique()+train['card_id'].nunique())
print('Num unique in merged: ', cards['card_id'].nunique())




del train, test
gc.collect()




cards.sample(10)




cards['card_id'].apply(len).unique()




cards['card_id'].str.slice(5, 15).sample(10)




(cards['card_id']
 .str.slice(5, 15)
 .apply(lambda x: all(e in '0123456789abcdef' for e in x))
 .all())




#cards['card_id'] = cards['card_id'].apply(lambda x: int(x, 16))




# Create a map from card_id to unique int
card_id_map = dict(zip(
    cards['card_id'].values,
    cards['card_id'].astype('category').cat.codes.values
))

# Map the values
cards['card_id'] = cards['card_id'].map(card_id_map).astype('uint32')




cards.sample(10)




# Convert first_active_month to datetime
cards['first_active_month'] = pd.to_datetime(cards['first_active_month'],
                                             format='%Y-%m')




# Make card_id the index
cards.set_index('card_id', inplace=True)
gc.collect()
cards.sample(10)




# Datatypes of each column
dtypes = {
    'city_id':              'int16', 
    'category_1':           'str',
    'merchant_category_id': 'int16',
    'merchant_id':          'str',
    'category_2':           'float16',
    'state_id':             'int8',
    'subsector_id':         'int8'
}

# Load the data
hist_trans = pd.read_csv('../input/historical_transactions.csv', 
                         usecols=dtypes.keys(),
                         dtype=dtypes)
new_trans = pd.read_csv('../input/new_merchant_transactions.csv', 
                        usecols=dtypes.keys(),
                        dtype=dtypes)
merchants = pd.read_csv('../input/merchants.csv', 
                        usecols=dtypes.keys(),
                        dtype=dtypes)

# Merge new_merchant and historical transactions
trans = pd.concat([hist_trans, new_trans])
del hist_trans
del new_trans
gc.collect()




# For each column, count merchant_ids w/ >1 unique vals
gbo = trans.groupby('merchant_id')
nuniques = []
cols = []
for col in trans:
    if col == 'merchant_id': continue
    nuniques.append((gbo[col].nunique() > 1).sum())
    cols.append(col)
print_table(['Column', 'Number unique'], 
            [cols, nuniques])




# Join trans w/ merchants on merchant_id

# Check that all feature_transactions==feature_merchants
df = trans.merge(merchants, how='outer', on='merchant_id', 
                 suffixes=('', '_merchants'))
cols = []
mismatches = []
for col in trans:
    if 'merchant_id' in col: continue
    sames = ((df[col] == df[col+'_merchants']) | 
             (df[col].isnull() & df[col+'_merchants'].isnull())).sum()
    cols.append(col)
    mismatches.append(df.shape[0]-sames)

# Print the number of mismatches
print_table(['Column', 'Num mismatches'],
            [cols, mismatches])
print('Total number of transactions: ', df.shape[0])

# Clean up
del trans, merchants, df, sames
gc.collect()




# Datatypes of each column
# (don't load cols which are in transactions data)
dtypes = {
  'merchant_id':                 'str',
  'merchant_group_id':           'uint32',
  'numerical_1':                 'float32',
  'numerical_2':                 'float32',
  'most_recent_sales_range':     'str',
  'most_recent_purchases_range': 'str',
  'avg_sales_lag3':              'float32',
  'avg_purchases_lag3':          'float32',
  'active_months_lag3':          'uint8',
  'avg_sales_lag6':              'float32',
  'avg_purchases_lag6':          'float32',
  'active_months_lag6':          'uint8',
  'avg_sales_lag12':             'float32',
  'avg_purchases_lag12':         'float32',
  'active_months_lag12':         'uint8',
  'category_4':                  'str',
}

# Load the data
merchants = pd.read_csv('../input/merchants.csv',
                        usecols=dtypes.keys(),
                        dtype=dtypes)




merchants.sample(10)




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




print("Number of duplicate rows in merchants.csv: %d" 
      % merchants.duplicated().sum())




print("Number of duplicate merchant_ids: %d" % 
      merchants['merchant_id'].duplicated().sum())




# Show some of the duplicates
duplicates = merchants['merchant_id'].duplicated(keep=False)
merchants[duplicates].sort_values('merchant_id').head(6)




# Drop duplicate entries
merchants.drop_duplicates(subset='merchant_id',
                          keep='first', inplace=True)




# Datatypes of each column
dtypes = {
    'authorized_flag':      'str',
    'card_id':              'str',
    'city_id':              'int16',
    'category_1':           'str',
    'installments':         'int8',
    'category_3':           'str',
    'merchant_category_id': 'int16',
    'merchant_id':          'str',
    'month_lag':            'int8',
    'purchase_amount':      'float32',
    'purchase_date':        'str',
    'category_2':           'float32',
    'state_id':             'int8',
    'subsector_id':         'int8',
}

# Load the data
hist_trans = pd.read_csv('../input/historical_transactions.csv', 
                         usecols=dtypes.keys(),
                         dtype=dtypes)
new_trans = pd.read_csv('../input/new_merchant_transactions.csv', 
                        usecols=dtypes.keys(),
                        dtype=dtypes)




def preprocess_trans_data(df):
    
    # Convert card_id and merchant_id to numbers
    df['card_id'] = df['card_id'].map(card_id_map).astype('uint32')
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




describe_df(hist_trans)




# Histogram of installments
plt.hist(hist_trans['installments'],
         bins=np.arange(-25.5, 12.5, 1.0))
plt.axvline(x=0, color='k', linestyle='--')
plt.ylabel('Count')
plt.xlabel('installments')
plt.yscale('log', nonposy='clip')
plt.show()




# Set negative installments to nan
hist_trans.loc[hist_trans['installments']<0, 'installments'] = np.nan
new_trans.loc[new_trans['installments']<0, 'installments'] = np.nan




# Histogram of city_id and state_id
plt.subplot(121)
plt.hist(hist_trans['city_id'],
         bins=np.arange(-10.5, 12.5, 1.0))
plt.axvline(x=0, color='k', linestyle='--')
plt.ylabel('Count')
plt.xlabel('city_id')
plt.yscale('log', nonposy='clip')

plt.subplot(122)
plt.hist(hist_trans['state_id'],
         bins=np.arange(-10.5, 12.5, 1.0))
plt.axvline(x=0, color='k', linestyle='--')
plt.ylabel('Count')
plt.xlabel('state_id')
plt.yscale('log', nonposy='clip')

plt.show()




# Set negative ids to nan
hist_trans.loc[hist_trans['city_id']<0, 'city_id'] = np.nan
new_trans.loc[new_trans['city_id']<0, 'city_id'] = np.nan
hist_trans.loc[hist_trans['state_id']<0, 'state_id'] = np.nan
new_trans.loc[new_trans['state_id']<0, 'state_id'] = np.nan




describe_df(cards)




# Loyalty score (aka the target)
cards['target'].hist(bins=50)
plt.xlabel('Loyalty score')
plt.ylabel('Number of cards')
plt.title('Loyalty Score Distribution')
plt.show()




print('Percent of cards with values <20: %0.3f'
      % (100*(cards['target']<-20).sum() /
              cards['target'].notnull().sum()))




cards.loc[cards['target']<-20, 'target'].unique()




# Show a log ratio distribution
a = np.random.rand(10000)
b = np.random.rand(10000)
c = np.log(a/b)
plt.hist(c, bins=np.linspace(-35, 20, 50))
plt.title('Log Ratio Distribution')
plt.show()




np.log(1e-14)




# Plot first_active_month distribution
fam = cards['first_active_month'].value_counts()
fam.plot()
plt.ylabel('Number of Cards')
plt.xlabel('first_active_month')
plt.show()




# first_active_month vs loyalty score
sns.lineplot(x='first_active_month', y='target', data=cards)
plt.show()




# category counts
plt.figure(figsize=(6.4, 10))
plt.subplot(311)
sns.countplot(x='feature_1', data=cards)
plt.subplot(312)
sns.countplot(x='feature_2', data=cards)
plt.subplot(313)
sns.countplot(x='feature_3', data=cards)
plt.tight_layout()
plt.show()




# Features vs loyalty score with std dev
plt.figure(figsize=(6.4, 10))
plt.subplot(311)
sns.barplot(x='feature_1', y='target', data=cards)
plt.subplot(312)
sns.barplot(x='feature_2', y='target', data=cards)
plt.subplot(313)
sns.barplot(x='feature_3', y='target', data=cards)
plt.tight_layout()
plt.show()




# Features vs loyalty score with std dev
plt.figure(figsize=(6.4, 10))
plt.subplot(311)
sns.barplot(x='feature_1', y='target',
            data=cards, ci='sd')
plt.subplot(312)
sns.barplot(x='feature_2', y='target',
            data=cards, ci='sd')
plt.subplot(313)
sns.barplot(x='feature_3', y='target',
            data=cards, ci='sd')
plt.tight_layout()
plt.show()




describe_df(merchants)




# Show number of merchants per merchant group
mpmg = merchants['merchant_group_id'].value_counts().values
plt.plot(np.arange(len(mpmg))+1, mpmg)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Unique merchant group')
plt.ylabel('Number of merchants')
plt.show()




# Raw distributions for numerical cols
plt.figure()
plt.subplot(121)
merchants['numerical_1'].hist()
plt.yscale('log')
plt.ylabel('Number of merchants')
plt.xlabel('numerical_1')
plt.subplot(122)
merchants['numerical_2'].hist()
plt.yscale('log')
plt.xlabel('numerical_2')
plt.tight_layout()
plt.show()




rho, pv = spearmanr(merchants['numerical_1'].values,
                    merchants['numerical_2'].values)
print('Correlation coefficient = %0.3f' % rho)




(merchants['numerical_1'] == merchants['numerical_2']).mean()




# most_recent_sales_range and most_recent_purchases_range
plt.figure()
plt.subplot(121)
sns.countplot(x='most_recent_sales_range',
              data=merchants)
plt.yscale('log')
plt.subplot(122)
sns.countplot(x='most_recent_purchases_range',
              data=merchants)
plt.yscale('log')
plt.tight_layout()
plt.show()




rho, pv = spearmanr(merchants['most_recent_sales_range'].values,
                    merchants['most_recent_purchases_range'].values)
print('Correlation coefficient = %0.3f' % rho)




# Show joint counts
countplot2d('most_recent_sales_range',
            'most_recent_purchases_range',
            merchants, log=True)




def plothist3(df, cols, bins=30):
    plt.figure(figsize=(6.4, 10))
    for i, lab in enumerate(cols):
        plt.subplot(3, 1, i+1)
        merchants[lab].hist(bins=bins)
        plt.xlabel(lab)
    plt.tight_layout()
    plt.show()
    
plothist3(merchants, 
          ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12'],
          bins=np.linspace(-1, 3, 41))




plothist3(merchants, 
          ['avg_purchases_lag3',
           'avg_purchases_lag6',
           'avg_purchases_lag12'],
          bins=np.linspace(-1, 3, 41))




plothist3(merchants, 
          ['active_months_lag3',
           'active_months_lag6',
           'active_months_lag12'],
          bins=np.linspace(0.5, 12.5, 13))




# category_4
sns.countplot(x='category_4', data=merchants)
plt.show()




# Merge new and historical transactions
trans = pd.concat([hist_trans, new_trans])

# Show info about the merged table
describe_df(trans)




# authorized_flag
sns.countplot(x='authorized_flag', data=trans)
plt.title('Number of authorized transactions')
plt.show()




# card_id
plt.plot(trans['card_id'].value_counts().values)
plt.xlabel('Unique card_id')
plt.ylabel('Number of transactions')
plt.title('Number of transactions per card_id')
plt.show()




# city_id
plt.figure(figsize=(5, 40))
sns.countplot(y='city_id', data=trans,
              order=trans['city_id'].value_counts().index)
plt.yticks(fontsize=8)
plt.xscale('log')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.title('Number of transactions per City')
plt.show()




# state_id
plt.figure(figsize=(8, 4))
sns.countplot(x='state_id', data=trans,
              order=trans['state_id'].value_counts().index)
plt.ylabel('Count')
plt.xlabel('state_id')
plt.yscale('log')
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.title('Number of transactions per State')
plt.show()




# category_1
sns.countplot(x='category_1', data=trans)
plt.show()




# category_2
sns.countplot(x='category_2', data=hist_trans)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.show()




# category_3
sns.countplot(x='category_3', data=trans)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.show()




# Joint distribution of authorized_flag vs category_1
N, e1, e2 = np.histogram2d(trans['authorized_flag'], 
                           trans['category_1'], 
                           bins=[[-0.5, 0.5, 1.5], [-0.5, 0.5, 1.5]])
sns.heatmap(N.astype('int64'), annot=True, fmt='d')
plt.xlabel('authorized_flag')
plt.ylabel('category_1')




# installments
plt.hist(trans['installments'], bins=np.arange(-0.5, 12.5, 1.0))
plt.axvline(x=0, color='k', linestyle='--')
plt.ylabel('Count')
plt.xlabel('installments')
plt.yscale('log', nonposy='clip')
plt.show()




# merchant_category_id
plt.figure(figsize=(5, 40))
sns.countplot(y='merchant_category_id', data=trans,
              order=trans['merchant_category_id'].value_counts().index)
plt.yticks(fontsize=8)
plt.gca().set(xscale='log')
plt.show()




# subsector_id
plt.hist(hist_trans['subsector_id'], bins=np.arange(0.5, 41.5, 1.0))
plt.ylabel('Count')
plt.xlabel('subsector_id')
plt.yscale('log')
plt.show()




# merchant_id
plt.plot(np.arange(1, trans['merchant_id'].nunique()+1),
         trans['merchant_id'].value_counts().values)
plt.xlabel('Unique merchant_id')
plt.ylabel('Number of transactions')
plt.title('Number of transactions per merchant_id')
plt.xscale('log')
plt.yscale('log')
plt.show()




# purchase_amount
counts, be = np.histogram(hist_trans['purchase_amount'],
                          bins=np.arange(-1, 6010605, 5))
plt.plot(be[:-1]-min(be[:-1])+1, counts)
plt.ylabel('Count')
plt.xlabel('purchase_amount')
plt.xscale('log')
plt.yscale('log')
plt.show()




# Function to plot transactions over time
def transactions_over_time(df):
    tpd = pd.DataFrame()
    tpd['Transactions'] = (
        df['purchase_date'].dt.year*10000 +                    
        df['purchase_date'].dt.month*100 +                    
        df['purchase_date'].dt.day
    ).value_counts()
    tpd['Date'] = pd.to_datetime(tpd.index, format='%Y%m%d')
    tpd.plot('Date', 'Transactions')
    plt.ylabel('Number of transactions')
    plt.show()
    
# purchase_date
transactions_over_time(trans)




# purchase_date
transactions_over_time(hist_trans)




# purchase_date
transactions_over_time(new_trans)




counts = (trans['purchase_date']
          .dt.dayofweek.value_counts())
plt.bar(x=counts.index, height=counts.values)
plt.ylabel('Number of transactions')
plt.xticks(range(7), ['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'])
plt.show()




# month_lag
plt.hist(trans['month_lag'], bins=np.arange(-13.5, 0.5, 1.0))
plt.ylabel('Count')
plt.xlabel('month_lag')
plt.yscale('log')
plt.show()

