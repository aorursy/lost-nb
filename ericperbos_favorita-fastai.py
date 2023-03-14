#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')




from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
import datetime

PATH = '../input/'




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", PATH]).decode("utf8"))

# Any results you write to the current directory are saved as output.




table_names = ['train', 'stores', 'items', 'transactions', 
               'holidays_events', 'oil', 'test', 'sample_submission']




tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]




from IPython.display import HTML




for t in tables: display(t.head(), t.shape)




# The following returns summarized aggregate information to each table accross each field.
for t in tables: display(DataFrameSummary(t).summary())




train, stores, items, transactions, holidays_events, oil, test, sample_submission = tables




len(train),len(test)

train.onpromotion = train.onpromotion!='0'
test.onpromotion = test.onpromotion!='0'


add_datepart(train, "date", drop=False)

%add_datepart


add_datepart(transactions, "date", drop=False)




add_datepart(holidays_events, "date", drop=False)
add_datepart(oil, "date", drop=False)
add_datepart(test, "date", drop=False)




for t in tables: display(t.head(), t.shape)




# If done on all train data, results in 125m rows. So, we're taking a small sample of the last 8 weeks:
train_mask_10w = (train['date'] >= '2016-06-28') & (train['date'] <= '2016-08-31')
print(train.shape)




train =  train[train_mask_10w]
print(train.shape)




train.head()




transactions_mask_10w = (transactions['date'] >= '2016-06-28') & (transactions['date'] <= '2016-08-31')
print(transactions.shape)




transactions =  transactions[transactions_mask_10w]
print(transactions.shape)




transactions.head()




holidays_events_mask_10w = (holidays_events['date'] >= '2016-06-28') & (holidays_events['date'] <= '2016-08-31')
print(holidays_events.shape)




holidays_events =  holidays_events[holidays_events_mask_10w]
print(holidays_events.shape)




holidays_events.head()




oil_mask_10w = (oil['date'] >= '2016-06-28') & (oil['date'] <= '2016-08-31')
print(oil.shape)




oil =  oil[oil_mask_10w]
print(oil.shape)




oil.head()




def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))




joined = join_df(train, stores, "store_nbr")
len(joined[joined.type.isnull()])




joined.head()




joined_test = join_df(test, stores, "store_nbr")
len(joined_test[joined_test.type.isnull()])




joined = join_df(joined, items, "item_nbr")
len(joined[joined.family.isnull()])




joined.head()




joined_test = join_df(joined_test, items, "item_nbr")
len(joined_test[joined_test.family.isnull()])




joined = join_df(joined, transactions, ["date", "store_nbr"] )
len(joined[joined.store_nbr.isnull()])




joined_test = join_df(joined_test, transactions, ["date", "store_nbr"] )
len(joined_test[joined_test.store_nbr.isnull()])









# we drop the duplicate columns ending with _y
for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns: df.drop(c, inplace=True, axis=1)




joined.head()




joined.describe()




joined_test.head()




# Check if any NANs
joined.isnull().values.any()




# Check if any NANs (slower, more complete)
joined.isnull().sum().sum()

def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v: last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1).astype(int))
    df[pre+fld] = res




%add_datepart


# Look at all columns pivoted to rows
joined.head().T.head(40)




# dropping "Elasped" as it generates an error later, due to crazy 10 digits
joined.drop(['Elapsed'],axis = 1, inplace = True)




joined.head().T.head(40)




cat_vars = ['store_nbr', 'item_nbr', 'onpromotion', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']




contin_vars = ['transactions']




n = len(joined); n




for v in cat_vars:
    joined[v] = joined[v].astype('category').cat.as_ordered()




for v in cat_vars:
    joined_test[v] = joined_test[v].astype('category').cat.as_ordered()

for v in contin_vars:
    joined[v] = joined[v].astype('float32')
    joined_test[v] = joined_test[v].astype('float32')


for v in contin_vars:
    joined[v] = joined[v].astype('float32')




dep = 'unit_sales'
joined = joined[cat_vars+contin_vars+[dep, 'date']].copy()




joined_test[dep] = 0
joined_test = joined_test[cat_vars+contin_vars+[dep, 'date', 'id']].copy()




joined.head().T.head(40)




joined_test.head().T.head(40)

# check this cell function ?
apply_cats(joined_test, joined)


idxs = get_cv_idxs(n)
joined_samp = joined.iloc[idxs].set_index("date")
samp_size = len(joined_samp)
samp_size




joined_samp = joined.set_index("date")




samp_size = len(joined_samp)
samp_size




joined_samp.head()




joined_samp.tail()

%proc_df
[x, y, nas, mapper(optional)]:

    x: x is the transformed version of df. x will not have the response variable
        and is entirely numeric.

    y: y is the response variable

    nas (handles missing values): returns a dictionary of which nas it created, and the associated median.

    mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continous
    variables which is then used for scaling of during test-time.



df, y, nas, mapper = proc_df(joined_samp, 'unit_sales', do_scale=True)




yl = np.log(y)




# df is now a entirely numeric dataframe, without the "unit sales" columns
df.head()




# y contains the "unit sales" now
y




min(y)




yl




max(y)




np.isnan(y).any()




np.isnan(y).




joined_test = joined_test.set_index("date")




joined_test.head()




# joined_test.drop(['transactions'], axis = 1, inplace = True)




df_test, _, nas, mapper = proc_df(joined_test, 'unit_sales', do_scale=True, skip_flds=['transactions'],
                                  na_dict=nas)




#ratio of .754 is 16 days by 65 days, to be close to real test duration
train_ratio = 0.754
# train_ratio = 0.9
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))

val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2016,8,16)) & (df.index>=datetime.datetime(2016,8,31)))


len(val_idx)




samp_size




1 - (len(val_idx)/ samp_size)









#from Rossmann
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(y)
y_range = (0, max_log_y*1.2)

# Favorita
# Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE)
# https://www.kaggle.com/tkm2261/eval-metric-and-evaluating-last-year-sales-bench
WEIGHTS = 
def NWRMSLE(y, pred):
    y = y.clip(0, y.max())
    pred = pred.clip(0, pred.max())
    score = np.nansum(WEIGHTS * ((np.log1p(pred) - np.log1p(y)) ** 2)) / WEIGHTS.sum()
    return np.sqrt(score)


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=512,
                                       test_df=df_test)




cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]




cat_sz




emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]




emb_szs




m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3




m.lr_find()




m.sched.plot(100)






m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3


m.fit(lr, 3, metrics=[exp_rmspe])





















