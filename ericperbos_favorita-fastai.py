#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
import datetime

PATH = '../input/'


# In[3]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", PATH]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[4]:


table_names = ['train', 'stores', 'items', 'transactions', 
               'holidays_events', 'oil', 'test', 'sample_submission']


# In[5]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[6]:


from IPython.display import HTML


# In[7]:


for t in tables: display(t.head(), t.shape)


# In[8]:


# The following returns summarized aggregate information to each table accross each field.
for t in tables: display(DataFrameSummary(t).summary())


# In[9]:


train, stores, items, transactions, holidays_events, oil, test, sample_submission = tables


# In[10]:


len(train),len(test)

train.onpromotion = train.onpromotion!='0'
test.onpromotion = test.onpromotion!='0'
# In[11]:


add_datepart(train, "date", drop=False)

%add_datepart
# In[12]:


add_datepart(transactions, "date", drop=False)


# In[13]:


add_datepart(holidays_events, "date", drop=False)
add_datepart(oil, "date", drop=False)
add_datepart(test, "date", drop=False)


# In[14]:


for t in tables: display(t.head(), t.shape)


# In[15]:


# If done on all train data, results in 125m rows. So, we're taking a small sample of the last 8 weeks:
train_mask_10w = (train['date'] >= '2016-06-28') & (train['date'] <= '2016-08-31')
print(train.shape)


# In[16]:


train =  train[train_mask_10w]
print(train.shape)


# In[17]:


train.head()


# In[18]:


transactions_mask_10w = (transactions['date'] >= '2016-06-28') & (transactions['date'] <= '2016-08-31')
print(transactions.shape)


# In[19]:


transactions =  transactions[transactions_mask_10w]
print(transactions.shape)


# In[20]:


transactions.head()


# In[21]:


holidays_events_mask_10w = (holidays_events['date'] >= '2016-06-28') & (holidays_events['date'] <= '2016-08-31')
print(holidays_events.shape)


# In[22]:


holidays_events =  holidays_events[holidays_events_mask_10w]
print(holidays_events.shape)


# In[23]:


holidays_events.head()


# In[24]:


oil_mask_10w = (oil['date'] >= '2016-06-28') & (oil['date'] <= '2016-08-31')
print(oil.shape)


# In[25]:


oil =  oil[oil_mask_10w]
print(oil.shape)


# In[26]:


oil.head()


# In[27]:


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))


# In[28]:


joined = join_df(train, stores, "store_nbr")
len(joined[joined.type.isnull()])


# In[29]:


joined.head()


# In[30]:


joined_test = join_df(test, stores, "store_nbr")
len(joined_test[joined_test.type.isnull()])


# In[31]:


joined = join_df(joined, items, "item_nbr")
len(joined[joined.family.isnull()])


# In[32]:


joined.head()


# In[33]:


joined_test = join_df(joined_test, items, "item_nbr")
len(joined_test[joined_test.family.isnull()])


# In[34]:


joined = join_df(joined, transactions, ["date", "store_nbr"] )
len(joined[joined.store_nbr.isnull()])


# In[35]:


joined_test = join_df(joined_test, transactions, ["date", "store_nbr"] )
len(joined_test[joined_test.store_nbr.isnull()])


# In[36]:





# In[36]:


# we drop the duplicate columns ending with _y
for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns: df.drop(c, inplace=True, axis=1)


# In[37]:


joined.head()


# In[38]:


joined.describe()


# In[39]:


joined_test.head()


# In[40]:


# Check if any NANs
joined.isnull().values.any()


# In[41]:


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
# In[42]:




%add_datepart
# In[42]:


# Look at all columns pivoted to rows
joined.head().T.head(40)


# In[43]:


# dropping "Elasped" as it generates an error later, due to crazy 10 digits
joined.drop(['Elapsed'],axis = 1, inplace = True)


# In[44]:


joined.head().T.head(40)


# In[45]:


cat_vars = ['store_nbr', 'item_nbr', 'onpromotion', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']


# In[46]:


contin_vars = ['transactions']


# In[47]:


n = len(joined); n


# In[48]:


for v in cat_vars:
    joined[v] = joined[v].astype('category').cat.as_ordered()


# In[49]:


for v in cat_vars:
    joined_test[v] = joined_test[v].astype('category').cat.as_ordered()

for v in contin_vars:
    joined[v] = joined[v].astype('float32')
    joined_test[v] = joined_test[v].astype('float32')
# In[50]:


for v in contin_vars:
    joined[v] = joined[v].astype('float32')


# In[51]:


dep = 'unit_sales'
joined = joined[cat_vars+contin_vars+[dep, 'date']].copy()


# In[52]:


joined_test[dep] = 0
joined_test = joined_test[cat_vars+contin_vars+[dep, 'date', 'id']].copy()


# In[53]:


joined.head().T.head(40)


# In[54]:


joined_test.head().T.head(40)

# check this cell function ?
apply_cats(joined_test, joined)
# In[55]:


idxs = get_cv_idxs(n)
joined_samp = joined.iloc[idxs].set_index("date")
samp_size = len(joined_samp)
samp_size


# In[56]:


joined_samp = joined.set_index("date")


# In[57]:


samp_size = len(joined_samp)
samp_size


# In[58]:


joined_samp.head()


# In[59]:


joined_samp.tail()

%proc_df
[x, y, nas, mapper(optional)]:

    x: x is the transformed version of df. x will not have the response variable
        and is entirely numeric.

    y: y is the response variable

    nas (handles missing values): returns a dictionary of which nas it created, and the associated median.

    mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continous
    variables which is then used for scaling of during test-time.

# In[60]:


df, y, nas, mapper = proc_df(joined_samp, 'unit_sales', do_scale=True)


# In[61]:


yl = np.log(y)


# In[62]:


# df is now a entirely numeric dataframe, without the "unit sales" columns
df.head()


# In[63]:


# y contains the "unit sales" now
y


# In[64]:


min(y)


# In[65]:


yl


# In[66]:


max(y)


# In[67]:


np.isnan(y).any()


# In[68]:


np.isnan(y).


# In[69]:


joined_test = joined_test.set_index("date")


# In[70]:


joined_test.head()


# In[71]:


# joined_test.drop(['transactions'], axis = 1, inplace = True)


# In[72]:


df_test, _, nas, mapper = proc_df(joined_test, 'unit_sales', do_scale=True, skip_flds=['transactions'],
                                  na_dict=nas)


# In[73]:


#ratio of .754 is 16 days by 65 days, to be close to real test duration
train_ratio = 0.754
# train_ratio = 0.9
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))

val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2016,8,16)) & (df.index>=datetime.datetime(2016,8,31)))
# In[74]:


len(val_idx)


# In[75]:


samp_size


# In[76]:


1 - (len(val_idx)/ samp_size)


# In[77]:





# In[77]:


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
# In[78]:


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=512,
                                       test_df=df_test)


# In[79]:


cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]


# In[80]:


cat_sz


# In[81]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# In[82]:


emb_szs


# In[83]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3


# In[84]:


m.lr_find()


# In[85]:


m.sched.plot(100)


# In[86]:




m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3
# In[86]:


m.fit(lr, 3, metrics=[exp_rmspe])


# In[87]:





# In[87]:





# In[87]:





# In[87]:




