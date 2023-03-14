#!/usr/bin/env python
# coding: utf-8

# In[1]:


from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb


# In[2]:


CAL_DTYPES={"event_name_1": "category", 
            "event_name_2": "category", 
            "event_type_1": "category", 
            "event_type_2": "category", 
            "weekday": "category", 
            'wm_yr_wk': 'int16', 
            "wday": "int16",
            "month": "int16", 
            "year": "int16", 
            "snap_CA": "float32", 
            'snap_TX': 'float32', 
            'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", 
                "item_id": "category", 
                "wm_yr_wk": "int16",
                "sell_price":"float32" }


# In[3]:


pd.options.display.max_columns = 50


# In[4]:


NDAYS = 28
MAX_LAGS = 57
TRAINING_LAST = 1913
FDAY = datetime(2016,4, 25) 
FDAY


# In[5]:


def create_calendar_df():
    calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            calendar_df[col] = calendar_df[col].cat.codes.astype("int16")
            calendar_df[col] -= calendar_df[col].min()
    
    return calendar_df


def create_prices_df():
    prices_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices_df[col] = prices_df[col].cat.codes.astype("int16")
            prices_df[col] -= prices_df[col].min()
    
    return prices_df
    

def create_dt(is_train = True, nrows = None, first_day = 1200):
    start_day = max(1 if is_train else TRAINING_LAST - MAX_LAGS, first_day)
    numeric_cols = [f"d_{day}" for day in range(start_day, TRAINING_LAST + 1)]
    category_cols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numeric_cols} 
    dtype.update({col: "category" for col in category_cols if col != "id"})
    sales_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 
                     nrows = nrows, usecols = category_cols + numeric_cols, dtype = dtype)
    
    for col in category_cols:
        if col != "id":
            sales_df[col] = sales_df[col].cat.codes.astype("int16")
            sales_df[col] -= sales_df[col].min()
    
    if not is_train:
        for day in range(TRAINING_LAST + 1, TRAINING_LAST + NDAYS +1):
            sales_df[f"d_{day}"] = np.nan
    
    sales_df = pd.melt(sales_df,
                  id_vars = category_cols,
                  value_vars = [col for col in sales_df.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    calendar_df = create_calendar_df()
    prices_df = create_prices_df()
    sales_df = sales_df.merge(calendar_df, on= "d", copy = False)
    sales_df = sales_df.merge(prices_df, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return sales_df


# In[6]:


FIRST_DAY = 350 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !


# In[7]:


get_ipython().run_cell_magic('time', '', '\ndf = create_dt(is_train=True, first_day= FIRST_DAY)\ndf.shape')


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


def create_features(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


# In[11]:


get_ipython().run_cell_magic('time', '', '\ncreate_features(df)\ndf.shape')


# In[12]:


df.info()


# In[13]:


df.head()


# In[14]:


df.dropna(inplace = True)
df.shape


# In[15]:


cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
cols_to_remove = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(cols_to_remove)]
X_train = df[train_cols]
y_train = df["sales"]


# In[16]:


get_ipython().run_cell_magic('time', '', "\nnp.random.seed(777)\n\nfake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)\ntrain_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)\ntrain_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], \n                         categorical_feature=cat_feats, free_raw_data=False)\nfake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],\n                              categorical_feature=cat_feats,\n                 free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!")


# In[17]:


del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()


# In[18]:


params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}


# In[19]:


get_ipython().run_cell_magic('time', '', '\nm_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) ')


# In[20]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 4)
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(12,8))
lgb.plot_importance(m_lgb, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# In[21]:


m_lgb.save_model("model.lgb")


# In[22]:


def create_lag_features_for_test(dt, day):
    # create lag feaures just for single day (faster)
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt.date == day, lag_col] =             dt.loc[dt.date ==day-timedelta(days=lag), 'sales'].values  # !!! main

    windows = [7, 28]
    for window in windows:
        for lag in lags:
            df_window = dt[(dt.date <= day-timedelta(days=lag)) & (dt.date > day-timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt.date==day,'id'])
            dt.loc[dt.date == day,f"rmean_{lag}_{window}"] =                 df_window_grouped.sales.values     


# In[23]:


def create_date_features_for_test(dt):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")


# In[24]:


get_ipython().run_cell_magic('time', '', '\nalphas = [1.028, 1.023, 1.018]\nweights = [1/len(alphas)]*len(alphas)  # equal weights\n\nte0 = create_dt(False)  # create master copy of `te`\ncreate_date_features_for_test (te0)\n\nfor icount, (alpha, weight) in enumerate(zip(alphas, weights)):\n    te = te0.copy()  # just copy\n    cols = [f"F{i}" for i in range(1, 29)]\n\n    for tdelta in range(0, 28):\n        day = FDAY + timedelta(days=tdelta)\n        print(tdelta, day.date())\n        tst = te[(te.date >= day - timedelta(days=MAX_LAGS))\n                 & (te.date <= day)].copy()\n#         create_fea(tst)  # correct, but takes much time\n        create_lag_features_for_test(tst, day)  # faster  \n        tst = tst.loc[tst.date == day, train_cols]\n        te.loc[te.date == day, "sales"] = \\\n            alpha * m_lgb.predict(tst)  # magic multiplier by kyakovlev\n\n    te_sub = te.loc[te.date >= FDAY, ["id", "sales"]].copy()\n\n    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")[\n        "id"].cumcount()+1]\n    te_sub = te_sub.set_index(["id", "F"]).unstack()[\n        "sales"][cols].reset_index()\n    te_sub.fillna(0., inplace=True)\n    te_sub.sort_values("id", inplace=True)\n    te_sub.reset_index(drop=True, inplace=True)\n    te_sub.to_csv(f"submission_{icount}.csv", index=False)\n    if icount == 0:\n        sub = te_sub\n        sub[cols] *= weight\n    else:\n        sub[cols] += te_sub[cols]*weight\n    print(icount, alpha, weight)')


# In[25]:


sub.head(10)


# In[26]:


sub.id.nunique(), sub["id"].str.contains("validation$").sum()


# In[27]:


sub.shape


# In[28]:


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission.csv",index=False)

